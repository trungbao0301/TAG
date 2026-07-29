import base64
import json
import os
import socket
import time

import cv2
import gym
import numpy as np
from ament_index_python.packages import get_package_share_directory

from cyberrunner_dreamer_thomas import cyberrunner_layout_custom
from cyberrunner_dreamer_thomas.path import LinearPath


def _recv_line(sock):
    data = bytearray()
    while True:
        b = sock.recv(1)
        if not b:
            raise ConnectionError("TCP bridge disconnected")
        if b == b"\n":
            return data.decode("utf-8")
        data.extend(b)


def _parse_checkpoint_ranges(value):
    ranges = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split("-", 1)
        start = int(parts[0])
        end = int(parts[1]) if len(parts) == 2 else start
        ranges.append((min(start, end), max(start, end)))
    return ranges


class CyberrunnerGym(gym.Env):
    def __init__(
        self,
        repeat=1,
        layout=cyberrunner_layout_custom.cyberrunner_dxf_layout,
        num_rel_path=5,
        num_wait_steps=30,
        reward_on_fail=0.0,
        reward_on_goal=0.5,
    ):
        super().__init__()

        self.observation_space = gym.spaces.Dict(
            image=gym.spaces.Box(0, 255, (64, 64, 1), np.uint8),
            states=gym.spaces.Box(-np.inf, np.inf, (4,), np.float32),
            goal=gym.spaces.Box(-np.inf, np.inf, (num_rel_path * 2,), np.float32),
            progress=gym.spaces.Box(-np.inf, np.inf, (1,), np.float32),
            log_reward=gym.spaces.Box(-np.inf, np.inf, (1,), np.float32),
            log_elapsed_sec=gym.spaces.Box(0.0, np.inf, (1,), np.float32),
            log_success=gym.spaces.Box(0.0, 1.0, (1,), np.float32),
        )
        self.action_space = gym.spaces.Box(-1.0, 1.0, (2,))
        self.obs = dict(self.observation_space.sample())

        self.num_rel_path = num_rel_path
        board_size = np.array(
            [layout["board_width"], layout["board_height"]], dtype=np.float32
        )
        self.norm_max = np.array(
            [10 * np.pi / 180.0, 10 * np.pi / 180.0, *board_size]
        )
        self.goal_norm_max = np.array(
            [0.0002 * 60 * k for k in range(1, self.num_rel_path + 1) for _ in range(2)]
        )

        self.offset = board_size / 2.0
        self.holes = np.asarray(layout["holes"], dtype=np.float32)
        self.danger_zones = layout.get("danger_zones", [])
        self.danger_lines = layout.get("danger_lines", [])
        self.segment_guards = layout.get("segment_guards", [])
        self.hole_detect_radius = float(
            os.environ.get("CYBERRUNNER_HOLE_DETECT_RADIUS_M", "0.005")
        )
        self.hole_detect_frames = max(
            1, int(os.environ.get("CYBERRUNNER_HOLE_DETECT_FRAMES", "3"))
        )
        self.hole_detect_count = 0
        self.ball_in_hole = False
        self.ball_hole_index = -1
        self.ball_hole_distance = np.inf
        shared = get_package_share_directory("cyberrunner_dreamer_thomas")
        self.p = LinearPath.load(os.path.join(shared, "path_custom.pkl"))

        self.cheat = False
        self.policy_mode = os.environ.get("CYBERRUNNER_POLICY_MODE", "guarded").lower()
        self.legacy_tcp_policy = self.policy_mode in ("legacy", "legacy_tcp", "old")
        self.legacy_shortcut_restrict = (
            os.environ.get("CYBERRUNNER_LEGACY_SHORTCUT_RESTRICT", "false").lower()
            in ("1", "true", "yes", "on")
        )
        default_cheat_threshold_m = "0.057" if self.legacy_tcp_policy else "0.10"
        cheat_threshold_m = float(
            os.environ.get("CYBERRUNNER_CHEAT_THRESHOLD_M", default_cheat_threshold_m)
        )
        self.cheat_threshold = int(cheat_threshold_m / self.p.distance)
        self.cheat_threshold_target_dt = 1.0 / float(
            os.environ.get("CYBERRUNNER_CHEAT_THRESHOLD_TARGET_FPS", "35.0")
        )
        self.cheat_threshold_max_scale = float(
            os.environ.get("CYBERRUNNER_CHEAT_THRESHOLD_MAX_SCALE", "2.0")
        )
        self.cheat_threshold_scale = 1.0
        self.progress_spike_count = 0
        self.max_progress_spikes = int(os.environ.get("CYBERRUNNER_MAX_PROGRESS_SPIKES", "8"))
        self.local_path_back = int(
            float(os.environ.get("CYBERRUNNER_LOCAL_PATH_BACK_M", "0.04"))
            / self.p.distance
        )
        self.local_path_forward = int(
            float(os.environ.get("CYBERRUNNER_LOCAL_PATH_FORWARD_M", "0.16"))
            / self.p.distance
        )
        self.enforce_checkpoint_order = (
            os.environ.get("CYBERRUNNER_ENFORCE_CHECKPOINT_ORDER", "true").lower()
            not in ("0", "false", "no")
        )
        self.checkpoint_radius = float(os.environ.get("CYBERRUNNER_CHECKPOINT_RADIUS_M", "0.010"))
        self.checkpoint_skip_tolerance = int(
            float(os.environ.get("CYBERRUNNER_CHECKPOINT_SKIP_TOL_M", "0.02"))
            / self.p.distance
        )
        self.enforce_path_corridor = (
            os.environ.get("CYBERRUNNER_ENFORCE_PATH_CORRIDOR", "true").lower()
            not in ("0", "false", "no")
        )
        self.path_corridor_radius = float(
            os.environ.get("CYBERRUNNER_PATH_CORRIDOR_RADIUS_M", "0.018")
        )
        self.waypoint_path_indices = self._make_waypoint_path_indices()
        print(
            f"[TCP ENV] cheat_threshold={self.cheat_threshold} "
            f"path points ({cheat_threshold_m:.3f} m), "
            f"policy_mode={self.policy_mode}, "
            f"legacy_shortcut_restrict={self.legacy_shortcut_restrict}, "
            f"dt_scale_max={self.cheat_threshold_max_scale:.2f}, "
            f"max_progress_spikes={self.max_progress_spikes}, "
            f"local_path_window=[-{self.local_path_back}, +{self.local_path_forward}], "
            f"checkpoint_order={self.enforce_checkpoint_order}, "
            f"checkpoint_radius={self.checkpoint_radius:.3f} m, "
            f"path_corridor={self.enforce_path_corridor}, "
            f"path_corridor_radius={self.path_corridor_radius:.3f} m, "
            f"hole_detect_radius={self.hole_detect_radius:.3f} m, "
            f"hole_detect_frames={self.hole_detect_frames}"
        )
        self.prev_pos_path = 0
        self.next_checkpoint = 1
        self.last_step_next_checkpoint = 1
        self.checkpoint_skip = False
        self.path_shortcut = False
        self.path_shortcut_distance = 0.0
        self.num_wait_steps = num_wait_steps
        default_reward_on_fail = str(reward_on_fail) if self.legacy_tcp_policy else "-0.10"
        self.reward_on_fail = float(
            os.environ.get("CYBERRUNNER_REWARD_ON_FAIL", default_reward_on_fail)
        )
        self.reward_on_hole = float(os.environ.get("CYBERRUNNER_REWARD_ON_HOLE", "-0.20"))
        self.reward_on_shortcut = float(
            os.environ.get("CYBERRUNNER_REWARD_ON_SHORTCUT", "-0.20")
        )
        self.reward_on_danger = float(
            os.environ.get("CYBERRUNNER_REWARD_ON_DANGER", "-0.40")
        )
        self.reward_on_checkpoint = float(
            os.environ.get("CYBERRUNNER_REWARD_ON_CHECKPOINT", "0.03")
        )
        self.reward_on_goal = float(os.environ.get("CYBERRUNNER_REWARD_ON_GOAL", str(reward_on_goal)))
        self.stuck_window_sec = max(
            0.0, float(os.environ.get("CYBERRUNNER_STUCK_WINDOW_SEC", "5.0"))
        )
        self.stuck_radius_m = max(
            0.0, float(os.environ.get("CYBERRUNNER_STUCK_RADIUS_M", "0.003"))
        )
        self.stuck_penalty = min(
            0.0, float(os.environ.get("CYBERRUNNER_STUCK_PENALTY", "0.0"))
        )
        self.stuck_anchor_pos = None
        self.stuck_since = None
        self.stuck_events = 0
        print(
            f"[TCP ENV] rewards: fail={self.reward_on_fail:.3f}, "
            f"hole={self.reward_on_hole:.3f}, "
            f"shortcut={self.reward_on_shortcut:.3f}, "
            f"danger={self.reward_on_danger:.3f}, "
            f"checkpoint={self.reward_on_checkpoint:.3f}, "
            f"goal={self.reward_on_goal:.3f}"
        )
        print(
            "[TCP ENV] Stuck penalty: "
            f"window={self.stuck_window_sec:.1f}s, "
            f"radius={self.stuck_radius_m * 1000.0:.1f}mm, "
            f"penalty={self.stuck_penalty:.3f} per window"
        )
        print(
            f"[TCP ENV] segment_guards={len(self.segment_guards)}, "
            f"danger_zones={len(self.danger_zones)}, "
            f"danger_lines={len(self.danger_lines)}"
        )

        self.ball_detected = False
        self.ball_occluded = False
        self.ball_missing_since = None
        self.ball_missing_grace_sec = 0.0
        self.ball_loss_reported = False
        self.ball_loss_grace_sec = max(
            0.0, float(os.environ.get("CYBERRUNNER_BALL_LOSS_GRACE_SEC", "0.35"))
        )
        self.occlusion_grace_sec = max(
            self.ball_loss_grace_sec,
            float(os.environ.get("CYBERRUNNER_OCCLUSION_GRACE_SEC", "1.50")),
        )
        self.occlusion_checkpoint_ranges = _parse_checkpoint_ranges(
            os.environ.get("CYBERRUNNER_OCCLUSION_CHECKPOINT_RANGES", "")
        )
        self.last_valid_obs = None
        self.last_valid_ball_pos = None
        self.last_valid_ball_time = None
        self.ball_velocity = np.zeros(2, dtype=np.float32)
        self.ball_prediction_max_speed = max(
            0.0,
            float(os.environ.get("CYBERRUNNER_BALL_PREDICTION_MAX_SPEED_MPS", "0.15")),
        )
        print(
            f"[TCP ENV] ball_loss_grace={self.ball_loss_grace_sec:.2f}s, "
            f"occlusion_grace={self.occlusion_grace_sec:.2f}s, "
            f"occlusion_checkpoint_ranges={self.occlusion_checkpoint_ranges or 'none'}"
        )
        default_max_angle_vel = "200" if self.legacy_tcp_policy else "180"
        self.max_angle_vel = float(
            os.environ.get("CYBERRUNNER_MAX_ANGLE_VEL", default_max_angle_vel)
        )
        self.alpha_fac = -1.0
        self.beta_fac = -1.0

        self.last_time = 0
        self.progress = 0
        self.accum_reward = 0.0
        self.steps = 0
        self.off_path = False
        self.episodes = 0
        self.success = False
        self.in_danger_zone = False
        self.danger_zone_name = ""
        self.danger_zone_distance = np.inf
        self.last_ball_pos = None
        self.segment_guard_hit = False
        self.segment_guard_name = ""
        self.episode_start_time = time.monotonic()

        host = os.environ.get("CYBERRUNNER_TCP_BIND", "0.0.0.0")
        port = int(os.environ.get("CYBERRUNNER_TCP_PORT", "5555"))

        print(f"[TCP ENV] Waiting for PC bridge on {host}:{port} ...")
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((host, port))
        self.server.listen(1)
        self.sock, addr = self.server.accept()
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"[TCP ENV] PC bridge connected from {addr}")

    def _request(self, obj):
        msg = json.dumps(obj, separators=(",", ":")).encode("utf-8") + b"\n"
        self.sock.sendall(msg)
        line = _recv_line(self.sock)
        return json.loads(line)

    def step(self, action):
        self.steps += 1

        if self.cheat and (self.p.num_points - self.prev_pos_path <= 200):
            action[1] = 1.0

        obs = self._step_remote(action)

        now = time.time()
        step_dt = now - self.last_time
        self.last_time = now
        if step_dt > 0.0:
            self.cheat_threshold_scale = min(
                max(step_dt / self.cheat_threshold_target_dt, 1.0),
                self.cheat_threshold_max_scale,
            )
        else:
            self.cheat_threshold_scale = 1.0

        reward = self._get_reward(obs)
        done = self._get_done(obs)
        elapsed_sec = time.monotonic() - self.episode_start_time

        if done and not self.success:
            if self.ball_in_hole:
                reward = self.reward_on_hole
            elif self.in_danger_zone or self.segment_guard_hit:
                reward = self.reward_on_danger
            elif self.checkpoint_skip or self.path_shortcut:
                reward = self.reward_on_shortcut
            else:
                reward = self.reward_on_fail
        if self.success:
            reward += self.reward_on_goal

        if done or self.steps == 3000:
            print("Reset board")
            self._reset_board()

        info = {"is_terminal": False} if self.success else {}

        if step_dt > (1.0 / 35.0):
            step_fps = 1.0 / step_dt if step_dt > 0.0 else 0.0
            print(f"Slower than 35fps: step_dt={step_dt * 1000.0:.1f} ms, fps={step_fps:.1f}")

        self.accum_reward += reward if not done else 0
        obs["states"] = (obs["states"] / self.norm_max).astype(np.float32)
        obs["goal"] = (obs["goal"] / self.goal_norm_max).astype(np.float32)
        obs["progress"] = np.asarray([1 + self.prev_pos_path], dtype=np.float32)
        obs["log_reward"] = np.asarray([reward if not done else 0], dtype=np.float32)
        obs["log_elapsed_sec"] = np.asarray([elapsed_sec], dtype=np.float32)
        obs["log_success"] = np.asarray([float(self.success)], dtype=np.float32)
        return obs, reward, done, info

    def reset(self):
        print("Resetting ...")
        self.episodes += 1
        print("Previous reward: {}".format(self.accum_reward))
        print("Previous episode length: {}".format(self.steps / 60.0))
        print("Episodes: {}".format(self.episodes))

        self.accum_reward = 0.0
        self.steps = 0
        self.success = False
        self.ball_detected = False
        self.ball_occluded = False
        self.ball_missing_since = None
        self.ball_missing_grace_sec = 0.0
        self.ball_loss_reported = False
        self.last_valid_obs = None
        self.last_valid_ball_pos = None
        self.last_valid_ball_time = None
        self.ball_velocity.fill(0.0)
        self.stuck_anchor_pos = None
        self.stuck_since = None
        self.stuck_events = 0
        self.hole_detect_count = 0
        self.ball_in_hole = False
        self.ball_hole_index = -1
        self.ball_hole_distance = np.inf
        self.checkpoint_skip = False
        self.path_shortcut = False
        self.path_shortcut_distance = 0.0
        self.last_step_next_checkpoint = 1
        self.in_danger_zone = False
        self.danger_zone_name = ""
        self.danger_zone_distance = np.inf
        self.last_ball_pos = None
        self.segment_guard_hit = False
        self.segment_guard_name = ""

        self._send_action(np.zeros((2,)))

        count = 0
        obs = self._get_obs()
        while count < self.num_wait_steps:
            obs = self._get_obs()
            count = count + 1 if self.ball_detected else 0

        self.prev_pos_path = self.p.closest_point(obs["states"][2:4])[0]
        self.last_ball_pos = obs["states"][2:4].copy()
        self.next_checkpoint = self._next_checkpoint_from_path(self.prev_pos_path)
        self.last_step_next_checkpoint = self.next_checkpoint
        self.progress = 0
        self.progress_spike_count = 0
        self.last_time = time.time()
        self.episode_start_time = time.monotonic()

        obs["states"] = (obs["states"] / self.norm_max).astype(np.float32)
        obs["goal"] = (obs["goal"] / self.goal_norm_max).astype(np.float32)
        obs["progress"] = np.asarray([1 + self.prev_pos_path], dtype=np.float32)
        obs["log_reward"] = np.asarray([0], dtype=np.float32)
        obs["log_elapsed_sec"] = np.asarray([0.0], dtype=np.float32)
        obs["log_success"] = np.asarray([0.0], dtype=np.float32)
        return obs

    def render(self, mode="human"):
        pass

    def _step_remote(self, action):
        vel_1, vel_2 = self._action_to_command(action)

        while True:
            rep = self._request({
                "cmd": "step",
                "vel_1": vel_1,
                "vel_2": vel_2,
                "timeout": 0.018,
            })
            if rep.get("ok", False):
                break
            time.sleep(0.01)

        return self._obs_from_reply(rep)

    def _obs_from_reply(self, rep):
        if not rep.get("ball", False):
            return self._obs_for_missing_ball(rep)

        was_occluded = self.ball_occluded
        self.ball_detected = True
        self.ball_occluded = False
        self.ball_missing_since = None
        self.ball_missing_grace_sec = 0.0
        self.ball_loss_reported = False

        states = np.array(
            [
                float(rep["alpha"]),
                float(rep["beta"]),
                float(rep["x_b"]),
                float(rep["y_b"]),
            ],
            dtype=np.float32,
        )
        states[2:] += self.offset
        img_bytes = base64.b64decode(rep["image_b64"])
        image = np.frombuffer(img_bytes, dtype=np.uint8).reshape(64, 64, 1)

        rel_path = self._get_rel_path(states[2:]).astype(np.float32)

        states = np.nan_to_num(states, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        rel_path = np.nan_to_num(rel_path, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        image = np.clip(image, 0, 255).astype(np.uint8)

        self.obs = {"states": states, "goal": rel_path, "image": image}
        self._remember_visible_ball(self.obs)
        if was_occluded:
            print("[Occlusion]: BALL REACQUIRED")
        return self._copy_obs(self.obs)

    @staticmethod
    def _copy_obs(obs):
        return {key: value.copy() for key, value in obs.items()}

    def _active_ball_loss_grace(self):
        for start, end in self.occlusion_checkpoint_ranges:
            if start <= self.next_checkpoint <= end:
                return self.occlusion_grace_sec
        return self.ball_loss_grace_sec

    def _remember_visible_ball(self, obs):
        now = time.monotonic()
        position = obs["states"][2:4].copy()
        if self.last_valid_ball_pos is not None and self.last_valid_ball_time is not None:
            dt = now - self.last_valid_ball_time
            if dt > 1e-4:
                velocity = (position - self.last_valid_ball_pos) / dt
                speed = float(np.linalg.norm(velocity))
                if speed > self.ball_prediction_max_speed > 0.0:
                    velocity *= self.ball_prediction_max_speed / speed
                self.ball_velocity = (
                    0.6 * self.ball_velocity + 0.4 * velocity
                ).astype(np.float32)
        self.last_valid_ball_pos = position
        self.last_valid_ball_time = now
        self.last_valid_obs = self._copy_obs(obs)

    def _obs_for_missing_ball(self, rep):
        now = time.monotonic()
        if self.ball_missing_since is None:
            self.ball_missing_since = now
            self.ball_missing_grace_sec = self._active_ball_loss_grace()
            print(
                "[Occlusion]: BALL HIDDEN "
                f"checkpoint={self.next_checkpoint} "
                f"grace={self.ball_missing_grace_sec:.2f}s"
            )

        missing_sec = now - self.ball_missing_since
        within_grace = (
            self.last_valid_obs is not None
            and missing_sec < self.ball_missing_grace_sec
        )
        self.ball_occluded = within_grace
        self.ball_detected = within_grace
        self.off_path = False
        self.path_shortcut = False
        self.path_shortcut_distance = 0.0

        if self.last_valid_obs is None:
            states = np.zeros(4, dtype=np.float32)
            image = np.zeros((64, 64, 1), dtype=np.uint8)
            rel_path = np.zeros((self.num_rel_path * 2,), dtype=np.float32)
            self.obs = {"states": states, "goal": rel_path, "image": image}
            return self._copy_obs(self.obs)

        obs = self._copy_obs(self.last_valid_obs)

        # Alpha and beta remain available from the bridge even while x/y are NaN.
        for index, key in enumerate(("alpha", "beta")):
            value = float(rep.get(key, obs["states"][index]))
            if np.isfinite(value):
                obs["states"][index] = value

        if within_grace and self.last_valid_ball_pos is not None:
            predicted = (
                self.last_valid_ball_pos
                + self.ball_velocity * min(missing_sec, self.ball_missing_grace_sec)
            )
            path_idx, path_point = self._closest_point(predicted)
            if path_idx >= 0:
                predicted = path_point
            obs["states"][2:4] = predicted
            obs["goal"] = self._get_rel_path(predicted).astype(np.float32)
        elif not self.ball_loss_reported:
            print(f"[Occlusion]: BALL LOSS CONFIRMED after {missing_sec:.2f}s")
            self.ball_loss_reported = True

        self.obs = obs
        return self._copy_obs(self.obs)

    def _send_action(self, action):
        vel_1, vel_2 = self._action_to_command(action)
        self._request({"cmd": "action", "vel_1": vel_1, "vel_2": vel_2})

    def _action_to_command(self, action):
        action = action.copy()
        action *= self.max_angle_vel
        vel_1 = float(self.alpha_fac * action[0])
        vel_2 = float(self.beta_fac * action[1])
        return vel_1, vel_2

    def _get_obs(self):
        while True:
            rep = self._request({"cmd": "obs"})
            if rep.get("ok", False):
                break
            time.sleep(0.01)

        return self._obs_from_reply(rep)

    def _get_reward(self, obs):
        if self.ball_occluded or not self.ball_detected:
            self.stuck_anchor_pos = None
            self.stuck_since = None
            reward = 0.0
        else:
            curr_pos_path, p = self._closest_point(obs["states"][2:4])
            self.off_path = curr_pos_path == -1
            self.path_shortcut = False
            self.path_shortcut_distance = 0.0
            if curr_pos_path != -1:
                self.path_shortcut_distance = float(np.linalg.norm(obs["states"][2:4] - p))
                self.path_shortcut = (
                    self.enforce_path_corridor
                    and (not self.cheat)
                    and self.path_shortcut_distance > self.path_corridor_radius
                )
                if self.path_shortcut:
                    print(
                        "[Shortcut]: outside path corridor "
                        f"dist={self.path_shortcut_distance:.4f} "
                        f"threshold={self.path_corridor_radius:.4f} "
                        f"path={curr_pos_path}"
                    )
            if self.off_path and self.cheat:
                curr_pos_path = self.prev_pos_path

            raw_progress = curr_pos_path - self.prev_pos_path
            if self.path_shortcut:
                self.progress = 0
                return 0.0

            if self.legacy_tcp_policy:
                self.progress = raw_progress
                reward = float(raw_progress) * 0.004 / 16.0
                self.prev_pos_path = curr_pos_path
                return reward + self._stuck_reward(obs["states"][2:4])

            scaled_cheat_threshold = int(self.cheat_threshold * self.cheat_threshold_scale)
            if (not self.cheat) and raw_progress > scaled_cheat_threshold:
                self.progress = 0
                self.progress_spike_count += 1
                reward = 0.0
                print(
                    "[Spike]: Ignoring too high progress "
                    f"(progress={raw_progress}, threshold={scaled_cheat_threshold}, "
                    f"scale={self.cheat_threshold_scale:.2f}, "
                    f"count={self.progress_spike_count}/{self.max_progress_spikes})"
                )
                return reward

            checkpoint_before = self.next_checkpoint
            if self._update_checkpoint_order(obs["states"][2:4], curr_pos_path):
                self.progress = 0
                reward = 0.0
                return reward

            self.progress_spike_count = 0
            self.progress = raw_progress
            checkpoints_passed = self.next_checkpoint - checkpoint_before
            reward = (
                float(raw_progress) * 0.004 / 16.0
                + checkpoints_passed * self.reward_on_checkpoint
            )
            self.prev_pos_path = curr_pos_path
            reward += self._stuck_reward(obs["states"][2:4])
        return reward

    def _stuck_reward(self, ball_pos):
        """Apply one penalty per stationary window without ending the episode."""
        if self.stuck_window_sec <= 0.0 or self.stuck_penalty == 0.0:
            return 0.0

        now = time.monotonic()
        position = np.asarray(ball_pos, dtype=np.float32)
        if self.stuck_anchor_pos is None or self.stuck_since is None:
            self.stuck_anchor_pos = position.copy()
            self.stuck_since = now
            return 0.0

        displacement = float(np.linalg.norm(position - self.stuck_anchor_pos))
        if displacement > self.stuck_radius_m:
            self.stuck_anchor_pos = position.copy()
            self.stuck_since = now
            return 0.0

        stationary_sec = now - self.stuck_since
        if stationary_sec < self.stuck_window_sec:
            return 0.0

        self.stuck_events += 1
        print(
            "[Penalty]: MARBLE STUCK "
            f"event={self.stuck_events}, stationary={stationary_sec:.1f}s, "
            f"movement={displacement * 1000.0:.1f}mm <= "
            f"{self.stuck_radius_m * 1000.0:.1f}mm, "
            f"penalty={self.stuck_penalty:.3f}"
        )
        self.stuck_anchor_pos = position.copy()
        self.stuck_since = now
        return self.stuck_penalty

    def _get_done(self, obs):
        if self.ball_occluded:
            return False

        done = not self.ball_detected
        if done:
            print("[Done]: BALL LOST")

        check_guarded_failures = (not self.legacy_tcp_policy) or self.legacy_shortcut_restrict

        if self.ball_detected and check_guarded_failures:
            if not self.legacy_tcp_policy:
                self._update_ball_in_hole(obs["states"][2:4])
            self._update_danger_zone_state(obs["states"][2:4])
            self._update_segment_guard_state(obs["states"][2:4])
            if (not self.legacy_tcp_policy) and self.ball_in_hole:
                done = True
                print(
                    "[Done]: BALL IN HOLE "
                    f"index={self.ball_hole_index} "
                    f"dist={self.ball_hole_distance:.4f}"
                )
            elif self.in_danger_zone:
                done = True
                print(
                    "[Done]: DANGER ZONE "
                    f"name={self.danger_zone_name} "
                    f"dist={self.danger_zone_distance:.4f}"
                )
            elif self.segment_guard_hit:
                done = True
                print(f"[Done]: SEGMENT GUARD name={self.segment_guard_name}")

        if (not self.cheat) and self.off_path:
            done = True
            print("[Done]: OFFPATH")

        if (not self.cheat) and self.path_shortcut:
            done = True
            print(
                "[Done]: PATH SHORTCUT "
                f"dist={self.path_shortcut_distance:.4f} "
                f"threshold={self.path_corridor_radius:.4f}"
            )

        if self.p.num_points - self.prev_pos_path <= 1:
            self.success = True
            done = True
            print("[Done]: SUCCESS")
            self._send_action(np.array([0.0, 0.0]))

        if self.legacy_tcp_policy and (not self.cheat) and self.progress > self.cheat_threshold:
            done = True
            print("[Done]: Too high progress")

        if (
            (not self.legacy_tcp_policy)
            and (not self.cheat)
            and self.progress_spike_count >= self.max_progress_spikes
        ):
            done = True
            print(
                "[Done]: Repeated too high progress "
                f"(count={self.progress_spike_count}/{self.max_progress_spikes})"
            )

        if (not self.legacy_tcp_policy) and (not self.cheat) and self.checkpoint_skip:
            done = True
            print(f"[Done]: CHECKPOINT SKIP next={self.next_checkpoint}")

        return done

    def _update_ball_in_hole(self, ball_pos):
        distances = np.linalg.norm(self.holes - ball_pos, axis=1)
        hole_index = int(np.argmin(distances))
        hole_distance = float(distances[hole_index])

        self.ball_hole_index = hole_index
        self.ball_hole_distance = hole_distance
        if hole_distance <= self.hole_detect_radius:
            self.hole_detect_count += 1
        else:
            self.hole_detect_count = 0

        self.ball_in_hole = self.hole_detect_count >= self.hole_detect_frames

    def _checkpoint_in_range(self, checkpoint, start_checkpoint, end_checkpoint):
        return start_checkpoint <= checkpoint <= end_checkpoint

    def _segment_distance_to_point(self, start, end, point):
        segment = end - start
        denom = float(np.dot(segment, segment))
        if denom <= 1e-12:
            return float(np.linalg.norm(point - end))
        t = float(np.dot(point - start, segment) / denom)
        t = min(1.0, max(0.0, t))
        closest = start + t * segment
        return float(np.linalg.norm(point - closest))

    def _cross2d(self, a, b):
        return float(a[0] * b[1] - a[1] * b[0])

    def _segments_intersect(self, a, b, c, d):
        ab = b - a
        cd = d - c
        o1 = self._cross2d(ab, c - a)
        o2 = self._cross2d(ab, d - a)
        o3 = self._cross2d(cd, a - c)
        o4 = self._cross2d(cd, b - c)
        return (o1 * o2 < 0.0) and (o3 * o4 < 0.0)

    def _point_line_side(self, point, line_start, line_end):
        return self._cross2d(line_end - line_start, point - line_start)

    def _danger_line_crossed(self, previous, current, line_start, line_end, width):
        if self._segments_intersect(previous, current, line_start, line_end):
            return True, 0.0
        distance = self._segment_distance_to_segment(previous, current, line_start, line_end)
        return distance <= width, distance

    def _segment_distance_to_segment(self, a, b, c, d):
        if self._segments_intersect(a, b, c, d):
            return 0.0
        return min(
            self._segment_distance_to_point(a, b, c),
            self._segment_distance_to_point(a, b, d),
            self._segment_distance_to_point(c, d, a),
            self._segment_distance_to_point(c, d, b),
        )

    def _update_danger_zone_state(self, ball_pos):
        self.in_danger_zone = False
        self.danger_zone_name = ""
        self.danger_zone_distance = np.inf
        current = np.asarray(ball_pos, dtype=np.float32)
        previous = None if self.last_ball_pos is None else np.asarray(self.last_ball_pos, dtype=np.float32)
        if previous is not None:
            for line in self.danger_lines:
                if not self._checkpoint_in_range(
                    self.next_checkpoint,
                    int(line["start_checkpoint"]),
                    int(line["end_checkpoint"]),
                ):
                    continue
                p1 = np.asarray(line["p1"], dtype=np.float32)
                p2 = np.asarray(line["p2"], dtype=np.float32)
                width = float(line.get("width", 0.0))
                crossed, distance = self._danger_line_crossed(previous, current, p1, p2, width)
                if crossed:
                    self.in_danger_zone = True
                    self.danger_zone_name = line.get("name", "danger_line")
                    self.danger_zone_distance = distance
                    print(
                        "[DangerLine]: crossed "
                        f"name={self.danger_zone_name} "
                        f"dist={distance:.4f} width={width:.4f}"
                    )
                    self.last_ball_pos = current
                    return
        for zone in self.danger_zones:
            if not self._checkpoint_in_range(
                self.next_checkpoint,
                int(zone["start_checkpoint"]),
                int(zone["end_checkpoint"]),
            ):
                continue
            center = np.asarray(zone["center"], dtype=np.float32)
            radius = float(zone["radius"])
            point_distance = float(np.linalg.norm(current - center))
            previous_distance = None if previous is None else float(np.linalg.norm(previous - center))
            distance = point_distance
            crossed_through = False
            if previous is not None:
                distance = self._segment_distance_to_point(previous, current, center)
                crossed_through = (
                    previous_distance is not None
                    and previous_distance > radius
                    and point_distance > radius
                    and distance <= radius
                )
            if crossed_through:
                self.in_danger_zone = True
                self.danger_zone_name = zone.get("name", "danger_zone")
                self.danger_zone_distance = distance
                self.last_ball_pos = current
                return
        self.last_ball_pos = current

    def _update_segment_guard_state(self, ball_pos):
        self.segment_guard_hit = False
        self.segment_guard_name = ""
        x, y = float(ball_pos[0]), float(ball_pos[1])
        for guard in self.segment_guards:
            if not self._checkpoint_in_range(
                self.next_checkpoint,
                int(guard["start_checkpoint"]),
                int(guard["end_checkpoint"]),
            ):
                continue
            y_min = float(guard.get("y_min", -np.inf))
            y_max = float(guard.get("y_max", np.inf))
            if not (y_min <= y <= y_max):
                continue
            x_min = guard.get("x_min")
            x_max = guard.get("x_max")
            if x_min is not None and x < float(x_min):
                self.segment_guard_hit = True
            if x_max is not None and x > float(x_max):
                self.segment_guard_hit = True
            if self.segment_guard_hit:
                self.segment_guard_name = guard.get("name", "segment_guard")
                return

    def _make_waypoint_path_indices(self):
        indices = [0]
        total = 0
        for i in range(1, self.p.orig_waypoints.shape[0]):
            segment = self.p.orig_waypoints[i] - self.p.orig_waypoints[i - 1]
            total += int(np.floor(np.linalg.norm(segment) / self.p.distance)) + 1
            indices.append(min(total, self.p.num_points - 1))
        return np.asarray(indices, dtype=np.int32)

    def _next_checkpoint_from_path(self, path_idx):
        if path_idx < 0:
            return 1
        passed = int(np.searchsorted(self.waypoint_path_indices, path_idx, side="right") - 1)
        return min(max(passed + 1, 1), len(self.waypoint_path_indices) - 1)

    def _update_checkpoint_order(self, ball_pos, curr_pos_path):
        if (not self.enforce_checkpoint_order) or self.next_checkpoint >= len(self.waypoint_path_indices):
            return False

        while self.next_checkpoint < len(self.waypoint_path_indices):
            waypoint = self.p.orig_waypoints[self.next_checkpoint]
            dist = float(np.linalg.norm(ball_pos - waypoint))
            target_idx = int(self.waypoint_path_indices[self.next_checkpoint])

            if curr_pos_path >= target_idx:
                print(
                    f"[Checkpoint]: passed {self.next_checkpoint} "
                    f"path={curr_pos_path} target={target_idx} dist={dist:.4f}"
                )
                self.next_checkpoint += 1
                continue

            return False

        return False

    def _closest_point(self, point):
        if self.steps <= 0 or self.prev_pos_path < 0:
            return self.p.closest_point(point)

        start = max(0, self.prev_pos_path - self.local_path_back)
        end = min(self.p.num_points, self.prev_pos_path + self.local_path_forward + 1)
        if end <= start:
            return self.p.closest_point(point)

        local_points = self.p.points[start:end]
        local_idx = int(np.argmin(np.linalg.norm(local_points - point, axis=1)))
        idx = start + local_idx
        return idx, self.p.points[idx]

    def _get_rel_path(self, point):
        idx = self._closest_point(point)[0]
        if idx == -1:
            return np.zeros((self.num_rel_path * 2,), dtype=np.float32)
        indices = np.clip(
            np.arange(idx, idx + self.num_rel_path * 60, 60),
            0,
            self.p.num_points - 1,
        )
        return (self.p.points[indices] - point).flatten().astype(np.float32)

    def _reset_board(self):
        self._request({"cmd": "reset"})
