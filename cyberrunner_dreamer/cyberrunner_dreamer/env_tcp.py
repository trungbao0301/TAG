import base64
import json
import os
import socket
import time

import cv2
import gym
import numpy as np
from ament_index_python.packages import get_package_share_directory

from cyberrunner_dreamer import cyberrunner_layout_custom
from cyberrunner_dreamer.path import LinearPath


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


def _parse_xy_zones(value):
    """Parse semicolon-separated xmin:xmax:ymin:ymax zones in board meters."""
    zones = []
    for item in value.split(";"):
        item = item.strip()
        if not item:
            continue
        values = [float(part.strip()) for part in item.split(":")]
        if len(values) != 4:
            raise ValueError(
                "CYBERRUNNER_OCCLUSION_XY_ZONES entries must be "
                "xmin:xmax:ymin:ymax"
            )
        x0, x1, y0, y1 = values
        zones.append((min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)))
    return zones


def _load_xy_zones_from_file(path):
    if not path:
        return []
    with open(os.path.expanduser(path), "r", encoding="utf-8") as stream:
        payload = json.load(stream)
    zones = []
    for zone in payload.get("zones", []):
        x0 = float(zone["xmin_m"])
        x1 = float(zone["xmax_m"])
        y0 = float(zone["ymin_m"])
        y1 = float(zone["ymax_m"])
        zones.append((min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)))
    return zones


class CyberrunnerGym(gym.Env):
    def __init__(
        self,
        repeat=1,
        layout=cyberrunner_layout_custom.cyberrunner_dxf_layout,
        num_rel_path=5,
        num_wait_steps=30,
        reward_on_fail=-0.20,
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
        shared = get_package_share_directory("cyberrunner_dreamer")
        self.p = LinearPath.load(os.path.join(shared, "path_custom.pkl"))

        # Keep the same convention as env.py: normal training has cheat=False,
        # so impossible progress is guarded. Setting this True is reserved for
        # the explicit debug/cheat path used by the original environment.
        self.cheat = False
        self.anti_cheat_max_step_m = max(
            0.0,
            float(os.environ.get("CYBERRUNNER_ANTICHEAT_MAX_STEP_M", "0.057")),
        )
        self.anti_cheat_max_step_points = max(
            1, int(self.anti_cheat_max_step_m / self.p.distance)
        )
        self.cheat_threshold = self.anti_cheat_max_step_points
        self.anti_cheat_penalty = float(
            os.environ.get("CYBERRUNNER_ANTICHEAT_PENALTY", "-0.50")
        )
        self.anti_cheat_triggered = False

        # A single implausible jump is far more often a bad state estimate than
        # an actual cheat: the marble detector can flip to a false lock tens of
        # mm away for a frame or two. Measured marble speed is p50 0.024 m/s /
        # p95 0.122 m/s, so a 58 mm single-step jump is not physical motion --
        # it is a measurement glitch, and terminating the episode with a
        # negative penalty trains the policy on a camera fault.
        #
        # So require the far position to PERSIST before calling it a cheat. A
        # real cheat (marble physically moved) stays at the new spot, so it
        # still trips after this many consecutive steps; a one-frame detector
        # flip reverts and resets the counter.
        # Anti-cheat TERMINATION is off by default on this board. A wall-crossing
        # check over all 157 wall segments found no place where the marble can hop
        # <=60 mm and skip >=50 mm of path: every near-self-approach of the path
        # (e.g. 81.8% passing 23.5 mm from 46.4%) is blocked by a wall. So a
        # progress shortcut is physically impossible here and every trigger this
        # ever produced was a bad state estimate, not a cheat.
        #
        # Implausible jumps are STILL detected and still denied progress credit,
        # so a detector glitch cannot earn reward -- it just no longer kills the
        # episode or applies a penalty. Set CYBERRUNNER_ANTICHEAT_ENABLED=1 to
        # restore termination (e.g. on a board where shortcuts are reachable).
        self.anti_cheat_enabled = str(
            os.environ.get("CYBERRUNNER_ANTICHEAT_ENABLED", "0")
        ).strip().lower() not in ("0", "false", "no", "off", "")
        # Sized against recorded speeds: runs above 1.0 m/s lasting >=5 steps
        # occur once per ~7500 samples (~1 per 160 episodes), versus once per
        # ~2900 for >=3 steps. Sustained runs cluster, so 3 was too few.
        self.anti_cheat_confirm_steps = max(
            1, int(os.environ.get("CYBERRUNNER_ANTICHEAT_CONFIRM_STEPS", "5"))
        )
        self.implausible_jump_steps = 0
        # Plausible top speed, used to scale the jump budget by the ACTUAL step
        # time. Measured marble speed is p50 0.024 / p95 0.122 / p99 0.277 m/s,
        # so 1.0 m/s is ~4x the p99 and only 0.29% of steps exceed it. The old
        # fixed 0.057 m/step meant 2.0 m/s at 35 fps but only 0.86 m/s at the
        # 15 fps this rig dips to -- both far too loose to catch a 58 mm
        # detector flip, which is why glitches were scored as real progress.
        self.anti_cheat_max_speed_mps = max(
            0.0,
            float(os.environ.get("CYBERRUNNER_ANTICHEAT_MAX_SPEED_MPS", "1.0")),
        )
        # Floor so a very fast frame cannot make the budget absurdly tight.
        # 10 mm matches the checkpoint pass radius and is a bit under one marble
        # diameter (12 mm), so a jump under the floor cannot skip a checkpoint.
        # Note this only binds above ~100 fps; at the 15-56 fps this pipeline
        # actually runs, 1.0 m/s x step_dt (18-66 mm) always dominates.
        self.anti_cheat_min_step_m = max(
            0.0,
            float(os.environ.get("CYBERRUNNER_ANTICHEAT_MIN_STEP_M", "0.010")),
        )
        # Set when the marble is reacquired after an occlusion gap. The marble
        # kept rolling while it was hidden, but prev_pos_path is from before the
        # gap, so the whole gap's travel would be charged to one step.
        self.resync_progress_after_gap = False

        self.path_tolerance = max(
            0.0,
            float(
                os.environ.get(
                    "CYBERRUNNER_PATH_TOLERANCE_M",
                    str(layout.get("ball_radius", 0.006)),
                )
            ),
        )
        self.prev_pos_path = 0
        self.num_wait_steps = num_wait_steps
        self.reward_on_fail = float(
            os.environ.get("CYBERRUNNER_REWARD_ON_FAIL", str(reward_on_fail))
        )
        self.reward_on_goal = reward_on_goal
        self.timeout_steps = max(
            1, int(os.environ.get("CYBERRUNNER_TIMEOUT_STEPS", "3000"))
        )
        self.timeout_penalty = float(
            os.environ.get(
                "CYBERRUNNER_TIMEOUT_PENALTY", str(self.reward_on_fail)
            )
        )
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

        self.ball_detected = False
        self.ball_occluded = False
        self.ball_missing_since = None
        self.ball_missing_grace_sec = 0.0
        self.ball_loss_reported = False
        self.ball_loss_grace_sec = max(
            0.0, float(os.environ.get("CYBERRUNNER_BALL_LOSS_GRACE_SEC", "0.0"))
        )
        self.occlusion_grace_sec = max(
            self.ball_loss_grace_sec,
            float(os.environ.get("CYBERRUNNER_OCCLUSION_GRACE_SEC", "0.0")),
        )
        self.occlusion_checkpoint_ranges = _parse_checkpoint_ranges(
            os.environ.get("CYBERRUNNER_OCCLUSION_CHECKPOINT_RANGES", "")
        )
        # Coordinates use the same lower-left-origin board frame as states[2:4].
        # Example: "0.10:0.14:0.08:0.12;0.20:0.23:0.15:0.18".
        self.occlusion_zones_file = os.environ.get(
            "CYBERRUNNER_OCCLUSION_ZONES_FILE", ""
        )
        self.occlusion_xy_zones = _load_xy_zones_from_file(
            self.occlusion_zones_file
        ) or _parse_xy_zones(
            os.environ.get("CYBERRUNNER_OCCLUSION_XY_ZONES", "")
        )
        self.last_valid_obs = None
        self.last_valid_ball_pos = None
        self.last_valid_ball_time = None
        self.ball_velocity = np.zeros(2, dtype=np.float32)
        self.ball_prediction_max_speed = max(
            0.0,
            float(os.environ.get("CYBERRUNNER_BALL_PREDICTION_MAX_SPEED_MPS", "0.15")),
        )
        self.max_angle_vel = float(os.environ.get("CYBERRUNNER_MAX_ANGLE_VEL", "240"))
        self.alpha_fac = float(os.environ.get("CYBERRUNNER_ALPHA_FAC", "-1.0"))
        self.beta_fac = float(os.environ.get("CYBERRUNNER_BETA_FAC", "-1.0"))
        self.max_cmd_1 = float(os.environ.get("CYBERRUNNER_MAX_CMD_1", "240"))
        self.max_cmd_2 = float(os.environ.get("CYBERRUNNER_MAX_CMD_2", "240"))
        self.action_repeat = max(1, int(os.environ.get("CYBERRUNNER_ACTION_REPEAT", "1")))
        self.rest_after_sec = float(os.environ.get("CYBERRUNNER_REST_AFTER_SEC", "3600"))
        self.rest_duration_sec = float(
            os.environ.get("CYBERRUNNER_REST_DURATION_SEC", "240")
        )
        self.rest_window_start = time.monotonic()

        self.last_time = 0
        self.progress = 0
        self.accum_reward = 0.0
        self.steps = 0
        self.episodes = 0
        self.success = False
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
        print(f"[TCP ENV] Action repeat: {self.action_repeat}")
        print(
            "[TCP ENV] Geometric path fallback tolerance: "
            f"{self.path_tolerance * 1000.0:.1f} mm"
        )
        print(
            "[TCP ENV] Rest policy: "
            f"after {self.rest_after_sec:.0f}s, pause {self.rest_duration_sec:.0f}s "
            "at episode boundary"
        )
        print(
            "[TCP ENV] Episode timeout: "
            f"steps={self.timeout_steps}, penalty={self.timeout_penalty:.3f}"
        )
        print(
            "[TCP ENV] Stuck penalty: "
            f"window={self.stuck_window_sec:.1f}s, "
            f"radius={self.stuck_radius_m * 1000.0:.1f}mm, "
            f"penalty={self.stuck_penalty:.3f} per window"
        )
        print(
            f"[TCP ENV] ball_loss_grace={self.ball_loss_grace_sec:.2f}s, "
            f"occlusion_grace={self.occlusion_grace_sec:.2f}s, "
            f"occlusion_checkpoint_ranges={self.occlusion_checkpoint_ranges or 'none'}, "
            f"occlusion_xy_zones={self.occlusion_xy_zones or 'none'}, "
            f"occlusion_zones_file={self.occlusion_zones_file or 'none'}"
        )
        print(
            "[TCP ENV] progress anti-cheat: "
            f"cheat={self.cheat}, "
            f"anticheat_termination="
            f"{'ON' if self.anti_cheat_enabled else 'OFF'}, "
            f"max_single_step={self.anti_cheat_max_step_m:.3f}m, "
            f"max_speed={self.anti_cheat_max_speed_mps:.2f}m/s, "
            f"confirm_steps={self.anti_cheat_confirm_steps}, "
            f"penalty={self.anti_cheat_penalty:.3f}"
        )

    def _request(self, obj):
        msg = json.dumps(obj, separators=(",", ":")).encode("utf-8") + b"\n"
        self.sock.sendall(msg)
        line = _recv_line(self.sock)
        return json.loads(line)

    def step(self, action):
        self.steps += 1

        obs = self._step_remote(action)

        reward = self._get_reward(obs)
        done = self._get_done(obs)
        timed_out = not done and self.steps >= self.timeout_steps
        if timed_out:
            done = True
            print(
                f"[Done]: TIMEOUT after {self.steps} steps; "
                f"penalty={self.timeout_penalty:.3f}"
            )
            self._send_action(np.zeros((2,), dtype=np.float32))
        elapsed_sec = time.monotonic() - self.episode_start_time

        if done and not self.success:
            if self.anti_cheat_triggered:
                reward = self.anti_cheat_penalty
            elif timed_out:
                reward = self.timeout_penalty
            else:
                reward = self.reward_on_fail
        if self.success:
            reward += self.reward_on_goal

        if done:
            if self.success:
                time.sleep(2)
            print("Reset board")
            self._reset_board()
            self._rest_if_due()

        info = {"is_terminal": False} if self.success or timed_out else {}
        if self.anti_cheat_triggered:
            info["anti_cheat"] = True

        now = time.time()
        step_dt = now - self.last_time
        if step_dt > (1.0 / 35.0):
            step_fps = 1.0 / step_dt if step_dt > 0.0 else 0.0
            print(f"Slower than 35fps: step_dt={step_dt * 1000.0:.1f} ms, fps={step_fps:.1f}")
        self.last_time = now

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
        self.anti_cheat_triggered = False
        self.implausible_jump_steps = 0
        self.resync_progress_after_gap = False
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

        self._send_action(np.zeros((2,)))

        count = 0
        path_idx = -1
        obs = self._get_obs()
        while count < self.num_wait_steps:
            obs = self._get_obs()
            count = count + 1 if self.ball_detected else 0
            if not self.ball_detected:
                time.sleep(0.02)

        path_idx = self._closest_point(obs["states"][2:4])[0]
        if path_idx == -1:
            distances = np.linalg.norm(self.p.points - obs["states"][2:4], axis=1)
            path_idx = int(np.argmin(distances))
        self.prev_pos_path = path_idx
        self.progress = 0
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

        rep = None
        for _ in range(self.action_repeat):
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

        rel_path = self._get_rel_path(states[2:]).flatten().astype(np.float32)

        states = np.nan_to_num(states, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        rel_path = np.nan_to_num(rel_path, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        image = np.clip(image, 0, 255).astype(np.uint8)

        self.obs = {"states": states, "goal": rel_path, "image": image}
        self._remember_visible_ball(self.obs)
        if was_occluded:
            # Re-reference progress on the next reward so the travel that
            # happened while hidden is not charged to a single step.
            self.resync_progress_after_gap = True
            print("[Occlusion]: BALL REACQUIRED")
        return self._copy_obs(self.obs)

    @staticmethod
    def _copy_obs(obs):
        return {key: value.copy() for key, value in obs.items()}

    def _checkpoint_for_occlusion(self):
        waypoint_path_indices = [0]
        total = 0
        for index in range(1, self.p.orig_waypoints.shape[0]):
            segment = self.p.orig_waypoints[index] - self.p.orig_waypoints[index - 1]
            total += int(np.floor(np.linalg.norm(segment) / self.p.distance)) + 1
            waypoint_path_indices.append(min(total, self.p.num_points - 1))
        return int(
            np.searchsorted(waypoint_path_indices, self.prev_pos_path, side="right")
        )

    def _active_ball_loss_grace(self):
        checkpoint = self._checkpoint_for_occlusion()
        if self.last_valid_ball_pos is not None:
            x, y = (float(value) for value in self.last_valid_ball_pos)
            for index, (x0, x1, y0, y1) in enumerate(self.occlusion_xy_zones):
                if x0 <= x <= x1 and y0 <= y <= y1:
                    return self.occlusion_grace_sec, checkpoint, f"xy_zone={index + 1}"
        for start, end in self.occlusion_checkpoint_ranges:
            if start <= checkpoint <= end:
                return self.occlusion_grace_sec, checkpoint, "checkpoint_zone"
        return self.ball_loss_grace_sec, checkpoint, "outside_zone"

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
            (
                self.ball_missing_grace_sec,
                checkpoint,
                grace_source,
            ) = self._active_ball_loss_grace()
            print(
                "[Occlusion]: BALL HIDDEN "
                f"checkpoint={checkpoint} "
                f"source={grace_source} "
                f"grace={self.ball_missing_grace_sec:.2f}s"
            )

        missing_sec = now - self.ball_missing_since
        within_grace = (
            self.last_valid_obs is not None
            and missing_sec < self.ball_missing_grace_sec
        )
        self.ball_occluded = within_grace
        self.ball_detected = within_grace

        if self.last_valid_obs is None:
            states = np.zeros(4, dtype=np.float32)
            image = np.zeros((64, 64, 1), dtype=np.uint8)
            rel_path = np.zeros((self.num_rel_path * 2,), dtype=np.float32)
            self.obs = {"states": states, "goal": rel_path, "image": image}
            return self._copy_obs(self.obs)

        obs = self._copy_obs(self.last_valid_obs)

        # The bridge still reports board angles when marble coordinates are NaN.
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
            obs["goal"] = self._get_rel_path(predicted).flatten().astype(np.float32)
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
        vel_1 = float(np.clip(vel_1, -self.max_cmd_1, self.max_cmd_1))
        vel_2 = float(np.clip(vel_2, -self.max_cmd_2, self.max_cmd_2))
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
            if curr_pos_path == -1:
                self.progress = 0
                return 0.0
            # Reacquired after an occlusion gap: adopt the current position as
            # the new reference instead of charging the gap's travel to one step.
            if self.resync_progress_after_gap:
                self.resync_progress_after_gap = False
                self.implausible_jump_steps = 0
                self.prev_pos_path = curr_pos_path
                self.progress = 0
                return self._stuck_reward(obs["states"][2:4])

            raw_step_progress = curr_pos_path - self.prev_pos_path
            # Scale the budget by the real step time (never below the fixed one).
            # last_time is 0 until the first reset and can be stale across one,
            # so ignore implausible dt and fall back to the fixed budget.
            step_dt = time.time() - self.last_time
            if not (0.0 < step_dt <= 1.0):
                step_dt = 0.0
            if step_dt > 0.0 and self.anti_cheat_max_speed_mps > 0.0:
                # Never looser than the old fixed budget, never tighter than the
                # floor. On slow frames this relaxes; on normal frames it is
                # tighter than 0.057 m, which is what actually catches a flip.
                allowed_m = min(
                    self.anti_cheat_max_step_m,
                    max(
                        self.anti_cheat_min_step_m,
                        self.anti_cheat_max_speed_mps * step_dt,
                    ),
                )
            else:
                allowed_m = self.anti_cheat_max_step_m
            allowed_points = max(1, int(allowed_m / self.p.distance))
            single_step_triggered = raw_step_progress > allowed_points

            if (not self.cheat) and single_step_triggered:
                self.implausible_jump_steps += 1
                self.progress = 0
                confirmed = (
                    self.anti_cheat_enabled
                    and self.implausible_jump_steps
                    >= self.anti_cheat_confirm_steps
                )
                if confirmed:
                    verdict = "terminating episode"
                elif self.anti_cheat_enabled:
                    verdict = (
                        f"strike {self.implausible_jump_steps}/"
                        f"{self.anti_cheat_confirm_steps}, "
                        "treating as a bad state estimate, ignoring this step"
                    )
                else:
                    verdict = (
                        "termination disabled, ignoring this step "
                        "(no progress credit)"
                    )
                print(
                    "[AntiCheat]: IMPLAUSIBLE PROGRESS JUMP "
                    "reason=single_step_distance "
                    f"from={100.0 * self.prev_pos_path / max(1, self.p.num_points - 1):.1f}% "
                    f"to={100.0 * curr_pos_path / max(1, self.p.num_points - 1):.1f}% "
                    f"single_step={raw_step_progress * self.p.distance:.3f}m "
                    f"threshold={allowed_m:.3f}m (step_dt={step_dt * 1000.0:.1f}ms); "
                    + verdict
                )
                if confirmed:
                    self.anti_cheat_triggered = True
                # Leave prev_pos_path alone so a one-frame detector flip that
                # reverts next step resumes scoring from the real position.
                return 0.0

            self.implausible_jump_steps = 0
            self.progress = curr_pos_path - self.prev_pos_path
            reward = float(curr_pos_path - self.prev_pos_path) * 0.004 / 16.0
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
            print(
                "[Done]: BALL LOST - immediate terminal; "
                f"penalty={self.reward_on_fail:.3f}"
            )
            self._send_action(np.zeros((2,), dtype=np.float32))

        if self.anti_cheat_triggered:
            done = True
            print(
                "[Done]: ANTI-CHEAT PROGRESS JUMP; "
                f"penalty={self.anti_cheat_penalty:.3f}"
            )
            self._send_action(np.zeros((2,), dtype=np.float32))

        if (
            not self.anti_cheat_triggered
            and self.p.num_points - self.prev_pos_path <= 1
        ):
            self.success = True
            done = True
            print("[Done]: SUCCESS")
            self._send_action(np.array([0.0, 0.0]))

        return done

    def _closest_point(self, point):
        """Return a path point, repairing sparse invalid cells near the path.

        The serialized custom path contains a precomputed closest-point grid.
        Some valid corridor cells in that grid are -1, including cells only a
        few millimeters from START. Keep valid grid results unchanged, but use
        the geometric nearest point when it is within the physical tolerance.
        """
        idx, closest = self.p.closest_point(point)
        if idx != -1 or self.path_tolerance <= 0.0:
            return idx, closest

        distances = np.linalg.norm(self.p.points - point, axis=1)
        nearest_idx = int(np.argmin(distances))
        if float(distances[nearest_idx]) <= self.path_tolerance:
            return nearest_idx, self.p.points[nearest_idx]
        return -1, None

    def _get_rel_path(self, point):
        idx = self._closest_point(point)[0]
        if idx == -1:
            return np.zeros((self.num_rel_path, 2), dtype=np.float32)

        indices = np.clip(
            np.arange(idx, idx + self.num_rel_path * 60, 60),
            0,
            self.p.num_points - 1,
        )
        return np.asarray(self.p.points[indices] - point, dtype=np.float32)

    def _reset_board(self):
        self._request({"cmd": "reset"})

    def _rest_if_due(self):
        if self.rest_after_sec <= 0.0 or self.rest_duration_sec <= 0.0:
            return

        now = time.monotonic()
        elapsed = now - self.rest_window_start
        if elapsed < self.rest_after_sec:
            return

        print(
            "[Rest]: training window reached "
            f"{elapsed / 60.0:.1f} min; pausing "
            f"{self.rest_duration_sec / 60.0:.1f} min at episode boundary"
        )
        self._send_action(np.zeros((2,), dtype=np.float32))
        time.sleep(self.rest_duration_sec)
        self.rest_window_start = time.monotonic()
        self.last_time = time.time()
        print("[Rest]: resume training")
