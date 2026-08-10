import base64
import json
import os
import socket
import time

import gym
import numpy as np

from tag_dreamer.path import LinearPath
from tag_dreamer.random_path import generate_waypoints

BOARD_WIDTH_M = 0.259
BOARD_HEIGHT_M = 0.229
BALL_RADIUS_M = 0.006


def _recv_line(sock):
    data = bytearray()
    while True:
        b = sock.recv(1)
        if not b:
            raise ConnectionError("TCP bridge disconnected")
        if b == b"\n":
            return data.decode("utf-8")
        data.extend(b)


class BallPlateGym(gym.Env):
    """Bare ball-on-plate: no maze, no fixed map. A fresh random path is
    generated every episode (see random_path.py), so the policy has to track
    a path's shape instead of memorizing the one corridor a fixed map would
    give it.

    Speaks the exact TCP protocol tag_dreamer.env_tcp.TagGym speaks, so
    tools/lite_board_server.py (or Isaac, or the real robot bridge) can drive
    it unmodified -- sim and real stay interchangeable the same way they
    already are for the maze task.
    """

    def __init__(
        self,
        board_width=BOARD_WIDTH_M,
        board_height=BOARD_HEIGHT_M,
        ball_radius=BALL_RADIUS_M,
        num_rel_path=5,
        num_wait_steps=30,
        reward_on_fail=-0.20,
        reward_on_goal=2.0,
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

        self.num_rel_path = num_rel_path
        self.board_width = board_width
        self.board_height = board_height
        self.ball_radius = ball_radius
        board_size = np.array([board_width, board_height], dtype=np.float32)
        self.norm_max = np.array([10 * np.pi / 180.0, 10 * np.pi / 180.0, *board_size])
        self.goal_norm_max = np.array(
            [0.0002 * 60 * k for k in range(1, num_rel_path + 1) for _ in range(2)]
        )
        self.offset = board_size / 2.0

        # Random-path generation. The path is the domain-randomization axis
        # here (see random_path.py), so these knobs double as the curriculum
        # surface for later work: widen them to make the task harder.
        self.path_distance = float(os.environ.get("TAG_BP_PATH_DISTANCE_M", "0.0002"))
        self.num_segments = int(os.environ.get("TAG_BP_NUM_SEGMENTS", "12"))
        self.min_step = float(os.environ.get("TAG_BP_MIN_STEP_M", "0.02"))
        self.max_step = float(os.environ.get("TAG_BP_MAX_STEP_M", "0.05"))
        self.max_turn_rad = float(os.environ.get("TAG_BP_MAX_TURN_RAD", "1.0"))
        self.path_margin = self.ball_radius + float(
            os.environ.get("TAG_BP_PATH_MARGIN_M", "0.01")
        )
        seed_env = os.environ.get("TAG_BP_SEED", "")
        self.rng = np.random.default_rng(int(seed_env) if seed_env else None)

        # There are no walls to make a shortcut geometrically impossible (the
        # maze env's whole anti-cheat scheme), so distance-to-path is the only
        # thing keeping "progress" tied to actually tracing the path's shape
        # rather than beelining across open board.
        self.path_tolerance = float(os.environ.get("TAG_BP_PATH_TOLERANCE_M", "0.02"))
        self.off_path_confirm_steps = max(
            1, int(os.environ.get("TAG_BP_OFFPATH_CONFIRM_STEPS", "10"))
        )
        self.checkpoint_bonus = float(os.environ.get("TAG_BP_CHECKPOINT_BONUS", "0.02"))
        self.ball_loss_grace_frames = max(
            0, int(os.environ.get("TAG_BP_BALL_LOSS_GRACE_FRAMES", "6"))
        )

        self.num_wait_steps = num_wait_steps
        self.reward_on_fail = float(
            os.environ.get("TAG_REWARD_ON_FAIL", str(reward_on_fail))
        )
        self.off_path_penalty = float(
            os.environ.get("TAG_BP_OFFPATH_PENALTY", str(self.reward_on_fail))
        )
        self.reward_on_goal = reward_on_goal
        self.timeout_steps = max(1, int(os.environ.get("TAG_TIMEOUT_STEPS", "1500")))
        self.timeout_penalty = float(
            os.environ.get("TAG_TIMEOUT_PENALTY", str(self.reward_on_fail))
        )
        self.action_repeat = max(1, int(os.environ.get("TAG_ACTION_REPEAT", "1")))
        self.max_angle_vel = float(os.environ.get("TAG_MAX_ANGLE_VEL", "300"))
        self.alpha_fac = float(os.environ.get("TAG_ALPHA_FAC", "-1.0"))
        self.beta_fac = float(os.environ.get("TAG_BETA_FAC", "-1.0"))
        self.max_cmd_1 = float(os.environ.get("TAG_MAX_CMD_1", "300"))
        self.max_cmd_2 = float(os.environ.get("TAG_MAX_CMD_2", "300"))

        self.p = None
        self._waypoint_indices = None
        self.prev_pos_path = 0
        self.best_checkpoint = 0
        self.off_path_steps = 0
        self.off_path_triggered = False
        self.ball_detected = False
        self.ball_missing_frames = 0
        self.last_valid_obs = None
        self.steps = 0
        self.episodes = 0
        self.success = False
        self.accum_reward = 0.0
        self.last_time = 0
        self.episode_start_time = time.monotonic()

        host = os.environ.get("TAG_TCP_BIND", "0.0.0.0")
        port = int(os.environ.get("TAG_TCP_PORT", "5555"))
        span = max(1, int(os.environ.get("TAG_TCP_PORT_SPAN", "1")))

        print(f"[BallPlate ENV] Waiting for board bridge on {host}:{port} ...")
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        for offset_try in range(span):
            try:
                self.server.bind((host, port + offset_try))
                port += offset_try
                break
            except OSError:
                if offset_try == span - 1:
                    raise
        if span > 1:
            print(f"[BallPlate ENV] This environment took port {port}")
        self.server.listen(1)
        self.sock, addr = self.server.accept()
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"[BallPlate ENV] board bridge connected from {addr}")

    def _request(self, obj):
        msg = json.dumps(obj, separators=(",", ":")).encode("utf-8") + b"\n"
        self.sock.sendall(msg)
        line = _recv_line(self.sock)
        return json.loads(line)

    def _new_path(self, start_lower_left):
        waypoints = generate_waypoints(
            self.rng,
            self.board_width,
            self.board_height,
            start=start_lower_left,
            margin=self.path_margin,
            num_segments=self.num_segments,
            min_step=self.min_step,
            max_step=self.max_step,
            max_turn_rad=self.max_turn_rad,
        )
        self.p = LinearPath(waypoints, distance=self.path_distance)
        self._waypoint_indices = None

    def _waypoint_path_indices(self):
        if self._waypoint_indices is None:
            indices = [0]
            total = 0
            for index in range(1, self.p.orig_waypoints.shape[0]):
                segment = self.p.orig_waypoints[index] - self.p.orig_waypoints[index - 1]
                total += int(np.floor(np.linalg.norm(segment) / self.p.distance)) + 1
                indices.append(min(total, self.p.num_points - 1))
            self._waypoint_indices = indices
        return self._waypoint_indices

    def _checkpoint_at(self, path_index):
        return int(np.searchsorted(self._waypoint_path_indices(), path_index, side="right"))

    def _closest_point(self, point):
        """Nearest path point, or -1 if it is further than the path tolerates.

        With no walls, LinearPath.closest_point always finds a nearest point
        (the sections/closest_idx grid this repo's maze layouts build is never
        constructed here) -- this tolerance is the only thing standing in for
        that grid's "off the corridor entirely" signal.
        """
        idx, closest = self.p.closest_point(point)
        if idx == -1:
            return -1, None
        if float(np.linalg.norm(closest - point)) > self.path_tolerance:
            return -1, None
        return idx, closest

    def _get_rel_path(self, point):
        idx = self._closest_point(point)[0]
        if idx == -1:
            return np.zeros((self.num_rel_path, 2), dtype=np.float32)
        indices = np.clip(
            np.arange(idx, idx + self.num_rel_path * 60, 60), 0, self.p.num_points - 1
        )
        return np.asarray(self.p.points[indices] - point, dtype=np.float32)

    def step(self, action):
        self.steps += 1
        obs = self._step_remote(action)

        reward = self._get_reward(obs)
        done = self._get_done(obs)
        timed_out = not done and self.steps >= self.timeout_steps
        if timed_out:
            done = True
            self._send_action(np.zeros((2,), dtype=np.float32))
        elapsed_sec = time.monotonic() - self.episode_start_time

        if done and not self.success:
            if self.off_path_triggered:
                reward = self.off_path_penalty
            elif timed_out:
                reward = self.timeout_penalty
            else:
                reward = self.reward_on_fail
        if self.success:
            reward += self.reward_on_goal

        if done:
            print("[BallPlate ENV] Resetting board")
            self._request({"cmd": "reset"})

        info = {"is_terminal": False} if self.success or timed_out else {}

        self.accum_reward += reward if not done else 0
        obs["states"] = (obs["states"] / self.norm_max).astype(np.float32)
        obs["goal"] = (obs["goal"] / self.goal_norm_max).astype(np.float32)
        obs["progress"] = np.asarray([1 + self.prev_pos_path], dtype=np.float32)
        obs["log_reward"] = np.asarray([reward if not done else 0], dtype=np.float32)
        obs["log_elapsed_sec"] = np.asarray([elapsed_sec], dtype=np.float32)
        obs["log_success"] = np.asarray([float(self.success)], dtype=np.float32)
        return obs, reward, done, info

    def reset(self):
        self.episodes += 1
        print(
            f"[BallPlate ENV] episode {self.episodes}, "
            f"previous reward {self.accum_reward:.3f}, previous length {self.steps}"
        )
        self.accum_reward = 0.0
        self.steps = 0
        self.success = False
        self.off_path_steps = 0
        self.off_path_triggered = False
        self.ball_detected = False
        self.ball_missing_frames = 0
        self.last_valid_obs = None
        self.best_checkpoint = 0

        self._send_action(np.zeros((2,)))

        count = 0
        raw = self._get_obs_raw()
        while count < self.num_wait_steps:
            raw = self._get_obs_raw()
            count = count + 1 if raw.get("ball", False) else 0
            if not raw.get("ball", False):
                time.sleep(0.02)

        start = np.array([raw["x_b"], raw["y_b"]], dtype=np.float32) + self.offset
        self._new_path(start)
        obs = self._obs_from_reply(raw)

        path_idx = self._closest_point(obs["states"][2:4])[0]
        if path_idx == -1:
            path_idx = 0
        self.prev_pos_path = path_idx
        self.best_checkpoint = self._checkpoint_at(path_idx)

        self.last_time = time.time()
        self.episode_start_time = time.monotonic()

        obs["states"] = (obs["states"] / self.norm_max).astype(np.float32)
        obs["goal"] = (obs["goal"] / self.goal_norm_max).astype(np.float32)
        obs["progress"] = np.asarray([1 + self.prev_pos_path], dtype=np.float32)
        obs["log_reward"] = np.asarray([0.0], dtype=np.float32)
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
                    "cmd": "step", "vel_1": vel_1, "vel_2": vel_2, "timeout": 0.018,
                })
                if rep.get("ok", False):
                    break
                time.sleep(0.01)
        return self._obs_from_reply(rep)

    def _get_obs_raw(self):
        while True:
            rep = self._request({"cmd": "obs"})
            if rep.get("ok", False):
                return rep
            time.sleep(0.01)

    def _obs_from_reply(self, rep):
        if not rep.get("ball", False):
            return self._obs_for_missing_ball()

        self.ball_detected = True
        self.ball_missing_frames = 0

        states = np.array(
            [float(rep["alpha"]), float(rep["beta"]), float(rep["x_b"]), float(rep["y_b"])],
            dtype=np.float32,
        )
        states[2:] += self.offset

        img_bytes = base64.b64decode(rep["image_b64"])
        image = np.frombuffer(img_bytes, dtype=np.uint8).reshape(64, 64, 1)

        rel_path = self._get_rel_path(states[2:]).flatten().astype(np.float32)

        states = np.nan_to_num(states, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        rel_path = np.nan_to_num(rel_path, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        image = np.clip(np.nan_to_num(image), 0, 255).astype(np.uint8)

        obs = {"states": states, "goal": rel_path, "image": image}
        self.last_valid_obs = {key: value.copy() for key, value in obs.items()}
        return {key: value.copy() for key, value in obs.items()}

    def _obs_for_missing_ball(self):
        self.ball_missing_frames += 1
        within_grace = (
            self.last_valid_obs is not None
            and self.ball_missing_frames <= self.ball_loss_grace_frames
        )
        self.ball_detected = within_grace
        if self.last_valid_obs is None:
            return {
                "states": np.zeros(4, dtype=np.float32),
                "goal": np.zeros((self.num_rel_path * 2,), dtype=np.float32),
                "image": np.zeros((64, 64, 1), dtype=np.uint8),
            }
        return {key: value.copy() for key, value in self.last_valid_obs.items()}

    def _send_action(self, action):
        vel_1, vel_2 = self._action_to_command(action)
        self._request({"cmd": "action", "vel_1": vel_1, "vel_2": vel_2})

    def _action_to_command(self, action):
        action = np.asarray(action, dtype=np.float32).copy()
        action *= self.max_angle_vel
        vel_1 = float(np.clip(self.alpha_fac * action[0], -self.max_cmd_1, self.max_cmd_1))
        vel_2 = float(np.clip(self.beta_fac * action[1], -self.max_cmd_2, self.max_cmd_2))
        return vel_1, vel_2

    def _get_reward(self, obs):
        if not self.ball_detected:
            return 0.0

        curr_pos_path, _ = self._closest_point(obs["states"][2:4])
        if curr_pos_path == -1:
            self.off_path_steps += 1
            return 0.0
        self.off_path_steps = 0

        progress = curr_pos_path - self.prev_pos_path
        reward = float(progress) * 0.004 / 16.0
        self.prev_pos_path = curr_pos_path
        reward += self._checkpoint_bonus_reward()
        return reward

    def _checkpoint_bonus_reward(self):
        if self.checkpoint_bonus == 0.0:
            return 0.0
        reached = self._checkpoint_at(self.prev_pos_path)
        if reached <= self.best_checkpoint:
            return 0.0
        gained = reached - self.best_checkpoint
        self.best_checkpoint = reached
        return gained * self.checkpoint_bonus

    def _get_done(self, obs):
        done = False
        if not self.ball_detected:
            done = True
            print("[BallPlate ENV] Done: BALL LOST")
            self._send_action(np.zeros((2,), dtype=np.float32))

        if self.off_path_steps >= self.off_path_confirm_steps:
            self.off_path_triggered = True
            done = True
            print(f"[BallPlate ENV] Done: OFF PATH after {self.off_path_steps} frames")
            self._send_action(np.zeros((2,), dtype=np.float32))

        if not done and self.p.num_points - self.prev_pos_path <= 1:
            self.success = True
            done = True
            print("[BallPlate ENV] Done: SUCCESS")
            self._send_action(np.zeros((2,), dtype=np.float32))

        return done
