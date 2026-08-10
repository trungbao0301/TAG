#!/usr/bin/env python3
"""A bare ball-on-plate physics stand-in, speaking the same TCP protocol as
tools/isaac_tcp_server.py and tcp_ros_bridge.py, so tag_dreamer's env code
cannot tell it apart from Isaac Sim or the real robot.

No ROS, no Isaac Sim, no GPU -- just numpy. Meant to run anywhere (including a
laptop with no CUDA GPU) and to be cheap enough to reset thousands of times
with fresh domain-randomized physics, which is the point: this is for
pretraining a map-agnostic skill at volume, not for the fidelity checks Isaac
Sim exists for.

    python3 tools/lite_board_server.py --host <learner-host> --port 5556

The servo/tilt command chain (ServoAxis below) is copied verbatim from
tools/isaac_tcp_server.py, which copied it from the real Hiwonder driver
(tag_hiwonder/scripts/hiwonder_compat_node.py). All three therefore share one
command path; only the marble dynamics past that point are this file's own.

DEG_PER_UNIT is inferred from one successful run, not measured (see
sim/README.md) -- so rather than trust it exactly, this sim samples a range
around it every episode along with the marble's rolling/friction behaviour,
sensor latency, position noise, and detection dropout. None of these ranges
are measurements either; they only need to be wide enough that a policy robust
across them stops depending on any one guess being right.
"""
import argparse
import base64
import json
import math
import socket
import time
import traceback

import numpy as np

# --- Hiwonder command chain, verbatim from tools/isaac_tcp_server.py ---
HOME_POS = 500.0
SERVO_SCALE = 1.5
SERVO_MIN = 100.0
SERVO_MAX = 900.0
MAX_STEP_PER_TICK = 20.0
COMMAND_RATE_HZ = 30.0
DEG_PER_UNIT_NOMINAL = 0.018

# --- board geometry (tag_dreamer/tag_dreamer/tag_layout_custom.py) ---
BOARD_WIDTH_M = 0.259
BOARD_HEIGHT_M = 0.229
MARBLE_RADIUS_M = 0.006
GRAVITY_M_S2 = 9.81


class ServoAxis:
    """One Hiwonder servo: affine command mapping plus the driver's slew limit."""

    def __init__(self, deg_per_unit):
        self.position = HOME_POS
        self.deg_per_unit = deg_per_unit

    def target(self, vel):
        return float(np.clip(HOME_POS + SERVO_SCALE * float(vel), SERVO_MIN, SERVO_MAX))

    def advance(self, vel, seconds):
        ticks = max(1, int(round(seconds * COMMAND_RATE_HZ)))
        goal = self.target(vel)
        for _ in range(ticks):
            delta = goal - self.position
            self.position += float(np.clip(delta, -MAX_STEP_PER_TICK, MAX_STEP_PER_TICK))
        return self.position

    @property
    def angle_deg(self):
        return self.deg_per_unit * (self.position - HOME_POS)

    def home(self):
        self.position = HOME_POS


class BallPlateSim:
    """A ball rolling on a tilting plate. No walls, no holes, no fixed map.

    Reports x_b/y_b in board-centred metres, alpha/beta in radians, and a
    64x64 grey composite image -- the exact fields env_tcp.py's TagGym (and
    its ball-on-plate sibling, env_ballplate.py) already read from Isaac or
    the robot bridge.
    """

    def __init__(self, args):
        self.args = args
        self.rng = np.random.default_rng(args.seed if args.seed >= 0 else None)
        self.control_dt = 1.0 / args.control_hz
        self.physics_substeps = max(1, args.physics_substeps)
        self.margin = MARBLE_RADIUS_M + 0.0005
        self.half_extent = np.array(
            [BOARD_WIDTH_M / 2.0 - self.margin, BOARD_HEIGHT_M / 2.0 - self.margin]
        )
        self._m_per_px = 1.15 * max(BOARD_WIDTH_M, BOARD_HEIGHT_M) / 64.0
        self.total_steps = 0
        self.episode_steps = 0
        self.servo1 = None
        self.servo2 = None
        self.last_state = None
        self.reset()

    def _sample_params(self):
        a = self.args
        rng = self.rng
        return {
            "deg_per_unit": float(rng.uniform(a.deg_per_unit_lo, a.deg_per_unit_hi)),
            "rolling_factor": float(rng.uniform(a.rolling_factor_lo, a.rolling_factor_hi)),
            "damping": float(rng.uniform(a.damping_lo, a.damping_hi)),
            "restitution": float(rng.uniform(a.restitution_lo, a.restitution_hi)),
            "latency_steps": int(rng.integers(a.latency_lo, a.latency_hi + 1)),
            "position_noise_m": float(rng.uniform(a.position_noise_lo, a.position_noise_hi)),
            "dropout_prob": float(rng.uniform(a.dropout_lo, a.dropout_hi)),
        }

    def reset(self):
        self.params = self._sample_params()
        if self.servo1 is None:
            self.servo1 = ServoAxis(self.params["deg_per_unit"])
            self.servo2 = ServoAxis(self.params["deg_per_unit"])
        else:
            self.servo1.deg_per_unit = self.params["deg_per_unit"]
            self.servo2.deg_per_unit = self.params["deg_per_unit"]
        self.servo1.home()
        self.servo2.home()
        self.pos = np.zeros(2, dtype=np.float64)
        self.vel = np.zeros(2, dtype=np.float64)
        self.episode_steps = 0
        self._history = []

    def apply(self, vel_1, vel_2):
        self.servo1.advance(vel_1, self.control_dt)
        self.servo2.advance(vel_2, self.control_dt)

    def _tilt_rad(self):
        if self.args.axis_map == "1beta":
            beta_deg, alpha_deg = self.servo1.angle_deg, self.servo2.angle_deg
        else:
            alpha_deg, beta_deg = self.servo1.angle_deg, self.servo2.angle_deg
        return math.radians(alpha_deg), math.radians(beta_deg)

    def advance(self):
        self.episode_steps += 1
        self.total_steps += 1
        alpha, beta = self._tilt_rad()
        dt = self.control_dt / self.physics_substeps
        k = self.params["rolling_factor"]
        damping = self.params["damping"]
        restitution = self.params["restitution"]
        # Board x runs with beta's tilt, board y with alpha's -- matching the
        # axis convention env_tcp.py's states[2], states[3] (x_b, y_b) expect.
        accel = k * GRAVITY_M_S2 * np.array([math.sin(beta), math.sin(alpha)])
        for _ in range(self.physics_substeps):
            self.vel += (accel - damping * self.vel) * dt
            self.pos += self.vel * dt
            for axis in (0, 1):
                if abs(self.pos[axis]) > self.half_extent[axis]:
                    self.pos[axis] = math.copysign(self.half_extent[axis], self.pos[axis])
                    self.vel[axis] *= -restitution

    def observation(self):
        alpha, beta = self._tilt_rad()
        true_pos = self.pos.copy()
        self._history.append(true_pos)
        delay = self.params["latency_steps"]
        keep = delay + 2
        if len(self._history) > keep:
            del self._history[:-keep]
        reported = self._history[max(0, len(self._history) - 1 - delay)]
        noisy = reported + self.rng.normal(0.0, self.params["position_noise_m"], size=2)
        ball = bool(self.rng.uniform() >= self.params["dropout_prob"])
        image = self._render(true_pos if ball else None)
        self.last_state = {
            "ball": ball, "alpha": alpha, "beta": beta,
            "pos": true_pos.copy(), "step": self.total_steps,
        }
        reply = {
            "ok": True,
            "ball": ball,
            "alpha": alpha,
            "beta": beta,
            "x_b": float(noisy[0]) if ball else float("nan"),
            "y_b": float(noisy[1]) if ball else float("nan"),
            "image_b64": base64.b64encode(image.tobytes()).decode("ascii"),
        }
        return reply

    def _render(self, pos_centered):
        frame = np.full((64, 64), 40, dtype=np.uint8)
        # A light rim, so the CNN has a fixed frame of reference even when the
        # marble (the only other thing in view) is near an edge.
        frame[0, :] = frame[-1, :] = frame[:, 0] = frame[:, -1] = 90
        if pos_centered is None:
            return frame.reshape(64, 64, 1)
        u = 32 + pos_centered[0] / self._m_per_px
        v = 32 - pos_centered[1] / self._m_per_px
        yy, xx = np.mgrid[0:64, 0:64]
        r_px = max(1, int(round(MARBLE_RADIUS_M / self._m_per_px)))
        mask = (xx - u) ** 2 + (yy - v) ** 2 <= r_px ** 2
        frame[mask] = 255
        return frame.reshape(64, 64, 1)


def start_overlay_server(sim, port):
    """Serve the same live-view page draw_path_server.py expects at
    /frame.png -- isaac_tcp_server.py's --overlay-port equivalent, so either
    physics backend plugs into the same drawing UI unmodified."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading

    import cv2

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            if not self.path.startswith("/frame.png"):
                self.send_error(404)
                return
            scale = 6
            st = sim.last_state
            frame = np.zeros((64, 64), dtype=np.uint8) if st is None else sim._render(
                st["pos"] if st["ball"] else None
            ).reshape(64, 64)
            img = cv2.resize(frame, (64 * scale, 64 * scale), interpolation=cv2.INTER_NEAREST)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            lines = ["step %d" % (st["step"] if st else 0)]
            if st is not None:
                lines += [
                    "ball %s" % ("seen" if st["ball"] else "LOST"),
                    "alpha %+.2f deg  beta %+.2f deg"
                    % (math.degrees(st["alpha"]), math.degrees(st["beta"])),
                ]
            for i, text in enumerate(lines):
                cv2.putText(img, text, (6, 16 + 16 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (230, 230, 230), 1, cv2.LINE_AA)
            ok, buf = cv2.imencode(".png", img)
            if not ok:
                self.send_error(500)
                return
            body = buf.tobytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print("[lite] overlay on http://0.0.0.0:%d/frame.png" % port)


def recv_line(sock, buffer):
    while b"\n" not in buffer:
        chunk = sock.recv(4096)
        if not chunk:
            return None, buffer
        buffer += chunk
    line, _, rest = buffer.partition(b"\n")
    return line, rest


def connect_to_learner(args):
    """Dial the learner, retrying, exactly as tcp_ros_bridge.py and
    isaac_tcp_server.py do -- the learner is the one that binds and listens."""
    while True:
        try:
            sock = socket.create_connection((args.host, args.port), timeout=None)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print("[lite] connected to learner at %s:%d" % (args.host, args.port))
            return sock
        except OSError as exc:
            print("[lite] learner not up yet (%s); retrying in 2 s" % exc)
            time.sleep(2.0)


def serve(sim, args):
    while True:
        conn = connect_to_learner(args)
        buffer = b""
        steps = 0
        t0 = time.time()
        try:
            while True:
                line, buffer = recv_line(conn, buffer)
                if line is None:
                    break
                request = json.loads(line)
                cmd = request.get("cmd", "")
                if cmd == "step":
                    sim.apply(request.get("vel_1", 0.0), request.get("vel_2", 0.0))
                    sim.advance()
                    reply = sim.observation()
                    steps += 1
                    if steps % 1000 == 0:
                        rate = steps / (time.time() - t0)
                        print("[lite] %d steps, %.0f steps/s, params=%s"
                              % (steps, rate, sim.params))
                elif cmd == "obs":
                    reply = sim.observation()
                elif cmd == "action":
                    sim.apply(request.get("vel_1", 0.0), request.get("vel_2", 0.0))
                    reply = {"ok": True}
                elif cmd == "reset":
                    sim.reset()
                    reply = {"ok": True}
                else:
                    reply = {"ok": False, "error": "unknown cmd %r" % cmd}
                conn.sendall(json.dumps(reply).encode("utf-8") + b"\n")
        except (ConnectionResetError, BrokenPipeError, json.JSONDecodeError) as exc:
            print("[lite] link lost (%s); waiting for the learner again" % exc)
        except Exception:
            print("[lite] request failed:\n%s" % traceback.format_exc(), flush=True)
            raise
        finally:
            conn.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1", help="learner address to dial")
    p.add_argument("--port", type=int, default=5556)
    p.add_argument("--control-hz", type=float, default=COMMAND_RATE_HZ)
    p.add_argument("--physics-substeps", type=int, default=4,
                    help="finer integration steps per control tick")
    p.add_argument("--axis-map", choices=("1beta", "1alpha"), default="1beta",
                    help="which board axis servo 1 drives -- matches "
                         "isaac_tcp_server.py's default")
    p.add_argument("--seed", type=int, default=-1, help="-1 = unseeded")

    # Domain randomization ranges. None of these are measurements -- see the
    # module docstring -- so they are meant to be wide, not exact.
    p.add_argument("--deg-per-unit-lo", type=float, default=0.012)
    p.add_argument("--deg-per-unit-hi", type=float, default=0.026)
    p.add_argument("--rolling-factor-lo", type=float, default=0.55,
                    help="fraction of g*sin(tilt) that reaches the marble's "
                         "linear acceleration; 5/7=0.714 is a solid sphere "
                         "rolling without slipping, lower values model slip")
    p.add_argument("--rolling-factor-hi", type=float, default=1.0)
    p.add_argument("--damping-lo", type=float, default=0.05, help="1/s, viscous")
    p.add_argument("--damping-hi", type=float, default=0.6)
    p.add_argument("--restitution-lo", type=float, default=0.2,
                    help="bounce off the board's outer rim (no walls exist "
                         "in this task, but the board's edge does)")
    p.add_argument("--restitution-hi", type=float, default=0.7)
    p.add_argument("--latency-lo", type=int, default=0, help="steps of reporting delay")
    p.add_argument("--latency-hi", type=int, default=3)
    p.add_argument("--position-noise-lo", type=float, default=0.0, help="metres, std dev")
    p.add_argument("--position-noise-hi", type=float, default=0.004)
    p.add_argument("--dropout-lo", type=float, default=0.0,
                    help="per-step probability the marble is reported missing, "
                         "modelling detector dropout -- sim/README.md notes "
                         "this is entirely unmodelled in Isaac today")
    p.add_argument("--dropout-hi", type=float, default=0.03)
    p.add_argument("--overlay-port", type=int, default=0,
                    help="serve a live view at http://host:PORT/frame.png "
                         "(0 = off) -- the same endpoint "
                         "isaac_tcp_server.py's --overlay-port serves, so "
                         "tools/draw_path_server.py's --overlay-url works "
                         "unmodified against either backend.")
    return p.parse_args()


def main():
    args = parse_args()
    sim = BallPlateSim(args)
    print("[lite] control_dt=%.4f substeps=%d axis_map=%s"
          % (sim.control_dt, sim.physics_substeps, args.axis_map))
    if args.overlay_port:
        start_overlay_server(sim, args.overlay_port)
    try:
        serve(sim, args)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
