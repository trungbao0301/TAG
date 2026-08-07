#!/usr/bin/env python3
"""Serve the TAG learner from Isaac Sim, speaking the robot bridge's protocol.

The learner in tag_dreamer/env_tcp.py does not know or care what is on the other
end of the socket: it sends JSON lines and expects JSON lines back. This stands
in for tcp_ros_bridge.py so the same env, the same reward, and the same
checkpoint logic drive a simulated board instead of the real one.

Run it inside the Isaac Sim environment (Python 3.10), NOT the ROS one:

    ~/micromamba/envs/env_isaaclab/bin/python3 -u tools/isaac_tcp_server.py \
        --stage ~/issac-projects/tag_board.usd --port 5555

Then point the learner at it exactly as if it were the robot.

Everything about the command path is taken from the Hiwonder driver, since this
rig runs Hiwonder servos and nothing else:

    vel   = clip(action * TAG_MAX_ANGLE_VEL * fac, +-300)      env_tcp.py
    pos_t = clip(500 + 1.5 * vel, 100, 900)                    hiwonder_compat_node.py
    pos  += clip(pos_t - pos, +-20)   every 1/30 s             the driver's rate limit
    angle = DEG_PER_UNIT * (pos - 500)

DEG_PER_UNIT is the one number not readable from the code. It comes from the
successful run on this map: the command saturates at +-266.7 units (the servo
clamp), and over that run the board covered alpha -7.25..+3.79 deg and beta
-6.76..+2.42 deg, i.e. about +-7 deg at saturation.
"""
import argparse
import base64
import os
import json
import math
import socket
import sys
import time

# --- Hiwonder command chain (tag_hiwonder/scripts/hiwonder_compat_node.py) ---
HOME_POS = 500.0
SERVO_SCALE = 1.5
SERVO_MIN = 100.0
SERVO_MAX = 900.0
MAX_STEP_PER_TICK = 20.0
COMMAND_RATE_HZ = 30.0
# Degrees of board tilt per unit of SERVO POSITION (not per command unit): the
# command saturates at +-266.7, which the 1.5 scale turns into 400 units of servo
# travel from home. Over the successful run on this map the board covered alpha
# -7.25..+3.79 deg and beta -6.76..+2.42 deg, so 7.25 / 400 -> 0.018.
DEG_PER_UNIT = 0.018

# --- board geometry (tag_state_estimation/core/maze_layout.py) ---
BOARD_WIDTH_M = 0.259
BOARD_HEIGHT_M = 0.229
MARBLE_RADIUS_M = 0.006

# Where the marble is placed on reset, in lower-left board metres. Taken from a
# real [CP-DIAG-RESET] line so episodes start where they start on the robot.
RESET_XY_M = (0.12495, 0.22500)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, help="path to tag_board.usd")
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--host", default="127.0.0.1",
                   help="learner address to dial (or bind address with --role listen)")
    p.add_argument("--role", choices=("connect", "listen"), default="connect",
                   help="the learner listens, so the simulator connects -- same as "
                        "tcp_ros_bridge.py. 'listen' is for standalone testing.")
    p.add_argument("--control-hz", type=float, default=30.0,
                   help="how much sim time one 'step' request advances")
    p.add_argument("--axis-map", choices=("1beta", "1alpha"), default="1beta",
                   help="which board axis servo 1 drives. '1beta' matches the "
                        "only experimental evidence available (the July sysid "
                        "runs); flip it once measured on the current rig.")
    p.add_argument("--marble-source", choices=("truth", "hsv"), default="truth",
                   help="truth = the simulator's own marble pose; hsv = run the "
                        "colour detector on the rendered frame, like the robot")
    p.add_argument("--image", choices=("camera", "composite", "black"),
                   default="camera",
                   help="camera = render every step (slow, most faithful); "
                        "composite = render the empty maze ONCE and draw the "
                        "marble onto it each step (nearly free, and unlike "
                        "'black' it keeps the CNN branch of the world model "
                        "learning something real); black = no image at all")
    p.add_argument("--reset-mode", choices=("fixed", "spread"), default="fixed",
                   help="fixed = always the robot's start position; spread = "
                        "start somewhere along the path, so the hard stretch "
                        "gets practised as often as the easy opening")
    p.add_argument("--reset-span", default="0.00,0.60",
                   help="fractions of the path that 'spread' samples between")
    p.add_argument("--reset-from-start-prob", type=float, default=0.3,
                   help="how often 'spread' still starts at the true beginning, "
                        "so the opening is not forgotten")
    p.add_argument("--overlay-port", type=int, default=0,
                   help="serve a live map overlay on this port (0 = off). Open "
                        "it in a browser: the maze, the path, and the marble, "
                        "drawn from the same layout the learner scores against.")
    p.add_argument("--repo", default=os.path.expanduser("~/tag"),
                   help="TAG checkout, for the maze layout and path files")
    p.add_argument("--camera-prim", default="/World/Camera_TAG")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=400)
    return p.parse_args()


ARGS = parse_args()

from isaacsim import SimulationApp  # noqa: E402

sim_app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
import cv2  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Usd, UsdGeom, Gf  # noqa: E402
from isaacsim.core.api import SimulationContext  # noqa: E402
from isaacsim.core.prims import SingleRigidPrim, SingleArticulation  # noqa: E402
from isaacsim.core.utils.types import ArticulationAction  # noqa: E402



# --------------------------------------------------------------------------
# Live overlay: the maze, the path, and the marble, served over HTTP.
#
# The repo's overlay_map_view_simple.py is a ROS node subscribing to
# StateEstimate, and there is no ROS in the Isaac environment. But the layout it
# draws is a plain dict of numbers, so the drawing can be done here directly.
# Segment conventions are taken from that script: walls_h is (x1, x2, y),
# walls_v is (y1, y2, x), walls_angled is (x1, y1, x2, y2).
# --------------------------------------------------------------------------
MARGIN_M = 0.022
PX_PER_M = 2200.0


class Overlay:
    def __init__(self, repo):
        self.layout = None
        self.path_xy = None
        # Load the layout module by FILE, not by package import: importing
        # tag_dreamer runs its __init__, which registers the gym env and needs
        # gym -- absent in the Isaac environment, so the whole overlay silently
        # fell back to an empty board.
        try:
            import importlib.util
            layout_file = os.path.join(repo, "tag_dreamer", "tag_dreamer",
                                       "tag_layout_custom.py")
            spec = importlib.util.spec_from_file_location("tag_layout_custom",
                                                          layout_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.layout = module.tag_dxf_layout
            print("[sim] overlay: layout loaded (%d holes, %d+%d walls)"
                  % (len(self.layout["holes"]), len(self.layout["walls_h"]),
                     len(self.layout["walls_v"])))
        except Exception as exc:  # the overlay is a convenience, never a blocker
            print("[sim] overlay: no layout (%s)" % exc)
        # The pickle holds a LinearPath instance, whose class lives in a module
        # that also cannot be imported here. Unpickle it with a stand-in class so
        # the stored arrays come through, then find the (N, 2) one.
        import pickle

        class _Any:
            pass

        class _TolerantUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                try:
                    return super().find_class(module, name)
                except Exception:
                    return type(name, (_Any,), {})

        for candidate in ("tag_dreamer/data/path_custom.pkl",
                          "install/tag_dreamer/share/tag_dreamer/path_custom.pkl"):
            try:
                with open(os.path.join(repo, candidate), "rb") as handle:
                    obj = _TolerantUnpickler(handle).load()
                state = getattr(obj, "__dict__", {}) or {}
                best = None
                for value in state.values():
                    arr = np.asarray(value)
                    if arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] > 10:
                        if best is None or arr.shape[0] > best.shape[0]:
                            best = arr.astype(float)
                if best is not None:
                    self.path_xy = best
                    print("[sim] overlay: path loaded (%d points from %s)"
                          % (len(best), candidate))
                    break
            except Exception:
                continue
        if self.path_xy is None:
            print("[sim] overlay: no path polyline (drawing maze only)")
        self.state = {}
        self.base = self._draw_static()

    def _px(self, x, y):
        return (int(round((x + MARGIN_M) * PX_PER_M)),
                int(round((BOARD_HEIGHT_M + MARGIN_M - y) * PX_PER_M)))

    def _draw_static(self):
        w = int(round((BOARD_WIDTH_M + 2 * MARGIN_M) * PX_PER_M))
        h = int(round((BOARD_HEIGHT_M + 2 * MARGIN_M) * PX_PER_M))
        img = np.full((h, w, 3), 32, np.uint8)
        cv2.rectangle(img, self._px(0, 0), self._px(BOARD_WIDTH_M, BOARD_HEIGHT_M),
                      (70, 70, 70), 2)
        if self.layout is None:
            return img
        holes = np.asarray(self.layout["holes"], dtype=float)
        radii = np.asarray(self.layout.get("hole_radii", [0.0075] * len(holes)), dtype=float)
        for (hx, hy), r in zip(holes, radii):
            cv2.circle(img, self._px(hx, hy), max(2, int(r * PX_PER_M)), (12, 12, 12), -1)
            cv2.circle(img, self._px(hx, hy), max(2, int(r * PX_PER_M)), (90, 90, 90), 1)
        yellow = (60, 200, 235)
        for x1, x2, y in np.asarray(self.layout["walls_h"], dtype=float):
            cv2.line(img, self._px(x1, y), self._px(x2, y), yellow, 2)
        for y1, y2, x in np.asarray(self.layout["walls_v"], dtype=float):
            cv2.line(img, self._px(x, y1), self._px(x, y2), yellow, 2)
        for seg in np.asarray(self.layout.get("walls_angled", []), dtype=float):
            if len(seg) == 4:
                cv2.line(img, self._px(seg[0], seg[1]), self._px(seg[2], seg[3]), yellow, 2)
        if self.path_xy is not None:
            pts = np.array([self._px(x, y) for x, y in self.path_xy], np.int32)
            cv2.polylines(img, [pts], False, (120, 255, 120), 1)
        return img

    def frame(self):
        img = self.base.copy()
        st = dict(self.state)
        if st.get("ball") and np.isfinite(st.get("x", np.nan)):
            centre = self._px(st["x"], st["y"])
            cv2.circle(img, centre, max(3, int(MARBLE_RADIUS_M * PX_PER_M)), (255, 170, 60), -1)
            cv2.circle(img, centre, max(3, int(MARBLE_RADIUS_M * PX_PER_M)), (255, 255, 255), 1)
        lines = [
            "step %d   episode steps %d" % (st.get("total", 0), st.get("steps", 0)),
            "ball %s   x %.1f mm  y %.1f mm" % (
                "seen" if st.get("ball") else "LOST",
                1000 * st.get("x", float("nan")), 1000 * st.get("y", float("nan"))),
            "alpha %+.2f deg   beta %+.2f deg" % (
                math.degrees(st.get("alpha", 0.0)), math.degrees(st.get("beta", 0.0))),
        ]
        for i, text in enumerate(lines):
            cv2.putText(img, text, (8, 18 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (230, 230, 230), 1, cv2.LINE_AA)
        return img


_PAGE = b"""<html><head><title>TAG sim overlay</title>
<style>body{background:#111;margin:0;text-align:center}img{max-width:100%;image-rendering:pixelated}</style>
</head><body><img id=v src="/frame.png">
<script>setInterval(function(){document.getElementById('v').src='/frame.png?'+Date.now()},150)</script>
</body></html>"""


def start_overlay_server(overlay, port, board=None):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path.startswith("/camera.png"):
                frame = getattr(board, "last_camera", None) if board else None
                if frame is None:
                    self.send_error(503, "no camera frame yet")
                    return
                ok, buf = cv2.imencode(".png", frame)
                body = buf.tobytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            elif self.path.startswith("/frame.png"):
                ok, buf = cv2.imencode(".png", overlay.frame())
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
            else:
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(_PAGE)))
                self.end_headers()
                self.wfile.write(_PAGE)

    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print("[sim] overlay on http://0.0.0.0:%d" % port)


class ServoAxis:
    """One Hiwonder servo: affine command mapping plus the driver's slew limit."""

    def __init__(self):
        self.position = HOME_POS

    def target(self, vel):
        return float(np.clip(HOME_POS + SERVO_SCALE * float(vel), SERVO_MIN, SERVO_MAX))

    def advance(self, vel, seconds):
        """Move toward the commanded position for `seconds` of driver ticks."""
        ticks = max(1, int(round(seconds * COMMAND_RATE_HZ)))
        goal = self.target(vel)
        for _ in range(ticks):
            delta = goal - self.position
            self.position += float(np.clip(delta, -MAX_STEP_PER_TICK, MAX_STEP_PER_TICK))
        return self.position

    @property
    def angle_deg(self):
        return DEG_PER_UNIT * (self.position - HOME_POS)

    def home(self):
        self.position = HOME_POS


def load_path_xy(repo):
    """The target path as (N, 2) board metres, or None."""
    import pickle

    class _Any:
        pass

    class _Tolerant(pickle.Unpickler):
        def find_class(self, module, name):
            try:
                return super().find_class(module, name)
            except Exception:
                return type(name, (_Any,), {})

    for candidate in ("tag_dreamer/data/path_custom.pkl",
                      "install/tag_dreamer/share/tag_dreamer/path_custom.pkl"):
        try:
            with open(os.path.join(repo, candidate), "rb") as handle:
                obj = _Tolerant(handle).load()
            best = None
            for value in (getattr(obj, "__dict__", {}) or {}).values():
                arr = np.asarray(value)
                if arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] > 10:
                    if best is None or arr.shape[0] > best.shape[0]:
                        best = arr.astype(float)
            if best is not None:
                return best
        except Exception:
            continue
    return None


class Board:
    def __init__(self, args):
        omni.usd.get_context().open_stage(args.stage)
        self.stage = omni.usd.get_context().get_stage()
        if self.stage is None:
            raise SystemExit("could not open stage: %s" % args.stage)

        scene = self.stage.GetPrimAtPath("/World/PhysicsScene")
        hz = scene.GetAttribute("physxScene:timeStepsPerSecond")
        self.physics_dt = 1.0 / float(hz.Get() if hz and hz.Get() else 60.0)
        self.control_dt = 1.0 / float(args.control_hz)

        self.sim = SimulationContext(physics_dt=self.physics_dt,
                                     rendering_dt=self.physics_dt,
                                     stage_units_in_meters=1.0)
        self.marble = SingleRigidPrim("/World/Marble")
        self.board = SingleRigidPrim("/World/TAG/Board")
        self.articulation = SingleArticulation("/World/TAG/Base")

        # Mesh-local -> lower-left board metres. The maze mesh is axis aligned
        # with x across the board and z along it; y is its 15 mm thickness.
        mesh = self.stage.GetPrimAtPath("/World/TAG/Board/mazeCad/node_/mesh_")
        self.mesh_xf = UsdGeom.Xformable(mesh).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default())
        # The plate TILTS, so the mesh's world transform is not the one captured
        # at startup. Keep the mesh's pose RELATIVE to the plate -- that part is
        # rigid -- and recompose it with the plate's live pose every frame.
        # Without this, a marble near the rim at 7 deg of tilt reads ~16 mm below
        # the floor in the stale frame and gets reported lost the moment it
        # reaches a wall.
        board_xf0 = UsdGeom.Xformable(
            self.stage.GetPrimAtPath("/World/TAG/Board")
        ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        self.mesh_in_board = self.mesh_xf * board_xf0.GetInverse()
        pts = np.array(UsdGeom.Mesh(mesh).GetPointsAttr().Get(), dtype=np.float64)
        self.mesh_lo = pts.min(axis=0)
        self.mesh_hi = pts.max(axis=0)

        self.servo1 = ServoAxis()
        self.servo2 = ServoAxis()
        self.args = args
        self.camera = None
        self.dof_alpha = None
        self.dof_beta = None
        self.episode_steps = 0
        self.total_steps = 0
        self.last_camera = None
        self.reset_count = 0
        self.path_xy = None
        if args.reset_mode == "spread":
            self.path_xy = load_path_xy(args.repo)
            if self.path_xy is None:
                raise SystemExit("--reset-mode spread needs path_custom.pkl under --repo")
            steps = np.linalg.norm(np.diff(self.path_xy, axis=0), axis=1)
            self.path_cum = np.concatenate([[0.0], np.cumsum(steps)])
            lo, hi = (float(v) for v in args.reset_span.split(","))
            self.reset_span = (lo, hi)
            print("[sim] spread resets over %.0f-%.0f%% of a %.3f m path, %.0f%% from the start"
                  % (lo * 100, hi * 100, self.path_cum[-1],
                     args.reset_from_start_prob * 100))
        self.overlay = None
        if args.overlay_port:
            self.overlay = Overlay(args.repo)
            start_overlay_server(self.overlay, args.overlay_port, self)

    def start(self):
        self.sim.play()
        self.articulation.initialize()
        names = list(self.articulation.dof_names)
        # baseToFrame turns about X -> alpha; frameToBoard about Y -> beta.
        for i, n in enumerate(names):
            if "baseToFrame" in n:
                self.dof_alpha = i
            elif "frameToBoard" in n:
                self.dof_beta = i
        if self.dof_alpha is None or self.dof_beta is None:
            raise SystemExit("could not find the two tilt joints in %s" % names)
        print("[sim] dofs: %s  alpha=%d beta=%d" % (names, self.dof_alpha, self.dof_beta))

        if self.args.image in ("camera", "composite"):
            from isaacsim.sensors.camera import Camera
            self.camera = Camera(prim_path=self.args.camera_prim,
                                 resolution=(self.args.width, self.args.height))
            self.camera.initialize()
            for _ in range(3):
                self.sim.step(render=True)
        if self.args.image == "composite":
            self._build_composite_background()
        self.reset()

    def _build_composite_background(self):
        """One render of the maze with the marble out of shot, kept as a backdrop.

        Rendering costs the same per frame whatever the resolution -- measured at
        38 steps/s for 640x400 against 53 for 128x80, versus 289 with no render
        at all -- so the only way to keep the image channel AND the speed is to
        stop rendering per step.
        """
        pose, orientation = self.marble.get_world_pose()
        self.marble.set_world_pose(position=np.array([0.0, 0.0, -5.0], dtype=np.float32))
        for _ in range(3):
            self.sim.step(render=True)
        self.background = self.gray64_from_camera()
        self.marble.set_world_pose(position=pose, orientation=orientation)
        self.marble.set_linear_velocity(np.zeros(3, dtype=np.float32))
        for _ in range(3):
            self.sim.step(render=False)
        # Marble radius in 64x64 pixels, from the board's own scale rather than
        # a guess: the maze is 259 mm across and fills most of the frame.
        span = self._project(np.array([[self.mesh_lo[0], self.mesh_hi[1], self.mesh_lo[2]],
                                       [self.mesh_hi[0], self.mesh_hi[1], self.mesh_lo[2]]]))
        px_per_m = abs(span[1][0] - span[0][0]) / BOARD_WIDTH_M if span is not None else 200.0
        self.marble_px = max(1, int(round(MARBLE_RADIUS_M * px_per_m)))
        print("[sim] composite backdrop built; marble drawn at r=%d px" % self.marble_px)

    def _project(self, points_3d):
        """World points -> 64x64 image coordinates, or None if unavailable."""
        try:
            coords = self.camera.get_image_coords_from_world_points(
                np.asarray(points_3d, dtype=np.float64))
        except Exception:
            return None
        coords = np.asarray(coords, dtype=np.float64)
        sx = 64.0 / float(self.args.width)
        sy = 64.0 / float(self.args.height)
        return np.column_stack([coords[:, 0] * sx, coords[:, 1] * sy])

    # --- geometry helpers -------------------------------------------------
    def mesh_world_now(self):
        """The maze mesh's transform for the plate's CURRENT tilt."""
        try:
            pos, quat = self.board.get_world_pose()
        except Exception:
            return self.mesh_xf
        rotation = Gf.Matrix4d()
        rotation.SetRotate(Gf.Quatd(float(quat[0]),
                                    Gf.Vec3d(float(quat[1]), float(quat[2]), float(quat[3]))))
        rotation.SetTranslateOnly(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
        return self.mesh_in_board * rotation

    def marble_board_xy(self):
        """Marble position in board metres measured from the CENTRE, or None.

        This is what StateEstimate.x_b carries on the robot: the node fills it
        from kinematics.update(measurement.centered_xy), and env_tcp then adds
        board_size/2 itself. Sending lower-left coordinates here made the learner
        see the marble 129 mm off the board and end every episode as OFFPATH on
        its first frame.
        """
        pos, _ = self.marble.get_world_pose()
        world = Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2]))
        local = self.mesh_world_now().GetInverse().Transform(world)
        x = (local[0] - self.mesh_lo[0])
        # The mesh runs the other way along this axis than the board layout does.
        # Checked, not assumed: rasterising the mesh gives 21 circular voids, and
        # against tag_layout_custom's 21 hole centres this convention lands all
        # of them within 0.08 mm, while the unflipped one is out by 23 mm median.
        y = BOARD_HEIGHT_M - (local[2] - self.mesh_lo[2])
        # Below the floor by more than a marble radius = it went through a hole.
        # A tighter bound reads contact jitter at a wall as a fall.
        below = local[1] < -0.012
        if below or not (-0.02 <= x <= BOARD_WIDTH_M + 0.02):
            return None
        if not (-0.02 <= y <= BOARD_HEIGHT_M + 0.02):
            return None
        return float(x - BOARD_WIDTH_M / 2.0), float(y - BOARD_HEIGHT_M / 2.0)

    def board_angles_rad(self):
        q = self.articulation.get_joint_positions()
        return float(q[self.dof_alpha]), float(q[self.dof_beta])

    def gray64(self):
        if self.args.image == "composite":
            return self.gray64_composite()
        if self.camera is None:
            return np.zeros((64, 64, 1), dtype=np.uint8)
        return self.gray64_from_camera()

    def gray64_composite(self):
        frame = self.background.copy()
        pos, _ = self.marble.get_world_pose()
        uv = self._project(np.array([[float(pos[0]), float(pos[1]), float(pos[2])]]))
        if uv is not None:
            u, v = int(round(uv[0][0])), int(round(uv[0][1]))
            if 0 <= u < 64 and 0 <= v < 64:
                cv2.circle(frame, (u, v), self.marble_px, 255, -1)
        return frame.reshape(64, 64, 1)

    def gray64_from_camera(self):
        if self.camera is None:
            return np.zeros((64, 64, 1), dtype=np.uint8)
        self._stash_camera()
        rgba = self.camera.get_rgba()
        if rgba is None or rgba.size == 0:
            return np.zeros((64, 64, 1), dtype=np.uint8)
        img = np.asarray(rgba)[:, :, :3].mean(axis=2).astype(np.uint8)
        img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
        return img.reshape(64, 64, 1)

    # --- control ----------------------------------------------------------
    def _stash_camera(self):
        """Keep the full-resolution frame so it can be served for recording.

        The learner only ever gets 64x64 grey; that is what the robot sends and
        what the policy must consume. The full frame exists purely for the human
        watching, so it is kept aside rather than shrunk into the observation.
        """
        try:
            rgba = self.camera.get_rgba()
            if rgba is not None and getattr(rgba, "size", 0):
                self.last_camera = np.asarray(rgba)[:, :, :3][:, :, ::-1].copy()
        except Exception:
            pass

    def apply(self, vel_1, vel_2):
        self.servo1.advance(vel_1, self.control_dt)
        self.servo2.advance(vel_2, self.control_dt)
        if self.args.axis_map == "1beta":
            beta_deg, alpha_deg = self.servo1.angle_deg, self.servo2.angle_deg
        else:
            alpha_deg, beta_deg = self.servo1.angle_deg, self.servo2.angle_deg
        targets = np.array(self.articulation.get_joint_positions(), dtype=np.float32)
        targets[self.dof_alpha] = math.radians(alpha_deg)
        targets[self.dof_beta] = math.radians(beta_deg)
        # apply_action drives toward the target through the joint's drive gains;
        # set_joint_positions would teleport the plate and skip the servo model.
        self.articulation.apply_action(ArticulationAction(joint_positions=targets))

    def advance(self):
        # Render only the substep whose frame is actually sent. Rendering every
        # substep costs the same again per step and nobody ever sees the extra
        # frames -- it halved the step rate for nothing.
        steps = max(1, int(round(self.control_dt / self.physics_dt)))
        # Only 'camera' mode needs a fresh frame each step. Composite mode keeps a
        # camera object around purely to project the marble onto its one-off
        # backdrop, and rendering for it threw away the whole point.
        render = self.args.image == "camera"
        for i in range(steps):
            self.sim.step(render=render and i == steps - 1)

    def reset(self):
        self.servo1.home()
        self.servo2.home()
        self.apply(0.0, 0.0)
        start_xy = self.reset_xy()
        # Same y flip as marble_board_xy, so the marble is placed where the robot
        # places it rather than at its mirror image.
        origin = Gf.Vec3d(float(self.mesh_lo[0] + start_xy[0]),
                          float(self.mesh_hi[1] + MARBLE_RADIUS_M),
                          float(self.mesh_lo[2] + BOARD_HEIGHT_M - start_xy[1]))
        world = self.mesh_world_now().Transform(origin)
        self.marble.set_world_pose(position=np.array([world[0], world[1], world[2]],
                                                     dtype=np.float32))
        self.marble.set_linear_velocity(np.zeros(3, dtype=np.float32))
        self.marble.set_angular_velocity(np.zeros(3, dtype=np.float32))
        for _ in range(int(0.2 / self.physics_dt)):
            self.sim.step(render=False)

    def observation(self):
        alpha, beta = self.board_angles_rad()
        xy = self.marble_board_xy()
        image = self.gray64()
        if self.overlay is not None:
            self.overlay.state = {
                "x": (xy[0] + BOARD_WIDTH_M / 2.0) if xy else float("nan"),
                "y": (xy[1] + BOARD_HEIGHT_M / 2.0) if xy else float("nan"),
                "alpha": alpha, "beta": beta, "ball": xy is not None,
                "steps": self.episode_steps, "total": self.total_steps,
            }
        reply = {
            "ok": True,
            "ball": xy is not None,
            "alpha": alpha,
            "beta": beta,
            "image_b64": base64.b64encode(image.tobytes()).decode("ascii"),
        }
        if xy is not None:
            reply["x_b"], reply["y_b"] = xy
        else:
            reply["x_b"] = float("nan")
            reply["y_b"] = float("nan")
        return reply


def recv_line(sock, buffer):
    while b"\n" not in buffer:
        chunk = sock.recv(4096)
        if not chunk:
            return None, buffer
        buffer += chunk
    line, _, rest = buffer.partition(b"\n")
    return line, rest


def connect_to_learner(args):
    """Dial the learner, retrying, exactly as tcp_ros_bridge.py does.

    The learner is the one that binds and listens (env_tcp.py), so the simulator
    takes the robot bridge's side of the link and connects to it.
    """
    while True:
        try:
            sock = socket.create_connection((args.host, args.port), timeout=None)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print("[sim] connected to learner at %s:%d" % (args.host, args.port))
            return sock
        except OSError as exc:
            print("[sim] learner not up yet (%s); retrying in 2 s" % exc)
            time.sleep(2.0)


def serve(board, args):
    while True:
        if args.role == "connect":
            conn = connect_to_learner(args)
        else:
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((args.host, args.port))
            listener.listen(1)
            print("[sim] listening on %s:%d" % (args.host, args.port))
            conn, addr = listener.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            listener.close()
            print("[sim] learner connected from %s" % (addr,))
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
                    board.episode_steps += 1
                    board.total_steps += 1
                    board.apply(request.get("vel_1", 0.0), request.get("vel_2", 0.0))
                    board.advance()
                    reply = board.observation()
                    steps += 1
                    if steps % 500 == 0:
                        rate = steps / (time.time() - t0)
                        print("[sim] %d steps, %.1f steps/s (%.1fx real time)"
                              % (steps, rate, rate * board.control_dt))
                elif cmd == "obs":
                    reply = board.observation()
                elif cmd == "action":
                    board.apply(request.get("vel_1", 0.0), request.get("vel_2", 0.0))
                    reply = {"ok": True}
                elif cmd == "reset":
                    board.episode_steps = 0
                    board.reset()
                    reply = {"ok": True}
                else:
                    reply = {"ok": False, "error": "unknown cmd %r" % cmd}
                conn.sendall(json.dumps(reply).encode("utf-8") + b"\n")
        except (ConnectionResetError, BrokenPipeError, json.JSONDecodeError) as exc:
            print("[sim] link lost (%s); waiting for the learner again" % exc)
        finally:
            conn.close()


def main():
    board = Board(ARGS)
    board.start()
    print("[sim] physics_dt=%.4f  control_dt=%.4f  axis_map=%s  marble=%s  image=%s"
          % (board.physics_dt, board.control_dt, ARGS.axis_map,
             ARGS.marble_source, ARGS.image))
    try:
        serve(board, ARGS)
    except KeyboardInterrupt:
        pass
    finally:
        sim_app.close()


main()
