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
import traceback

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
    p.add_argument("--pendulum-backend", choices=("physx", "mujoco"), default="mujoco",
                   help="where the pendulum is simulated. 'mujoco' runs it in the "
                        "model its policy was trained in, driven by the plate "
                        "angles Isaac reports -- PhysX ignores authored inertia "
                        "and armature on articulation links, so the arm there "
                        "runs at a quarter of its real inertia.")
    p.add_argument("--pendulum-model", default="",
                   help="furuta_2d.xml, for the mujoco backend")
    p.add_argument("--pendulum-policy", default="",
                   help="weights (.npz) for the rig's trained Furuta balance "
                        "policy. Enabling it forces physics to 200 Hz, the rate "
                        "the policy was trained and deployed at.")
    p.add_argument("--pendulum-start-upright", type=float, default=0.0,
                   metavar="RAD",
                   help="place the rod this far from upright at reset, as the "
                        "training env does (pole = pi + theta_up). Zero releases "
                        "it hanging, which is fine: the policy trained with "
                        "init_angle_max = pi and swings up from there in ~0.4 s.")
    p.add_argument("--pendulum-reaction", type=int, default=1,
                   help="feed the pendulum's reaction torque back onto the "
                        "board's two tilt joints. Off, the rod is a decoration "
                        "that cannot touch the marble; on, the marble feels it "
                        "and the maze policy has to live with the disturbance.")
    p.add_argument("--board-stiffness", type=float, default=-1.0,
                   help="override the two tilt drives' stiffness. Negative "
                        "leaves the asset alone. The units are whatever the "
                        "articulation controller uses, so calibrate by applying "
                        "a known torque and measuring the droop rather than "
                        "trusting a number read off a datasheet.")
    p.add_argument("--board-probe-torque", type=float, default=0.0,
                   metavar="N_M",
                   help="hold this constant torque on the alpha tilt joint. "
                        "Droop divided by torque is the drive's real stiffness "
                        "in N.m/rad, which is the only way to know what the "
                        "controller's stiffness number actually means.")
    p.add_argument("--board-damping", type=float, default=-1.0,
                   help="damping to go with --board-stiffness. A soft drive "
                        "with the asset's original damping rings.")
    p.add_argument("--pendulum-reaction-scale", type=float, default=1.0,
                   help="scale that reaction. The plate's drive stiffness here "
                        "is not calibrated against the real servos, so this is "
                        "the knob for matching a measured deflection.")
    p.add_argument("--pendulum-torque-sign", type=float, default=1.0,
                   help="flip if positive motor torque turns the arm the other "
                        "way here than in the model the policy was trained in. "
                        "Independent of the angle convention, and just as fatal.")
    p.add_argument("--pendulum-theta-sign", type=float, default=1.0,
                   help="flip if the rod's joint axis runs opposite to the one "
                        "the policy trained on (furuta_2d.xml has the pole on "
                        "axis '-1 0 0'). A balance policy fed a mirrored angle "
                        "pushes the wrong way and can never catch the rod.")
    p.add_argument("--view-every", type=int, default=0,
                   help="render one full frame every N control steps, for the "
                        "/camera.png endpoint only. The observation is untouched, "
                        "so watching costs a fraction of a render instead of one "
                        "per step and the policy keeps seeing what it was seeing.")
    p.add_argument("--hide-maze", action="store_true",
                   help="deactivate the maze mesh (and its collider with it) "
                        "after reading the board extents from it, leaving a "
                        "bare plate -- for the ball-plate task's arbitrary "
                        "drawn/random paths, which the maze's walls and holes "
                        "would otherwise just get in the way of.")
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
from pxr import Usd, UsdGeom, UsdPhysics, Gf  # noqa: E402
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
        ]
        if st.get("pendulum"):
            p = st["pendulum"]
            lines.append("rod %+7.1f deg from upright   %s   arm %+6.2f rad"
                         % (math.degrees(p["theta"]),
                            "UPRIGHT" if p["upright"] else "fallen", p["phi"]))
            lines.append("policy action %+6.3f   torque %+8.5f N.m"
                         % (p["action"], p["torque"]))
        lines += [
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

    # A viewer is a convenience; training is not. If the port is still held by a
    # sim that has not finished dying, warn and carry on -- this killed a run
    # once, at startup, for nothing.
    try:
        server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    except OSError as exc:
        print("[sim] overlay port %d unavailable (%s); running without a viewer"
              % (port, exc))
        return
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
        self.policy = None
        self.mj_pendulum = None
        if args.pendulum_policy and args.pendulum_backend == "mujoco":
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from furuta_policy import MujocoFuruta
            self.mj_pendulum = MujocoFuruta(args.pendulum_model, args.pendulum_policy)
            print("[sim] pendulum runs in MuJoCo, fed the plate angles from Isaac")
        elif args.pendulum_policy:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from furuta_policy import FurutaPolicy, CONTROL_DT
            self.policy = FurutaPolicy(args.pendulum_policy)
            # The policy's observation encodes rates and an action history sampled
            # every 5 ms. Running it at any other rate feeds it numbers it never
            # saw, so the simulation moves to its clock rather than the reverse.
            # The policy ticks at 200 Hz, but the model it was trained in steps
            # at 1 kHz underneath -- five substeps per tick. Stepping physics at
            # the control rate instead let the arm turn 1.6 rad in a single step
            # under full torque, and the solver produced NaN.
            self.policy_substeps = int(round(CONTROL_DT / 0.001))
            print("[sim] pendulum policy at %.0f Hz over physics at 1000 Hz "
                  "(%d substeps per tick, as in furuta_2d.xml)"
                  % (1 / CONTROL_DT, self.policy_substeps))
            self.physics_dt = 0.001
            self._substep = 0
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

        # Board extents, mesh transform, and mesh_in_board are already pulled
        # out above -- deactivating now (after reading them, not before) drops
        # the maze's render geometry and its collider together, since USD
        # deactivation removes a prim's whole subtree from composition.
        if args.hide_maze:
            maze_prim = self.stage.GetPrimAtPath("/World/TAG/Board/mazeCad")
            if maze_prim.IsValid():
                maze_prim.SetActive(False)
                print("[sim] --hide-maze: maze mesh and its collider deactivated")
                self._add_bare_floor()
            else:
                print("[sim] --hide-maze: WARNING, /World/TAG/Board/mazeCad not found")

        self.servo1 = ServoAxis()
        self.servo2 = ServoAxis()
        self.args = args
        self.camera = None
        self.dof_alpha = None
        self.dof_beta = None
        self.dof_arm = None
        self.dof_rod = None
        self.episode_steps = 0
        self.total_steps = 0
        # How well the pendulum holds while the board works the marble. Without
        # this the only evidence is a rod that looks short from a top-down camera.
        self.pend_hits = 0
        self.pend_samples = 0
        self.pend_worst = 0.0
        self.reaction_torque = None
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
        # The pendulum may be part of this articulation (older assets) or its
        # own -- a 15 g chain shares a solve badly with a 650 g plate, so the
        # asset now roots it separately and joins the two with a fixed joint
        # outside either articulation.
        self.pend_art = self.articulation
        self.dof_arm = next((i for i, n in enumerate(names) if "RotaryArm" in n
                             and "Rod" not in n), None)
        self.dof_rod = next((i for i, n in enumerate(names) if "PendulumRod" in n), None)
        if self.dof_rod is None:
            try:
                art = SingleArticulation("/World/TAG/FurutaPendulum/PendulumBase")
                art.initialize()
                pnames = list(art.dof_names)
                arm = next((i for i, n in enumerate(pnames) if "RotaryArm" in n
                            and "Rod" not in n), None)
                rod = next((i for i, n in enumerate(pnames) if "PendulumRod" in n), None)
                if rod is not None:
                    self.pend_art, self.dof_arm, self.dof_rod = art, arm, rod
                    print("[sim] pendulum is its own articulation: %s" % pnames)
            except Exception as exc:
                print("[sim] no separate pendulum articulation (%s)" % exc)
        if self.dof_rod is not None and self.pend_art is self.articulation:
            print("[sim] pendulum shares the board articulation: arm dof=%s rod dof=%s"
                  % (self.dof_arm, self.dof_rod))
        print("[sim] dofs: %s  alpha=%d beta=%d" % (names, self.dof_alpha, self.dof_beta))

        # The plate here is held far more rigidly than two hobby serial servos
        # hold the real one, and a rigid plate cannot be disturbed: measured, the
        # pendulum's reaction moved it 0.00002 deg. Softening the two tilt drives
        # is what lets that reaction reach the marble at all.
        ctrl = self.articulation.get_articulation_controller()
        kp, kd = ctrl.get_gains()
        idx = [self.dof_alpha, self.dof_beta]
        print("[sim] board drives as authored: stiffness %s damping %s"
              % ([float(kp[i]) for i in idx], [float(kd[i]) for i in idx]))
        if self.args.board_stiffness >= 0.0:
            kp = np.asarray(kp, dtype=np.float32).copy()
            kd = np.asarray(kd, dtype=np.float32).copy()
            for i in idx:
                kp[i] = self.args.board_stiffness
                if self.args.board_damping >= 0.0:
                    kd[i] = self.args.board_damping
            ctrl.set_gains(kps=kp, kds=kd)
            print("[sim] board drives softened to: stiffness %s damping %s"
                  % ([float(kp[i]) for i in idx], [float(kd[i]) for i in idx]))

        if self.args.image in ("camera", "composite") or self.args.view_every:
            from isaacsim.sensors.camera import Camera
            self.camera = Camera(prim_path=self.args.camera_prim,
                                 resolution=(self.args.width, self.args.height))
            self.camera.initialize()
            for _ in range(3):
                self.sim.step(render=True)
        if self.args.image == "composite":
            self._build_composite_background()
        if self.camera is not None:
            # Put a frame in the window before anyone asks for a step, so the
            # view endpoint shows the board while the learner is still starting
            # instead of answering "no frame yet" for two minutes.
            self.sim.step(render=True)
            self._stash_camera()
        self.rest_height = None
        self.reset()
        self._calibrate_rest_height()

    def _add_bare_floor(self):
        """A flat collider where the maze's own floor used to be.

        mazeCad is walls, holes, AND the floor in one mesh -- deactivating it
        for --hide-maze drops the marble's only support, and it just
        free-falls (measured: settles ~74mm below the wall tops, nowhere near
        a floor, instead of the couple of mm a real rest should read). This
        substitutes a plain box collider at the height the maze's floor sat
        at (wall tops minus the 15mm wall height, both taken from
        tag_state_estimation/core/maze_layout.py's real measurements), as a
        sibling of mazeCad under the same rigid body, so it tilts with the
        plate exactly as the maze surface did. _calibrate_rest_height() below
        then measures the marble's actual settle against it -- no need to
        get this exactly right by construction, only close enough for that
        settle to land near the marble's radius instead of in free fall.
        """
        wall_height_m = 0.015
        thickness_m = 0.004
        floor_top_local_y = self.mesh_hi[1] - wall_height_m
        center_mesh_local = Gf.Vec3d(
            (self.mesh_lo[0] + self.mesh_hi[0]) / 2.0,
            floor_top_local_y - thickness_m / 2.0,
            (self.mesh_lo[2] + self.mesh_hi[2]) / 2.0,
        )
        center_board_local = self.mesh_in_board.Transform(center_mesh_local)

        cube = UsdGeom.Cube.Define(self.stage, "/World/TAG/Board/BarePlateFloor")
        cube.CreateSizeAttr(1.0)
        prim = cube.GetPrim()
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(center_board_local))
        xform.AddScaleOp().Set(Gf.Vec3d(
            float(self.mesh_hi[0] - self.mesh_lo[0]),
            thickness_m,
            float(self.mesh_hi[2] - self.mesh_lo[2]),
        ))
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdGeom.Imageable(prim).MakeInvisible()  # a physics stand-in, not meant to be seen
        print("[sim] --hide-maze: added a bare flat-plate collider, "
              "board-local y=%.4f" % center_board_local[1])

    def _calibrate_rest_height(self):
        """Measure where a settled marble sits, instead of assuming.

        Placements used mesh_hi (the TOP OF THE WALLS) plus a radius, which is
        fine from the fixed start -- open floor there, so the marble drops in --
        but the spread starts sit anywhere along the path, and the path runs
        close to walls. A marble released at wall height above a wall lands on
        top of it and stays: the tops are flat and 7 degrees of tilt will not
        roll it off. That is the marble that "gets stuck and never moves".
        """
        for _ in range(int(0.6 / self.physics_dt)):
            self.sim.step(render=False)
        pos, _ = self.marble.get_world_pose()
        world = Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2]))
        local = self.mesh_world_now().GetInverse().Transform(world)
        self.rest_height = float(local[1])
        print("[sim] marble rests %.1f mm up the mesh (wall tops are %.1f mm); "
              "spread starts will use that height"
              % (self.rest_height * 1000.0, self.mesh_hi[1] * 1000.0))

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

    def _drive_pendulum(self):
        """One 200 Hz tick of the balance policy, if one is loaded.

        Held between ticks: the policy sees the world every 5 ms and its torque
        stays applied through the substeps, exactly as a zero-order hold on the
        real motor would.
        """
        if self.policy is None or self.dof_rod is None:
            return
        self._substep += 1
        if self._substep % self.policy_substeps != 1 % self.policy_substeps:
            # between ticks: hold the last torque rather than recompute
            if getattr(self, "last_pendulum_torque", None) is not None:
                self.pend_art.apply_action(ArticulationAction(
                    joint_efforts=np.array([self.last_pendulum_torque], dtype=np.float32),
                    joint_indices=np.array([self.dof_arm], dtype=np.int32)))
            return
        q = np.asarray(self.pend_art.get_joint_positions(), dtype=np.float64)
        dq = np.asarray(self.pend_art.get_joint_velocities(), dtype=np.float64)
        board_q = np.asarray(self.articulation.get_joint_positions(), dtype=np.float64)
        board_dq = np.asarray(self.articulation.get_joint_velocities(), dtype=np.float64)
        theta_up = float(np.arctan2(np.sin(q[self.dof_rod] - math.pi),
                                    np.cos(q[self.dof_rod] - math.pi)))
        sign = self.args.pendulum_theta_sign
        obs = self.policy.observe(
            theta_up=sign * theta_up,
            theta_dot=sign * float(dq[self.dof_rod]),
            phi=float(q[self.dof_arm]),
            phi_dot=float(dq[self.dof_arm]),
            # the rig's IMU reads the board's own tilt, which here is simply the
            # state of the two joints the maze policy is driving
            roll=float(board_q[self.dof_alpha]), pitch=float(board_q[self.dof_beta]),
            roll_rate=float(board_dq[self.dof_alpha]), pitch_rate=float(board_dq[self.dof_beta]),
        )
        action = self.policy.act(obs)
        torque = self.args.pendulum_torque_sign * self.policy.torque(action)
        self.last_pendulum_action = action
        self.last_pendulum_torque = torque
        self.pend_art.apply_action(ArticulationAction(
            joint_efforts=np.array([torque], dtype=np.float32),
            joint_indices=np.array([self.dof_arm], dtype=np.int32)))

    def pendulum_state(self):
        if self.mj_pendulum is not None:
            return self.mj_pendulum.state()
        """Pendulum state in the form the integration design asks for, or None.

        Angles come straight from the articulation. In this asset the rod is
        authored hanging, so a joint angle of 0 is DOWN; theta is reported
        measured from upright to match the firmware's convention, and `upright`
        follows from it.
        """
        if self.dof_rod is None:
            return None
        q = np.asarray(self.pend_art.get_joint_positions(), dtype=np.float64)
        dq = np.asarray(self.pend_art.get_joint_velocities(), dtype=np.float64)
        theta = float(np.arctan2(np.sin(q[self.dof_rod] - math.pi),
                                 np.cos(q[self.dof_rod] - math.pi)))
        theta_dot = float(dq[self.dof_rod])
        phi = float(q[self.dof_arm]) if self.dof_arm is not None else float("nan")
        phi_dot = float(dq[self.dof_arm]) if self.dof_arm is not None else float("nan")
        upright = 1.0 if abs(theta) < 0.35 else 0.0
        return {
            "action": float(getattr(self, "last_pendulum_action", 0.0)),
            "torque": float(getattr(self, "last_pendulum_torque", 0.0)),
            "theta": theta, "theta_dot": theta_dot,
            "phi": phi, "phi_dot": phi_dot, "upright": upright,
            # exactly the vector CYBERRUNNER_PENDULUM_INTEGRATION.md specifies
            "obs": [math.cos(theta), math.sin(theta),
                    float(np.clip(theta_dot / 15.0, -2.0, 2.0)), upright],
        }

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
        # Command ONLY the two tilt joints. The articulation also carries the
        # pendulum's arm and rod, and writing the full position vector told them
        # to hold station every step -- a software clamp that froze the pendulum
        # no matter what its drive or geometry allowed.
        self.tilt_target = np.array([math.radians(alpha_deg), math.radians(beta_deg)],
                                    dtype=np.float32)
        self._push_tilt()

    def _push_tilt(self):
        if getattr(self, "tilt_target", None) is None:
            return
        efforts = None
        if self.args.board_probe_torque:
            efforts = np.array([self.args.board_probe_torque, 0.0], dtype=np.float32)
        elif self.reaction_torque is not None:
            # The pendulum pushes back on the plate it stands on. Isaac drives
            # that pendulum kinematically, so this is the only path by which the
            # rod's motion can reach the marble. It rides on the same two joints
            # the servos hold, so what actually deflects is set by their drive
            # stiffness, not by this number alone.
            efforts = np.asarray(self.reaction_torque, dtype=np.float32)
        self.articulation.apply_action(ArticulationAction(
            joint_positions=self.tilt_target,
            joint_efforts=efforts,
            joint_indices=np.array([self.dof_alpha, self.dof_beta], dtype=np.int32)))

    def advance(self):
        # Render only the substep whose frame is actually sent. Rendering every
        # substep costs the same again per step and nobody ever sees the extra
        # frames -- it halved the step rate for nothing.
        steps = max(1, int(round(self.control_dt / self.physics_dt)))
        # Only 'camera' mode needs a fresh frame each step. Composite mode keeps a
        # camera object around purely to project the marble onto its one-off
        # backdrop, and rendering for it threw away the whole point.
        render = self.args.image == "camera"
        # A view frame is for the human, not the policy: render it on its own
        # schedule and keep it out of the observation.
        view = (self.args.view_every > 0 and self.camera is not None
                and self.total_steps % self.args.view_every == 0)
        for i in range(steps):
            self._push_tilt()
            self._drive_pendulum()
            self.sim.step(render=(render or view) and i == steps - 1)
        if self.mj_pendulum is not None:
            alpha, beta = self.board_angles_rad()
            self.mj_pendulum.step(alpha, beta, self.control_dt)
            if self.args.pendulum_reaction:
                # Read after stepping, so the plate feels it one control step
                # later. At 33 ms against a rod that leans over tenths of a
                # second, that lag is smaller than the effect it carries.
                r, p = self.mj_pendulum.reaction()
                s = self.args.pendulum_reaction_scale
                self.reaction_torque = (r * s, p * s)
            st = self.mj_pendulum.state()
            self.pend_samples += 1
            self.pend_hits += int(st["upright"])
            self.pend_worst = max(self.pend_worst, abs(math.degrees(st["theta"])))
            self._mirror_pendulum()

    def _mirror_pendulum(self):
        """Show in Isaac what MuJoCo is actually doing.

        The pendulum that balances lives in MuJoCo; the one in the USD is there
        to be looked at. Without this the render shows a rod hanging dead while
        the real one is holding itself upright -- which is exactly as confusing
        as it sounds.
        """
        if self.dof_rod is None or self.mj_pendulum is None:
            return
        st = self.mj_pendulum.state()
        q = np.asarray(self.pend_art.get_joint_positions(), dtype=np.float32)
        q[self.dof_rod] = math.pi + st["theta"]     # joint zero is hanging here
        if self.dof_arm is not None:
            q[self.dof_arm] = st["phi"]
        self.pend_art.set_joint_positions(q)
        self.pend_art.set_joint_velocities(np.zeros_like(q))

    def reset_xy(self):
        """Where to put the marble this episode, in lower-left board metres.

        Starting every episode at the beginning practises the opening thousands
        of times and the hard stretch only when the policy survives that far.
        Measured on this map: 74 of 200 episodes died between 40% and 50% of the
        path, so that is exactly where the data is thin.
        """
        if self.args.reset_mode != "spread":
            return RESET_XY_M
        self.reset_count += 1
        # Keep some episodes starting where the robot starts, or the opening is
        # forgotten and evaluation -- which always starts there -- collapses.
        if (self.reset_count % 10) < int(round(self.args.reset_from_start_prob * 10)):
            return RESET_XY_M
        lo, hi = self.reset_span
        # A golden-ratio sweep rather than a random draw: evenly spread in fact,
        # not merely in expectation, and reproducible without seeding anything.
        frac = lo + (hi - lo) * ((self.reset_count * 0.6180339887) % 1.0)
        index = int(np.searchsorted(self.path_cum, frac * self.path_cum[-1]))
        point = self.path_xy[min(index, len(self.path_xy) - 1)]
        return float(point[0]), float(point[1])

    def wall_segments(self):
        """Every wall as a segment, in lower-left board metres."""
        if getattr(self, "_wall_segs", None) is not None:
            return self._wall_segs
        segs = []
        layout = getattr(self.overlay, "layout", None) if self.overlay else None
        if layout:
            for x1, x2, y in np.asarray(layout["walls_h"], dtype=float):
                segs.append(((x1, y), (x2, y)))
            for y1, y2, x in np.asarray(layout["walls_v"], dtype=float):
                segs.append(((x, y1), (x, y2)))
            for s in np.asarray(layout.get("walls_angled", []), dtype=float):
                segs.append(((s[0], s[1]), (s[2], s[3])))
        self._wall_segs = [(np.asarray(a), np.asarray(b)) for a, b in segs]
        return self._wall_segs

    def clear_of_walls(self, xy, need=MARBLE_RADIUS_M + 0.0005):
        """Move a start point off any wall it would be born inside.

        The marble is placed at its resting height, which is halfway up a 15 mm
        wall, so a start point closer to a wall than the marble's radius spawns
        it interpenetrating. PhysX then shoves it out -- which is why the marble
        arrived up to 13 mm from where it was put, and occasionally stuck fast
        inside the wall instead. The shipped fixed start sits 4.0 mm from the top
        boundary wall, 2 mm inside a 6 mm marble.
        """
        segs = self.wall_segments()
        if not segs:
            return xy
        p = np.asarray(xy, dtype=float)
        for _ in range(6):
            worst_d, worst_c = None, None
            for a, b in segs:
                ab = b - a
                t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0)
                c = a + t * ab
                d = float(np.linalg.norm(p - c))
                if worst_d is None or d < worst_d:
                    worst_d, worst_c = d, c
            if worst_d >= need:
                break
            away = p - worst_c
            n = float(np.linalg.norm(away))
            # Dead centre on the segment gives no direction; step along its normal.
            away = away / n if n > 1e-9 else np.array([0.0, 1.0])
            p = worst_c + away * need
        moved = float(np.linalg.norm(p - np.asarray(xy, dtype=float)))
        if moved > 1e-6 and self.reset_count < 3:
            print("[sim] start moved %.1f mm clear of a wall: (%.5f, %.5f) -> (%.5f, %.5f)"
                  % (1000 * moved, xy[0], xy[1], p[0], p[1]))
        return float(p[0]), float(p[1])

    def reset(self):
        self.servo1.home()
        self.servo2.home()
        self.apply(0.0, 0.0)
        if self.policy is not None:
            self.policy.reset()
        if self.mj_pendulum is not None:
            self.mj_pendulum.reset(float(self.args.pendulum_start_upright or 0.0))
        if self.args.pendulum_start_upright is not None and self.dof_rod is not None:
            q = np.asarray(self.pend_art.get_joint_positions(), dtype=np.float32)
            q[self.dof_rod] = math.pi + float(self.args.pendulum_start_upright)
            q[self.dof_arm] = 0.0
            self.pend_art.set_joint_positions(q)
            self.pend_art.set_joint_velocities(np.zeros_like(q))
            # Read it back. A reset that quietly fails leaves the rod hanging,
            # the policy sees a state it has no answer for, and every later
            # measurement is about a situation that never happened.
            back = np.asarray(self.pend_art.get_joint_positions(), dtype=np.float64)
            placed = math.degrees(math.atan2(math.sin(back[self.dof_rod] - math.pi),
                                             math.cos(back[self.dof_rod] - math.pi)))
            if abs(placed) > 5.0:
                print("[sim] WARNING: asked for the rod upright, it reads %+.1f deg" % placed)
            elif self.reset_count < 2:
                print("[sim] rod placed %+.2f deg from upright" % placed)
        start_xy = self.clear_of_walls(self.reset_xy())
        # Same y flip as marble_board_xy, so the marble is placed where the robot
        # places it rather than at its mirror image.
        height = (self.rest_height if getattr(self, "rest_height", None) is not None
                  else self.mesh_hi[1] + MARBLE_RADIUS_M)
        origin = Gf.Vec3d(float(self.mesh_lo[0] + start_xy[0]),
                          float(height),
                          float(self.mesh_lo[2] + BOARD_HEIGHT_M - start_xy[1]))
        # The static transform, not the live one: reset runs before the physics
        # view is ready on the first call, and asking the plate for its pose then
        # segfaults. The plate is homed here anyway, so the two agree.
        world = self.mesh_xf.Transform(origin)
        self.marble.set_world_pose(position=np.array([world[0], world[1], world[2]],
                                                     dtype=np.float32))
        self.marble.set_linear_velocity(np.zeros(3, dtype=np.float32))
        self.marble.set_angular_velocity(np.zeros(3, dtype=np.float32))
        for _ in range(int(0.2 / self.physics_dt)):
            # Drive the pendulum through the settle too. An inverted rod diverges
            # with a ~70 ms time constant, so 0.2 s of free fall turns a 3 degree
            # release into a 46 degree one -- far outside what a hold policy can
            # catch, and it looks exactly like the policy failing.
            self._drive_pendulum()
            self.sim.step(render=False)

    def observation(self):
        alpha, beta = self.board_angles_rad()
        xy = self.marble_board_xy()
        image = self.gray64()
        pendulum = self.pendulum_state()
        if self.overlay is not None:
            self.overlay.state = {
                "x": (xy[0] + BOARD_WIDTH_M / 2.0) if xy else float("nan"),
                "y": (xy[1] + BOARD_HEIGHT_M / 2.0) if xy else float("nan"),
                "alpha": alpha, "beta": beta, "ball": xy is not None,
                "steps": self.episode_steps, "total": self.total_steps,
                "pendulum": pendulum,
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
        if pendulum is not None:
            # Extra keys, so a learner that does not know about the pendulum is
            # unaffected: env_tcp reads the fields it wants and ignores the rest.
            reply["pendulum"] = pendulum
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
                        pend = ""
                        if board.pend_samples:
                            pend = ("  pendulum upright %.0f%% of %d, worst %.0f deg"
                                    % (100.0 * board.pend_hits / board.pend_samples,
                                       board.pend_samples, board.pend_worst))
                            board.pend_hits = board.pend_samples = 0
                            board.pend_worst = 0.0
                        print("[sim] %d steps, %.1f steps/s (%.1fx real time)%s"
                              % (steps, rate, rate * board.control_dt, pend))
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
        except Exception:
            # Print here, not in main(): sim_app.close() tears down carb's stderr,
            # so a traceback raised past this point is written into a dead stream
            # and lost. That is why these crashes looked like clean exits.
            print("[sim] request failed:\n%s" % traceback.format_exc(), flush=True)
            raise
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
    except Exception:
        print("[sim] fatal:\n%s" % traceback.format_exc(), flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        sim_app.close()


main()
