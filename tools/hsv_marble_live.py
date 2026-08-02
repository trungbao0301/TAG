#!/usr/bin/env python3
"""Live view of exactly what the HSV marble source sees, with tunable bounds.

Subscribe-only, like marble_hsv_picker.py -- it never opens the camera device,
so it is safe to run beside the estimator.

The difference from the picker is that this runs the REAL pipeline: the maze
polygon is rebuilt every frame from the tracked moving dots, the same
morphology runs, and the same area gate decides. So a setting that looks good
here is a setting that will behave the same way inside the estimator.

Three panels:
    left    the frame, with the maze polygon, the accepted marble, and every
            blob the gate rejected (so you can see WHAT you are excluding)
    middle  the colour mask after the maze clip and the morphology
    right   the readout: bounds, blob areas, and the hit rate so far

Trackbars change the bounds live. Press p to print a block ready to paste into
tag_state_estimation/core/hsv_marble.py.

    python3 tools/hsv_marble_live.py
    python3 tools/hsv_marble_live.py --local_disc 32   # mimic the local search
"""
import argparse
import os
import sys

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "tag_state_estimation"))

from tag_state_estimation.core import hsv_marble  # noqa: E402
from tag_state_estimation.core.detection import Detector  # noqa: E402
from tag_state_estimation.core.ai_map_state import MarkerQuadGuard  # noqa: E402

WINDOW = "HSV marble - live"
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


class Frames(Node):
    def __init__(self, topic):
        super().__init__("hsv_marble_live")
        self.bridge = CvBridge()
        self.frame = None
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(Image, topic, self.cb, qos)

    def cb(self, msg):
        self.frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")


def nothing(_):
    pass


def default_values(args):
    lo, hi = hsv_marble.DEFAULT_HSV_LO, hsv_marble.DEFAULT_HSV_HI
    return {
        "H min": lo[0], "H max": hi[0],
        "S min": lo[1], "S max": hi[1],
        "V min": lo[2], "V max": hi[2],
        "area min": int(hsv_marble.DEFAULT_AREA_PX2[0]),
        "inset mm": int(round(hsv_marble.DEFAULT_INSET_M * 1000)),
        "disc px (0=off)": int(args.local_disc),
    }


TOPS = {
    "H min": 179, "H max": 179, "S min": 255, "S max": 255,
    "V min": 255, "V max": 255, "area min": 200, "inset mm": 20,
    "disc px (0=off)": 200,
}


def build_trackbars(values):
    """Create the window and its trackbars, seeded from ``values``."""
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    for name, top in TOPS.items():
        cv2.createTrackbar(name, WINDOW, int(values[name]), top, nothing)
    # GTK does not attach the trackbars until the window has been pumped once,
    # and getTrackbarPos answers -1 in the meantime -- which is the same thing
    # it says about a window the user has closed.
    cv2.imshow(WINDOW, np.zeros((16, 16, 3), np.uint8))
    cv2.waitKey(60)


def read_trackbars():
    """Current bounds, or None once the window has been closed.

    getTrackbarPos returns -1 for a window that no longer exists, and feeding
    that into inRange only produced a confusing uint8 overflow further down.
    """
    try:
        values = {n: cv2.getTrackbarPos(n, WINDOW) for n in TOPS}
    except cv2.error:
        return None
    if any(v < 0 for v in values.values()):
        return None
    return values


def unpack(values):
    return (
        (values["H min"], values["S min"], values["V min"]),
        (values["H max"], values["S max"], values["V max"]),
        float(values["area min"]),
        values["inset mm"] / 1000.0,
        float(values["disc px (0=off)"]),
    )


def panel(lines, height, width=330):
    img = np.full((height, width, 3), 32, np.uint8)
    y = 26
    for text, colour in lines:
        cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                    colour, 1, cv2.LINE_AA)
        y += 21
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/tag_camera/image")
    parser.add_argument("--markers", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "install",
        "tag_state_estimation", "share", "tag_state_estimation", "markers.csv"))
    parser.add_argument("--local_disc", type=float, default=0.0,
                        help="Restrict the search to this radius around the "
                             "previous hit, as the estimator does.")
    parser.add_argument("--scale", type=float, default=1.3)
    args, ros_args = parser.parse_known_args()

    markers = np.loadtxt(args.markers, delimiter=",")
    tracker = Detector(markers[4:], ai_mode="off", corner_subimage_half_size=12)
    guard = MarkerQuadGuard(markers[4:][:, ::-1].astype(np.float64),
                            mode="moving", occlusion_grace_sec=0.20,
                            max_speed_px_s=300.0, acquire_radius_px=14.0)

    rclpy.init(args=ros_args)
    node = Frames(args.topic)
    values = default_values(args)
    build_trackbars(values)
    print("Trackbars adjust the bounds. p = print a paste-ready block, q = quit.")

    seen = hits = unreadable = 0
    last_hit = None
    stamp = 0.0

    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.05)
        if node.frame is None:
            continue
        frame = node.frame
        stamp += 1 / 45.0
        settings = read_trackbars()
        if settings is None:
            # A single -1 is the toolkit still catching up right after the window
            # is created; a run of them means the user closed it, so quit rather
            # than reopening a window they just dismissed.
            unreadable += 1
            if unreadable > 8:
                print("window closed")
                break
            cv2.waitKey(30)
            continue
        unreadable = 0
        values = settings
        lo, hi, area_min, inset_m, disc_px = unpack(values)

        moving_rc, quad_ok, quad_status = guard.update(
            tracker.detect_corners(frame.copy()), tracker.corner_found, stamp)
        tracker.corners = moving_rc.astype(np.float32)
        tracker.corners_missing = False

        display = frame.copy()
        mask = np.zeros(frame.shape[:2], np.uint8)
        blob_mask = np.zeros(frame.shape[:2], np.uint8)
        accepted, rejected, chosen_hsv = None, [], None
        best, px_per_mm = 0.0, float("nan")

        if quad_ok:
            polygon = hsv_marble.maze_polygon_px(moving_rc, inset_m=inset_m)
            if polygon is not None:
                area = np.zeros(frame.shape[:2], np.uint8)
                cv2.fillConvexPoly(area, polygon.astype(np.int32), 255)
                if disc_px > 0 and last_hit is not None:
                    disc = np.zeros_like(area)
                    cv2.circle(disc, tuple(np.round(last_hit).astype(int)),
                               int(disc_px), 255, -1)
                    area = cv2.bitwise_and(area, disc)
                    cv2.circle(display, tuple(np.round(last_hit).astype(int)),
                               int(disc_px), (255, 200, 0), 1)
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask = cv2.bitwise_and(
                    cv2.inRange(hsv, np.asarray(lo, np.uint8),
                                np.asarray(hi, np.uint8)), area)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)
                n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
                best, best_label = 0.0, -1
                for i in range(1, n):
                    a = float(stats[i, cv2.CC_STAT_AREA])
                    if area_min <= a <= hsv_marble.DEFAULT_AREA_PX2[1] and a > best:
                        if accepted is not None:
                            rejected.append((best, accepted))
                        best, accepted, best_label = a, cents[i], i
                    else:
                        rejected.append((a, cents[i]))
                if accepted is not None:
                    blob_mask = (labels == best_label).astype(np.uint8) * 255
                    chosen_hsv = hsv[int(accepted[1]), int(accepted[0])]
                    last_hit = np.asarray(accepted, float)
                # px per mm, live from the dot spacing, so the readout can say
                # whether the blob is really the size of a marble.
                span = np.linalg.norm(moving_rc[1] - moving_rc[0])
                px_per_mm = span / 269.0 if span > 1.0 else float("nan")
                cv2.polylines(display, [polygon.astype(np.int32)], True,
                              (0, 220, 220), 1)
            seen += 1
            hits += accepted is not None

        for a, c in rejected:
            p = tuple(np.round(c).astype(int))
            cv2.drawMarker(display, p, (0, 0, 255), cv2.MARKER_TILTED_CROSS, 11, 1)
            cv2.putText(display, f"{a:.0f}", (p[0] + 7, p[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1, cv2.LINE_AA)
        if accepted is not None:
            p = tuple(np.round(accepted).astype(int))
            # Outline the pixels actually detected, not a fixed circle. A fixed
            # circle is meaningless next to a 12 px marble -- it reads as the
            # detector having found something far larger than it did.
            contours, _ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(display, contours, -1, (0, 255, 0), 1)
            cv2.drawMarker(display, p, (0, 255, 0), cv2.MARKER_CROSS, 7, 1)
            cv2.putText(display, f"{best:.0f} px2", (p[0] + 12, p[1] - 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

        ok, warn, bad = (180, 255, 180), (0, 210, 255), (110, 110, 255)
        lines = [
            (f"H {lo[0]:3d}-{hi[0]:3d}   S {lo[1]:3d}-{hi[1]:3d}", ok),
            (f"V {lo[2]:3d}-{hi[2]:3d}   area >= {area_min:.0f} px2", ok),
            (f"inset {inset_m*1000:.0f} mm   disc {disc_px:.0f} px", ok),
            ("", ok),
            (f"moving quad: {quad_status}", ok if quad_ok else bad),
            (f"marble blob: {best:.0f} px2" if accepted is not None
             else "marble: NOT FOUND", ok if accepted is not None else bad),
            (f"blob r={np.sqrt(best/np.pi):.1f}px "
             f"({np.sqrt(best/np.pi)/px_per_mm:.1f}mm) vs marble 6.0mm"
             if accepted is not None and np.isfinite(px_per_mm) else "", warn),
            (f"margin over gate: {best/area_min:.1f}x"
             if accepted is not None and area_min > 0 else "", warn),
            (f"HSV at centre: {chosen_hsv[0]} {chosen_hsv[1]} {chosen_hsv[2]}"
             if chosen_hsv is not None else "", ok),
            (f"other blobs: {len(rejected)}", ok if not rejected else warn),
            ("", ok),
            (f"hit rate: {100.0*hits/max(1,seen):.1f}%  ({hits}/{seen})",
             ok if hits == seen else warn),
            ("", ok),
            ("p = print block", (200, 200, 200)),
            ("r = reset hit rate", (200, 200, 200)),
            ("q = quit", (200, 200, 200)),
        ]

        # A 12 px marble cannot be judged at 640x400. Magnify the neighbourhood
        # of the last hit, with the detected outline drawn on it, so the blob
        # can actually be compared against the marble it is supposed to be.
        zoom = np.full((display.shape[0], display.shape[0], 3), 32, np.uint8)
        if last_hit is not None:
            half = 30
            cx, cy = int(round(last_hit[0])), int(round(last_hit[1]))
            h_, w_ = display.shape[:2]
            x0, x1 = max(0, cx - half), min(w_, cx + half)
            y0, y1 = max(0, cy - half), min(h_, cy + half)
            if x1 > x0 and y1 > y0:
                side = display.shape[0]
                zoom = cv2.resize(display[y0:y1, x0:x1], (side, side),
                                  interpolation=cv2.INTER_NEAREST)
                cv2.putText(zoom, f"zoom x{side/(x1-x0):.0f}", (8, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                            cv2.LINE_AA)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined = np.hstack((display, zoom, mask_bgr,
                              panel(lines, display.shape[0])))
        if args.scale != 1.0:
            combined = cv2.resize(combined, None, fx=args.scale, fy=args.scale,
                                  interpolation=cv2.INTER_NEAREST)
        cv2.imshow(WINDOW, combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            seen = hits = 0
        if key == ord("p"):
            print("\n# paste into tag_state_estimation/core/hsv_marble.py")
            print(f"DEFAULT_HSV_LO = {tuple(int(v) for v in lo)}")
            print(f"DEFAULT_HSV_HI = {tuple(int(v) for v in hi)}")
            print(f"DEFAULT_INSET_M = {inset_m:.4f}")
            print(f"DEFAULT_AREA_PX2 = ({area_min:.1f}, "
                  f"{hsv_marble.DEFAULT_AREA_PX2[1]:.1f})")
            print(f"# hit rate {100.0*hits/max(1,seen):.1f}% over {seen} frames\n")

    cv2.destroyAllWindows()
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
