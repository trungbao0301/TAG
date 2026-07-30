#!/usr/bin/env python3
"""Pick by hand which regions of the board must not be credited.

The build-time jump-detection pass blanks the ridge between two corridors, which
is what stops a 25 mm sideways hop from claiming 811 mm of path. It is also
indiscriminate: it removed 68% of the legitimately walkable cells, and the
automatic repair in tools/rebuild_path_grid.py still leaves 2.5% of cells that
are within 15 mm of the path and can see it, because the only route into them
crosses a cell whose index is too far away.

Nobody has to guess at that. Drag rectangles over the places a marble could
actually hop between corridors; everything else the walls allow gets credited.

    python3 tools/select_shortcut_zones.py                  # draw and save
    python3 tools/select_shortcut_zones.py --apply          # re-apply saved zones

  drag left button   add a blocked rectangle
  u                  undo the last one
  c                  clear all
  h                  toggle the path and the credited overlay
  s                  save zones, write the grid, print the verification
  q / Esc            quit without writing

Colours, and what each means once training runs:

  green     credited -- the marble is given progress here
  RED       a zone you drew. Those cells become -1, so env_tcp sees off_path and
            after TAG_OFFPATH_CONFIRM_STEPS frames ends the episode and charges
            TAG_OFFPATH_PENALTY. This is the shortcut trap.
  magenta   a hop still payable: two credited cells adjacent to each other whose
            path indices are more than 0.057 m apart, so the marble can cross
            between corridors and be paid for it. Cover these in red.
  white     walls, from the layout
  yellow    the path centreline

Zones live in tools/shortcut_zones.json in board millimetres, so they can be
edited by hand and survive a grid rebuild.
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tag_dreamer"))
# maze_layout lives in the estimator package and is where the hole centres are.
sys.path.insert(0, os.path.join(HERE, "..", "tag_state_estimation"))
sys.path.insert(0, HERE)

WINDOW = "shortcut zones -- drag to block, s to save, q to quit"
JUMP_M = 0.057


def load(path_pkl):
    import importlib
    with open(path_pkl, "rb") as fh:
        p = pickle.load(fh)
    mod = importlib.import_module("tag_dreamer.tag_layout_custom")
    layout = next(
        v for v in vars(mod).values() if isinstance(v, dict) and "walls_h" in v
    )
    return p, layout


def jump_mask(grid):
    """Cells taking part in an adjacent pair that jumps more than JUMP_M of path.

    These are the hops still payable, so they are what has to be covered. Drawn
    on the overlay rather than left to be guessed at -- knowing where they are is
    the whole reason to pick zones by hand instead of by threshold.
    """
    thr = int(JUMP_M / 0.0002)
    g = grid.astype(np.int64)
    mask = np.zeros(grid.shape, dtype=bool)
    count = 0
    for sl_a, sl_b in (
        ((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
        ((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
    ):
        a, b = g[sl_a], g[sl_b]
        bad = (a != -1) & (b != -1) & (np.abs(a - b) > thr)
        count += int(bad.sum())
        mask[sl_a] |= bad
        mask[sl_b] |= bad
    return mask, count


def jump_pairs(grid):
    return jump_mask(grid)[1]


def solid_mask(p, layout, wall_mm, hole_margin_mm, holes_mode="allow"):
    """Cells the marble centre cannot be in: wall bodies and hole mouths.

    Walls are stored as pairs of parallel segments 1.9-4.4 mm apart, so a wall is
    a thin rectangle rather than a line. Stroking each segment with wall_mm of
    thickness covers both faces and the body between them.
    """
    import cv2
    cell = p.distance
    ny, nx = p.closest_idx.shape
    mask = np.zeros((ny, nx), np.uint8)
    t = max(1, int(round(wall_mm / 1000.0 / cell)))
    for x0, x1_, y in np.asarray(layout["walls_h"], dtype=float):
        cv2.line(mask, (int(x0 / cell), int(y / cell)),
                 (int(x1_ / cell), int(y / cell)), 1, t)
    for y0, y1_, x in np.asarray(layout["walls_v"], dtype=float):
        cv2.line(mask, (int(x / cell), int(y0 / cell)),
                 (int(x / cell), int(y1_ / cell)), 1, t)
    if holes_mode != "block":
        # Rolling across a hole mouth without falling in is legal play, so the
        # mouth stays credited by default. Blocking it would end the episode on
        # a crossing, which is the opposite of that.
        return mask.astype(bool)
    try:
        import importlib
        ml = importlib.import_module("tag_state_estimation.core.maze_layout")
        holes = np.asarray(ml.HOLES_LOWER_LEFT_M, dtype=float)
        radii = np.asarray(ml.HOLE_RADII_M, dtype=float).ravel()
        for i, (hx, hy) in enumerate(holes):
            r = radii[min(i, len(radii) - 1)] + hole_margin_mm / 1000.0
            cv2.circle(mask, (int(hx / cell), int(hy / cell)),
                       max(1, int(r / cell)), 1, -1)
    except Exception as exc:                       # noqa: BLE001
        print(f"  holes not excluded ({exc})")
    return mask.astype(bool)


def build(p, layout, zones_mm, base="visibility", wall_mm=3.0,
          hole_margin_mm=0.0, holes_mode="allow", seal=True):
    """Credit cells, minus the chosen rectangles.

    base="all" credits every cell, so the only thing that is ever off-path is
    what has been painted. base="open" also excludes wall bodies, and
    base="visibility" only credits where the walls let a path point be seen.
    """
    import rebuild_path_grid as R
    from scipy.spatial import cKDTree

    cell = p.distance
    ny, nx = p.closest_idx.shape

    ys, xs = np.meshgrid(np.arange(ny) * cell, np.arange(nx) * cell, indexing="ij")
    _, near = cKDTree(p.points).query(np.column_stack([xs.ravel(), ys.ravel()]))
    near = near.reshape(ny, nx)

    if base == "all":
        # Everything credited, nothing excluded. The operator paints every trap,
        # including walls if they want them to end the episode. Until something
        # is painted the marble is never off-path.
        allowed = np.ones((ny, nx), dtype=bool)
    elif base == "open":
        allowed = ~solid_mask(p, layout, wall_mm, hole_margin_mm, holes_mode)
    else:
        x1, x2, y1, y2 = R.boxes(p, layout)
        vis = R.visibility(p, x1, x2, y1, y2)
        npx, npy = p.points[near, 0], p.points[near, 1]
        allowed = vis & (npx >= x1) & (npx <= x2) & (npy >= y1) & (npy <= y2)
    in_box = allowed

    blocked = np.zeros((ny, nx), dtype=bool)
    for x0, y0, x3, y3 in zones_mm:
        c0, c1 = int(min(x0, x3) / 1000 / cell), int(max(x0, x3) / 1000 / cell)
        r0, r1 = int(min(y0, y3) / 1000 / cell), int(max(y0, y3) / 1000 / cell)
        blocked[max(0, r0):r1 + 1, max(0, c0):c1 + 1] = True

    credited = allowed & ~blocked
    if seal:
        # Also blank the boundary lines themselves. A magenta cell is one side of
        # an adjacent pair whose path indices are more than JUMP_M apart, i.e. a
        # crossing between two corridors, so blanking it is what stops the marble
        # being paid for the crossing. Blanking can expose new pairs one cell
        # further out, so repeat until none are left; this converges because the
        # credited set only ever shrinks.
        for _ in range(200):
            hops, n = jump_mask(np.where(credited, near, -1))
            if not n:
                break
            credited &= ~hops
    return np.where(credited, near, -1), credited, allowed


def report(p, grid, allowed, marble=None):
    cell = p.distance
    ny, nx = grid.shape
    ok = grid != -1
    r = np.clip((p.points[:, 1] / cell).astype(int), 0, ny - 1)
    c = np.clip((p.points[:, 0] / cell).astype(int), 0, nx - 1)
    print(f"  cells credited   : {100 * ok.mean():5.1f}%  "
          f"(the chosen base allows at most {100 * allowed.mean():.1f}%)")
    print(f"  centreline        : {100 * ok[r, c].mean():5.1f}%   want 100%")
    if marble is not None:
        mr = np.clip((marble[:, 1] / cell).astype(int), 0, ny - 1)
        mc = np.clip((marble[:, 0] / cell).astype(int), 0, nx - 1)
        print(f"  recorded marble   : {100 * ok[mr, mc].mean():5.1f}%")
    n = jump_pairs(grid)
    print(f"  jump pairs        : {n:7d}   "
          f"{'ok' if n == 0 else 'each one is a hop still payable'}")
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--path",
        default=os.path.join(HERE, "..", "tag_dreamer", "data", "path_custom.pkl"),
        help="grid to read geometry from; --apply writes back here",
    )
    ap.add_argument("--zones", default=os.path.join(HERE, "shortcut_zones.json"))
    ap.add_argument("--out", default=None, help="grid to write (default: --path)")
    ap.add_argument(
        "--apply", action="store_true",
        help="rebuild from the saved zones without opening a window",
    )
    ap.add_argument("--logdir", default="", help="a run logdir, to score real play")
    ap.add_argument("--scale", type=float, default=2.4, help="display px per mm")
    ap.add_argument(
        "--base", choices=("all", "visibility", "open"), default="all",
        help="all: the whole board credited, you paint every trap. open: minus "
             "wall bodies. visibility: only where the walls let a path point be "
             "seen.",
    )
    ap.add_argument(
        "--wall_mm", type=float, default=3.0,
        help="stroke width for wall bodies; walls are pairs of segments "
             "1.9-4.4 mm apart so 3 mm covers both faces and the body",
    )
    ap.add_argument("--hole_margin_mm", type=float, default=0.0)
    ap.add_argument(
        "--no_seal", action="store_true",
        help="leave the corridor-to-corridor boundary lines credited. They are "
             "blanked by default, so crossing one is off-path.",
    )
    ap.add_argument(
        "--holes", choices=("allow", "block"), default="allow",
        help="allow: a hole mouth stays credited, so rolling across one without "
             "falling in is legal. block: entering one ends the episode.",
    )
    args = ap.parse_args()

    p, layout = load(args.path)
    out_path = args.out or args.path

    marble = None
    if args.logdir:
        import rebuild_path_grid as R
        marble = R.marble_positions(args.logdir, layout)

    zones = []
    if os.path.exists(args.zones):
        zones = json.loads(open(args.zones).read()).get("zones_mm", [])
        print(f"  loaded {len(zones)} zone(s) from {args.zones}")

    def write(zones_now):
        grid, _, allowed = build(p, layout, zones_now, args.base, args.wall_mm, args.hole_margin_mm, args.holes,
                                     not args.no_seal)
        n = report(p, grid, allowed, marble)
        with open(args.zones, "w") as fh:
            json.dump({"zones_mm": zones_now}, fh, indent=2)
            fh.write("\n")
        p.closest_idx = grid
        with open(out_path, "wb") as fh:
            pickle.dump(p, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  wrote {os.path.abspath(out_path)}")
        if n:
            print("  WARNING: shortcuts remain payable; add zones over them")

    if args.apply:
        write(zones)
        return 0

    import cv2

    cell = p.distance
    ny, nx = p.closest_idx.shape
    bw_mm, bh_mm = 1000 * nx * cell, 1000 * ny * cell
    s = args.scale
    W, H = int(bw_mm * s), int(bh_mm * s)

    def to_px(x_mm, y_mm):
        return int(x_mm * s), int(H - y_mm * s)

    def recompute(zones_now):
        grid, credited, allowed = build(p, layout, zones_now, args.base, args.wall_mm, args.hole_margin_mm, args.holes,
                                     not args.no_seal)
        hops, n = jump_mask(grid)
        return credited, allowed, hops, n

    credited, allowed, hops, n_hops = recompute(zones)

    def base_image(show_overlay):
        img = np.full((H, W, 3), 32, np.uint8)
        if show_overlay:
            small = np.zeros((ny, nx, 3), np.uint8)
            small[allowed] = (40, 110, 40)            # walkable, credited
            small[allowed & ~credited] = (40, 40, 190)  # blocked by a zone
            small[hops] = (255, 0, 255)                # hop still payable
            img = cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST)
            img = np.flipud(img).copy()
        for x0, x1_, y in np.asarray(layout["walls_h"]):
            cv2.line(img, to_px(1000 * x0, 1000 * y), to_px(1000 * x1_, 1000 * y),
                     (255, 255, 255), 1)
        for y0, y1_, x in np.asarray(layout["walls_v"]):
            cv2.line(img, to_px(1000 * x, 1000 * y0), to_px(1000 * x, 1000 * y1_),
                     (255, 255, 255), 1)
        if show_overlay:
            pts = np.array([to_px(1000 * q[0], 1000 * q[1]) for q in p.points[::12]])
            cv2.polylines(img, [pts], False, (0, 220, 255), 1)
        return img

    state = {"drag": None, "overlay": True, "zones": list(zones)}

    def on_mouse(event, px, py, flags, _):
        x_mm, y_mm = px / s, (H - py) / s
        if event == cv2.EVENT_LBUTTONDOWN:
            state["drag"] = [x_mm, y_mm]
        elif event == cv2.EVENT_LBUTTONUP and state["drag"]:
            x0, y0 = state["drag"]
            state["drag"] = None
            if abs(x_mm - x0) > 1 and abs(y_mm - y0) > 1:
                state["zones"].append([round(x0, 1), round(y0, 1),
                                       round(x_mm, 1), round(y_mm, 1)])
        state["cursor"] = (x_mm, y_mm)

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, on_mouse)
    print("  drag to block a region | u undo | c clear | h overlay | s save | q quit")

    shown = list(state["zones"])
    while True:
        if state["zones"] != shown:
            # Compare by value, not by length: undo followed by a different
            # rectangle leaves the count unchanged but the grid different.
            # Recomputing every frame would be wasted work.
            shown = list(state["zones"])
            credited, allowed, hops, n_hops = recompute(shown)
        img = base_image(state["overlay"])
        for x0, y0, x1_, y1_ in state["zones"]:
            cv2.rectangle(img, to_px(x0, y0), to_px(x1_, y1_), (0, 0, 255), 2)
        cv2.putText(
            img,
            f"{len(state['zones'])} zone(s)   "
            f"magenta = {n_hops} hop(s) still payable, cover them   "
            f"credited {100 * credited.mean():.1f}%",
            (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0, 255, 0) if n_hops == 0 else (255, 255, 255), 1,
        )
        cv2.putText(
            img,
            "green = credited   RED = marble entering here ends the episode "
            "and is charged TAG_OFFPATH_PENALTY",
            (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1,
        )
        cv2.imshow(WINDOW, img)
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            print("  quit without writing")
            break
        if key == ord("u") and state["zones"]:
            state["zones"].pop()
        elif key == ord("c"):
            state["zones"].clear()
        elif key == ord("h"):
            state["overlay"] = not state["overlay"]
        elif key == ord("s"):
            write(state["zones"])
            credited, allowed, hops, n_hops = recompute(state["zones"])
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
