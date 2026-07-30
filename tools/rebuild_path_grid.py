#!/usr/bin/env python3
"""Give the marble back the corridor jump-detection over-painted.

path_custom.pkl credits only 22.5% of cells. The visibility test -- can this
cell see any path point inside the box the walls leave it -- allows 72.3%, so
jump-detection removed 49.8 percentage points, and that includes four stretches
of the path's own centreline. Those cells are where the marble actually rolls.

Not all of them can come back. Jump-detection exists to blank the ridge between
two different corridors, which is the route a shortcut hop takes: index 0 sits
24.9 mm from index 4055, 811 mm further along the path. A ridge is always more
than 10 mm from both corridors it separates, while a cell the marble legitimately
occupies is within a few mm of its own corridor, so distance to the nearest path
point separates the two.

This reports what each threshold buys and costs, and with --write saves the
rebuilt grid. Three things decide whether a threshold is acceptable:

  * centreline credited -- must be 100%, or the path runs through dead reward
  * marble frames credited -- how much of real recorded play now scores
  * jump pairs -- adjacent credited cells whose index differs by more than
    0.057 m of path. Every one of these is a shortcut the grid no longer blocks,
    so this must stay at 0.

    python3 tools/rebuild_path_grid.py                    # report only
    python3 tools/rebuild_path_grid.py --write --tol_mm 6 # save the rebuilt grid
"""
import argparse
import glob
import os
import pickle
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tag_dreamer"))

JUMP_M = 0.057   # same threshold the original build script used


def wall_bounds(coords_along, spans, ball_r, n_cells, cell):
    """Nearest wall coordinate each side of every cell, per perpendicular line."""
    n_perp = len(coords_along)
    above = np.full((n_perp, n_cells), np.inf)
    below = np.full((n_perp, n_cells), -np.inf)
    positions = np.arange(n_cells) * cell
    for i, along in enumerate(coords_along):
        covering = spans[
            (along >= spans[:, 0] - ball_r) & (along <= spans[:, 1] + ball_r), 2
        ]
        if covering.size == 0:
            continue
        covering = np.sort(covering)
        at = np.searchsorted(covering, positions, side="left")
        has_above = at < covering.size
        above[i, has_above] = covering[at[has_above]]
        has_below = at > 0
        below[i, has_below] = covering[at[has_below] - 1]
    return above, below


def boxes(p, layout):
    """The wall-bounded box around every cell, as (x1, x2, y1, y2) grids."""
    cell = p.distance
    ny, nx = p.closest_idx.shape
    wh = np.asarray(layout["walls_h"], dtype=float)
    wv = np.asarray(layout["walls_v"], dtype=float)
    y_above, y_below = wall_bounds(np.arange(nx) * cell, wh, p.wall_r, ny, cell)
    x_above, x_below = wall_bounds(np.arange(ny) * cell, wv, p.wall_r, nx, cell)
    return x_below, x_above, y_below.T, y_above.T


def visibility(p, x1, x2, y1, y2):
    """Whether each cell's box holds any path point, via an integral image."""
    cell = p.distance
    ny, nx = p.closest_idx.shape
    occ = np.zeros((ny, nx), dtype=np.int32)
    occ[
        np.clip((p.points[:, 1] / cell).astype(int), 0, ny - 1),
        np.clip((p.points[:, 0] / cell).astype(int), 0, nx - 1),
    ] = 1
    integral = np.zeros((ny + 1, nx + 1), dtype=np.int32)
    integral[1:, 1:] = occ.cumsum(0).cumsum(1)

    def edge(bound, limit, ceil):
        out = np.where(
            np.isfinite(bound),
            np.ceil(bound / cell) if ceil else np.floor(bound / cell),
            0 if ceil else limit - 1,
        )
        return np.clip(out, 0, limit - 1).astype(int)

    r0, r1 = edge(y1, ny, True), edge(y2, ny, False)
    c0, c1 = edge(x1, nx, True), edge(x2, nx, False)
    count = (
        integral[r1 + 1, c1 + 1] - integral[r0, c1 + 1]
        - integral[r1 + 1, c0] + integral[r0, c0]
    )
    return (count > 0) & (r1 >= r0) & (c1 >= c0)


def jump_pairs(grid):
    """Adjacent credited cells whose credited index differs by more than JUMP_M."""
    thr = int(JUMP_M / 0.0002)
    total = 0
    for a, b in ((grid[:-1, :], grid[1:, :]), (grid[:, :-1], grid[:, 1:])):
        ok = (a != -1) & (b != -1)
        total += int((np.abs(a[ok].astype(np.int64) - b[ok].astype(np.int64)) > thr).sum())
    return total


def marble_positions(logdir, layout):
    """Recorded marble positions in board metres, from a run's replay chunks."""
    if not logdir:
        return None
    bw, bh = layout["board_width"], layout["board_height"]
    out, seen = [], set()
    for f in sorted(glob.glob(os.path.join(logdir, "replay", "*.npz"))):
        m = re.search(r"-([A-Za-z0-9]{22})-(\d+)\.npz$", f)
        if not m or m.group(1) in seen:
            continue
        seen.add(m.group(1))
        z = np.load(f)
        out.append(z["states"][: int(m.group(2)), 2:4] * np.array([bw, bh]))
    if not out:
        return None
    pos = np.concatenate(out)
    return pos[np.isfinite(pos).all(axis=1)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--path", default=os.path.join(HERE, "..", "tag_dreamer", "data", "path_custom.pkl")
    )
    ap.add_argument("--logdir", default="", help="a run logdir, to score real play")
    ap.add_argument("--tol_mm", type=float, default=None, help="threshold to write")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from scipy.spatial import cKDTree
    import importlib

    with open(args.path, "rb") as fh:
        p = pickle.load(fh)
    mod = importlib.import_module("tag_dreamer.tag_layout_custom")
    layout = next(
        v for v in vars(mod).values() if isinstance(v, dict) and "walls_h" in v
    )

    cell = p.distance
    ny, nx = p.closest_idx.shape
    x1, x2, y1, y2 = boxes(p, layout)
    vis = visibility(p, x1, x2, y1, y2)
    stored = p.closest_idx

    # Nearest path point for every cell, and whether it is inside that cell's box.
    ys, xs = np.meshgrid(np.arange(ny) * cell, np.arange(nx) * cell, indexing="ij")
    tree = cKDTree(p.points)
    dist, near = tree.query(np.column_stack([xs.ravel(), ys.ravel()]))
    dist = dist.reshape(ny, nx)
    near = near.reshape(ny, nx)
    npx = p.points[near, 0]
    npy = p.points[near, 1]
    in_box = (npx >= x1) & (npx <= x2) & (npy >= y1) & (npy <= y2)

    pos = marble_positions(args.logdir, layout)
    centre_r = np.clip((p.points[:, 1] / cell).astype(int), 0, ny - 1)
    centre_c = np.clip((p.points[:, 0] / cell).astype(int), 0, nx - 1)

    def score(grid, label):
        ok = grid != -1
        line = f"  {label:<22} cells {100 * ok.mean():5.1f}%"
        line += f"   centreline {100 * ok[centre_r, centre_c].mean():5.1f}%"
        if pos is not None:
            r = np.clip((pos[:, 1] / cell).astype(int), 0, ny - 1)
            c = np.clip((pos[:, 0] / cell).astype(int), 0, nx - 1)
            line += f"   marble {100 * ok[r, c].mean():5.1f}%"
        line += f"   jump pairs {jump_pairs(grid):>7d}"
        print(line)

    def rebuild(tol_m):
        grid = stored.copy()
        add = (stored == -1) & vis & in_box & (dist <= tol_m)
        grid[add] = near[add]
        return grid, int(add.sum())

    def grow():
        """Widen the credited region while keeping zero jump pairs by construction.

        A fixed distance threshold is the wrong tool: corridors are not all the
        same width, so 10 mm leaves the wide ones -- the strip between the
        leftmost corridor and the board edge is 14-19 mm from its own centreline
        -- uncredited, while 12 mm already breaks the guarantee elsewhere.

        Instead start from the centreline and add a cell only when EVERY
        credited cell it touches is within JUMP_M of it. Admitting on "some
        neighbour is close" is not enough and does not hold the invariant -- a
        cell can sit between one corridor and another, close to the first and far
        from the second, and admitting it creates exactly the pair the rule is
        meant to forbid. Measured: that version produced 343 jump pairs.

        Checking at admission is sufficient, because pairs between two cells that
        are already credited never change afterwards.
        """
        thr = int(JUMP_M / 0.0002)
        ok = vis & in_box
        cred = np.zeros_like(ok)
        cred[centre_r, centre_c] = ok[centre_r, centre_c]

        # Index difference to each neighbour, precomputed once.
        near64 = near.astype(np.int64)
        close_up = np.abs(near64[1:, :] - near64[:-1, :]) <= thr
        close_left = np.abs(near64[:, 1:] - near64[:, :-1]) <= thr

        while True:
            touches = np.zeros_like(cred)
            far = np.zeros_like(cred)
            # neighbour below (row-1) / above (row+1)
            touches[1:, :] |= cred[:-1, :]
            far[1:, :] |= cred[:-1, :] & ~close_up
            touches[:-1, :] |= cred[1:, :]
            far[:-1, :] |= cred[1:, :] & ~close_up
            # neighbour left (col-1) / right (col+1)
            touches[:, 1:] |= cred[:, :-1]
            far[:, 1:] |= cred[:, :-1] & ~close_left
            touches[:, :-1] |= cred[:, 1:]
            far[:, :-1] |= cred[:, 1:] & ~close_left

            new = ok & ~cred & touches & ~far
            # Cells admitted in the same sweep are also neighbours of each other,
            # and that pair is not covered by the check above -- two cells can
            # each be fine against the existing region while being far from one
            # another. Measured: without this prune, 70 jump pairs survived. Drop
            # both sides until the sweep is internally consistent.
            while True:
                clash = np.zeros_like(new)
                clash[1:, :] |= new[:-1, :] & ~close_up
                clash[:-1, :] |= new[1:, :] & ~close_up
                clash[:, 1:] |= new[:, :-1] & ~close_left
                clash[:, :-1] |= new[:, 1:] & ~close_left
                bad = new & clash
                if not bad.any():
                    break
                new &= ~bad
            if not new.any():
                break
            cred |= new

        grid = np.where(cred, near, -1)
        return grid, int(cred.sum())

    print(f"  grid {ny}x{nx} at {1000 * cell:.1f} mm/cell, wall_r {1000 * p.wall_r:.1f} mm")
    if pos is not None:
        print(f"  scoring against {len(pos)} recorded marble positions")
    print()
    score(stored, "stored (as shipped)")
    full = np.where(vis & in_box, near, -1)
    score(full, "visibility only")
    print()
    for tol_mm in (6, 10, 12):
        grid, added = rebuild(tol_mm / 1000.0)
        score(grid, f"restore <= {tol_mm:2d} mm")
    grown, grown_cells = grow()
    score(grown, "grow from centreline")

    if args.write:
        if args.tol_mm is None:
            grid, added = grown, grown_cells
        else:
            grid, added = rebuild(args.tol_mm / 1000.0)
        jumps = jump_pairs(grid)
        if jumps:
            print(
                f"\n  refusing to write: {jumps} jump pairs at "
                f"{args.tol_mm} mm, so shortcuts would be unblocked",
                file=sys.stderr,
            )
            return 1
        p.closest_idx = grid
        suffix = "grown" if args.tol_mm is None else f"grid{int(args.tol_mm)}mm"
        out = args.out or args.path.replace(".pkl", f"_{suffix}.pkl")
        with open(out, "wb") as fh:
            pickle.dump(p, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"\n  restored {added} cells, 0 jump pairs")
        print(f"  wrote {os.path.abspath(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
