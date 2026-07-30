#!/usr/bin/env python3
"""Show which cells of the path grid are -1, and which of those are over-painted.

closest_idx[y, x] is the path index a marble in that cell is credited with, or
-1 for no credit -- which the original CyberRunner env treated as off-path and
ended the episode on. Two separate build-time passes write -1:

  * the visibility test, when the box the surrounding walls leave around a cell
    contains no path point at all (wrong side of a wall, or a dead-end pocket)
  * jump-detection, which marks both cells of any adjacent pair whose credited
    index differs by more than 0.057 m of path -- the ridge between two
    different corridors, i.e. exactly the route a shortcut hop would take

Only the first is reconstructed here. Anything the stored grid calls -1 that
the reconstruction calls valid was therefore removed by jump-detection, and
this is what separates the two so the over-painting is visible.

    python3 tools/plot_path_grid.py [--out FILE] [--path PKL]
"""
import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tag_dreamer")
)


def wall_bounds(coords_along, spans, ball_r, n_cells, cell):
    """Nearest wall on each side of every cell, per row or per column.

    spans holds (start, end, coord) as LinearPath stores walls. Returns two
    arrays shaped (n_perp, n_cells): the first wall coordinate at or above each
    cell, and the last one strictly below.
    """
    n_perp = len(coords_along)
    above = np.full((n_perp, n_cells), np.inf)
    below = np.full((n_perp, n_cells), -np.inf)
    positions = np.arange(n_cells) * cell
    for i, along in enumerate(coords_along):
        # LinearPath inflates the span by the ball radius, not the coordinate.
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


def reconstruct_visibility(p, layout):
    """The grid the visibility test alone would produce, without jump-detection."""
    cell = p.distance
    ny, nx = p.closest_idx.shape
    walls_h = np.asarray(layout["walls_h"], dtype=float)
    walls_v = np.asarray(layout["walls_v"], dtype=float)
    r = p.wall_r

    # Horizontal walls bound y, and which of them apply depends on the column.
    y_above, y_below = wall_bounds(np.arange(nx) * cell, walls_h, r, ny, cell)
    # Vertical walls bound x, and which apply depends on the row.
    x_above, x_below = wall_bounds(np.arange(ny) * cell, walls_v, r, nx, cell)

    # y_above is (nx, ny) -> transpose to (ny, nx) to match the grid.
    y2 = y_above.T
    y1 = y_below.T

    # Occupancy of path points per cell, then an integral image so "does the box
    # hold a path point" is a constant-time query instead of a scan.
    occ = np.zeros((ny, nx), dtype=np.int32)
    pr = np.clip((p.points[:, 1] / cell).astype(int), 0, ny - 1)
    pc = np.clip((p.points[:, 0] / cell).astype(int), 0, nx - 1)
    occ[pr, pc] = 1
    integral = np.zeros((ny + 1, nx + 1), dtype=np.int32)
    integral[1:, 1:] = occ.cumsum(0).cumsum(1)

    def to_cell(bound, limit, ceil):
        out = np.where(
            np.isfinite(bound),
            np.ceil(bound / cell) if ceil else np.floor(bound / cell),
            0 if ceil else limit - 1,
        )
        return np.clip(out, 0, limit - 1).astype(int)

    r0 = to_cell(y1, ny, ceil=True)
    r1 = to_cell(y2, ny, ceil=False)
    c0 = to_cell(x_below, nx, ceil=True)
    c1 = to_cell(x_above, nx, ceil=False)

    count = (
        integral[r1 + 1, c1 + 1]
        - integral[r0, c1 + 1]
        - integral[r1 + 1, c0]
        + integral[r0, c0]
    )
    return (count > 0) & (r1 >= r0) & (c1 >= c0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument(
        "--path",
        default=os.path.join(here, "..", "tag_dreamer", "data", "path_custom.pkl"),
    )
    ap.add_argument("--out", default=os.path.join(here, "..", "docs", "path_grid.png"))
    ap.add_argument(
        "--zoom",
        default="",
        help="x0,x1,y0,y1 in mm, to inspect one region instead of the whole board",
    )
    ap.add_argument(
        "--simple", action="store_true",
        help="one panel, credited against not credited. The three-panel view "
             "compares the grid to the visibility test, which is only meaningful "
             "when the grid was built from it.",
    )
    args = ap.parse_args()
    zoom = [float(v) for v in args.zoom.split(",")] if args.zoom else None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    with open(args.path, "rb") as fh:
        p = pickle.load(fh)

    import importlib
    mod = importlib.import_module("tag_dreamer.tag_layout_custom")
    layout = next(
        v for v in vars(mod).values() if isinstance(v, dict) and "walls_h" in v
    )

    grid = p.closest_idx
    stored_ok = grid != -1

    if args.simple:
        ny_, nx_ = grid.shape
        ext = [0, 1000 * nx_ * p.distance, 0, 1000 * ny_ * p.distance]
        fig, ax = plt.subplots(figsize=(11, 9.6))
        ax.imshow(
            np.where(stored_ok, 1.0, 0.0), extent=ext, origin="lower",
            cmap=ListedColormap(["#d1382c", "#1f9d55"]), interpolation="nearest",
            vmin=0, vmax=1,
        )
        for x0, x1_, y in np.asarray(layout["walls_h"]):
            ax.plot([1000 * x0, 1000 * x1_], [1000 * y, 1000 * y], "k", lw=1.3)
        for y0, y1_, x in np.asarray(layout["walls_v"]):
            ax.plot([1000 * x, 1000 * x], [1000 * y0, 1000 * y1_], "k", lw=1.3)
        ax.plot(1000 * p.points[:, 0], 1000 * p.points[:, 1], color="#ffd400", lw=1.0)
        ax.legend(handles=[
            Patch(color="#1f9d55", label=f"walkable, scored  {100 * stored_ok.mean():.1f}%"),
            Patch(color="#d1382c",
                  label=f"off-path: episode ends  {100 * (~stored_ok).mean():.1f}%"),
            Patch(color="#ffd400", label="path centreline"),
            Patch(color="black", label="walls"),
        ], loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2, fontsize=9)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_title(os.path.basename(os.path.abspath(args.path)))
        if zoom:
            ax.set_xlim(zoom[0], zoom[1])
            ax.set_ylim(zoom[2], zoom[3])
        fig.tight_layout()
        out = os.path.abspath(args.out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.savefig(out, dpi=120, bbox_inches="tight")
        print(f"wrote {out}")
        print(f"  walkable {100 * stored_ok.mean():.1f}%   "
              f"off-path {100 * (~stored_ok).mean():.1f}%")
        return 0

    visible = reconstruct_visibility(p, layout)

    # Sanity: the stored grid must be a subset of what visibility allows. Any
    # cell the stored grid credits but the reconstruction rejects means the
    # reconstruction is wrong, not the file.
    contradiction = stored_ok & ~visible
    over_painted = ~stored_ok & visible          # removed by jump-detection
    genuine = ~stored_ok & ~visible              # blocked by a wall

    ny, nx = grid.shape
    extent = [0, 1000 * nx * p.distance, 0, 1000 * ny * p.distance]

    # class map: 0 valid, 1 over-painted, 2 genuinely blocked
    klass = np.where(stored_ok, 0, np.where(over_painted, 1, 2))
    cmap = ListedColormap(["#1f9d55", "#ff9f1a", "#5b6270"])

    fig, axes = plt.subplots(1, 3, figsize=(23, 7.4))

    def walls(ax, lw=1.0):
        for x0, x1, y in np.asarray(layout["walls_h"]):
            ax.plot([1000 * x0, 1000 * x1], [1000 * y, 1000 * y], "k", lw=lw)
        for y0, y1, x in np.asarray(layout["walls_v"]):
            ax.plot([1000 * x, 1000 * x], [1000 * y0, 1000 * y1], "k", lw=lw)

    ax = axes[0]
    ax.imshow(
        np.where(stored_ok, np.nan, 1.0), extent=extent, origin="lower",
        cmap=ListedColormap(["#ff3b30"]), interpolation="nearest",
    )
    ax.imshow(
        np.where(stored_ok, grid, np.nan).astype(float), extent=extent,
        origin="lower", cmap="viridis", interpolation="nearest",
    )
    walls(ax)
    ax.set_title(
        "stored grid: red = -1\n"
        f"{100 * (~stored_ok).mean():.1f}% of cells give no credit"
    )

    ax = axes[1]
    ax.imshow(
        np.where(visible, np.nan, 1.0), extent=extent, origin="lower",
        cmap=ListedColormap(["#ff3b30"]), interpolation="nearest",
    )
    ax.imshow(
        np.where(visible, 1.0, np.nan), extent=extent, origin="lower",
        cmap=ListedColormap(["#1f9d55"]), interpolation="nearest",
    )
    walls(ax)
    ax.set_title(
        "visibility test alone (no jump-detection)\n"
        f"{100 * (~visible).mean():.1f}% of cells give no credit"
    )

    ax = axes[2]
    ax.imshow(klass, extent=extent, origin="lower", cmap=cmap,
              interpolation="nearest", vmin=0, vmax=2)
    walls(ax)
    ax.plot(1000 * p.points[:, 0], 1000 * p.points[:, 1], "w-", lw=0.7, alpha=0.9)
    ax.legend(handles=[
        Patch(color="#1f9d55", label=f"credited  {100 * stored_ok.mean():.1f}%"),
        Patch(color="#ff9f1a",
              label=f"-1 only from jump-detection  {100 * over_painted.mean():.1f}%"),
        Patch(color="#5b6270",
              label=f"-1 from walls  {100 * genuine.mean():.1f}%"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.07), fontsize=9)
    ax.set_title("what jump-detection removed (orange)")

    for ax in axes:
        ax.set_xlabel("x (mm)")
        if zoom:
            ax.set_xlim(zoom[0], zoom[1])
            ax.set_ylim(zoom[2], zoom[3])
    axes[0].set_ylabel("y (mm)")

    if zoom:
        cell = p.distance
        c0, c1 = int(zoom[0] / 1000 / cell), int(zoom[1] / 1000 / cell)
        r0, r1 = int(zoom[2] / 1000 / cell), int(zoom[3] / 1000 / cell)
        sub_ok = stored_ok[r0:r1, c0:c1]
        sub_over = over_painted[r0:r1, c0:c1]
        sub_gen = genuine[r0:r1, c0:c1]
        print(
            f"  zoom {args.zoom}: credited {100 * sub_ok.mean():.1f}%  "
            f"jump-detection {100 * sub_over.mean():.1f}%  "
            f"walls {100 * sub_gen.mean():.1f}%"
        )

    fig.tight_layout()
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=118, bbox_inches="tight")

    print(f"wrote {out}")
    print(f"  credited by stored grid        : {100 * stored_ok.mean():5.1f}%")
    print(f"  credited by visibility alone   : {100 * visible.mean():5.1f}%")
    print(f"  -1 only from jump-detection    : {100 * over_painted.mean():5.1f}%")
    print(f"  -1 from walls                  : {100 * genuine.mean():5.1f}%")
    print(f"  reconstruction contradictions  : {contradiction.sum()} "
          f"(must be 0; stored must be a subset)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
