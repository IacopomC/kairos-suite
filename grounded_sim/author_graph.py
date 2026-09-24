#!/usr/bin/env python3
r"""Author the ATC TSDF-voxel graph (+ occupancy layer) from the given map.

The environment geometry is given, not reconstructed: walls come from the ATC
localization occupancy grid (the building blueprint), and the detection corpus
delineates only the *instrumented region* the pedestrian trackers cover. Emits
one graph node per supported cell at the voxel size, 8-adjacent -- the TSDF
voxels where the spectral-dynamics (flow) model lives (created incrementally by
the TSDF in the live system). Stored in the spark_dsg PLACES layer only as a
sim-storage convenience (the layer the loaders read); they are TSDF voxels, not
navigational places.

Two nested masks, mirroring the live system where the flow model lives on
observed voxels (including wall surfaces) while the robot can only drive on
free floor:

- traversable: instrumented-coverage cells with no wall blockage, largest
  connected component. The robot patrol routes on these (attrs.distance > 0,
  the clearance from the traversable distance transform).
- support: traversable plus coverage cells adjacent to the traversable
  component that the robot cannot drive over (wall-straddling cells and
  routing-disconnected pockets; people are detected brushing storefronts and
  pillars, so those cells are observable and carry flow state,
  attrs.distance == 0).

Edges: 8-adjacency among traversable cells, with diagonals dropped unless both
shared orthogonal neighbours are traversable (no corner-cutting through a
wall); support-only cells link to adjacent traversable cells (observation/
aggregation edges), never to each other (an edge between two wall-straddling
cells could cross the wall between them).

The wall grid is binary at 5 cm (mid-grey navigable vs black walls); each
0.4 m cell pools the 8x8 wall pixels under it: >= --wall-block-px blocks
traversal, >= --wall-solid-px excludes the cell from support entirely.

The coverage region uses the TRAINING days only (pass the same day set the
flow model trains on; never the held-out test days). A provenance sidecar
(<out>.provenance.json) records the exact inputs and parameters.

Runs in a container with spark_dsg + numpy/scipy (kairos-suite).

Usage:
    python3 author_graph.py --days $DATA_ROOT/atc/npz/atc-2012*.npz \
        --map-pgm  $DATA_ROOT/atc/ATC-map/localization_grid.pgm \
        --map-yaml $DATA_ROOT/atc/ATC-map/localization_grid.yaml \
        --out $DATA_ROOT/atc/graph/atc_graph.json \
        --out-occupancy $DATA_ROOT/atc/graph/atc_occupancy.npz \
        --coverage-cache $DATA_ROOT/atc/graph/coverage_cache.npz

Run once, and pass only TRAIN days: the coverage region becomes the robot's
mapped world, so including test days would leak the test set into the map. The
`.provenance.json` sidecar records which days and parameters were used.
"""
import argparse
import json
import pathlib
import numpy as np
from scipy import ndimage
import spark_dsg as dsg
from spark_dsg._dsg_bindings import NodeSymbol

from atc_stream import load_occupancy

PGM_WALL_THRESH = 60          # grid is binary: >=60 navigable, <60 wall
PGM_PX_PER_CELL = 8           # 0.05 m px -> 0.4 m cell


def load_map(yaml_path, pgm_path):
    """Map extent (m) and the per-0.4m-cell wall-pixel counts, y up: (nx, ny)."""
    from PIL import Image
    res = origin = None
    with open(yaml_path) as f:
        for line in f:
            if line.startswith("resolution:"):
                res = float(line.split(":")[1])
            if line.startswith("origin:"):
                origin = [float(v) for v in line.split("[")[1].split("]")[0].split(",")]
    img = np.array(Image.open(pgm_path))
    if img.ndim == 3:
        img = img[..., 0]
    H, W = img.shape
    ext = (origin[0], origin[0] + W * res, origin[1], origin[1] + H * res)
    wall_px = img < PGM_WALL_THRESH
    k = PGM_PX_PER_CELL
    ph, pw = (-H) % k, (-W) % k
    wp = np.pad(wall_px, ((0, ph), (0, pw)))
    pooled = wp.reshape((H + ph) // k, k, (W + pw) // k, k).sum(axis=(1, 3))
    wall_counts = pooled[::-1, :].T          # image row 0 is max y -> flip to y-up
    return ext, res, wall_counts


def coverage_counts(days, ext, vs, cache=None):
    """Per-cell detection counts unioned over the given training days."""
    if cache and pathlib.Path(cache).exists():
        z = np.load(cache)
        if list(z["days"]) == [pathlib.Path(d).stem for d in days] and \
                float(z["voxel_size"]) == vs:
            print(f"[author] coverage counts from cache {cache}")
            return z["counts"]
        print(f"[author] cache {cache} does not match inputs, recomputing")
    x0, x1, y0, y1 = ext
    nx = int(np.ceil((x1 - x0) / vs))
    ny = int(np.ceil((y1 - y0) / vs))
    counts = np.zeros((nx, ny), dtype=np.int64)
    for day in days:
        z = np.load(day)
        xm, ym = z["x"] / 1000.0, z["y"] / 1000.0     # mm -> m
        m = (xm >= x0) & (xm < x1) & (ym >= y0) & (ym < y1)
        xm, ym = xm[m], ym[m]
        ix = ((xm - x0) / vs).astype(int)
        iy = ((ym - y0) / vs).astype(int)
        np.add.at(counts, (ix, iy), 1)
        print(f"[author]   {pathlib.Path(day).stem}: {len(xm):,} in-bounds detections")
    if cache:
        np.savez_compressed(cache, counts=counts, voxel_size=vs,
                            days=[pathlib.Path(d).stem for d in days])
        print(f"[author] coverage counts cached -> {cache}")
    return counts


def edge_connected_floor(trav):
    """Largest component of the traversable mask under the ROUTING connectivity:
    orthogonal adjacency always, diagonal only when both shared orthogonal
    neighbours are traversable (the same no-corner-cutting rule the graph edges
    use). A plain 8-connected mask can be held together by corner-cut diagonals
    the edge rule forbids, leaving waypoint fragments Dijkstra cannot reach."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    nx, ny = trav.shape
    idx = -np.ones(trav.shape, dtype=np.int64)
    fi, fj = np.where(trav)
    idx[fi, fj] = np.arange(len(fi))
    rows, cols = [], []

    def link(di, dj, corner_rule):
        i, j = np.where(trav)
        ni, nj = i + di, j + dj
        valid = (ni >= 0) & (ni < nx) & (nj >= 0) & (nj < ny)
        i, j, ni, nj = i[valid], j[valid], ni[valid], nj[valid]
        ok = trav[ni, nj]
        if corner_rule:
            ok &= trav[ni, j] & trav[i, nj]
        rows.extend(idx[i[ok], j[ok]].tolist())
        cols.extend(idx[ni[ok], nj[ok]].tolist())

    for di, dj in [(0, 1), (1, 0)]:
        link(di, dj, corner_rule=False)
    for di, dj in [(1, 1), (1, -1)]:
        link(di, dj, corner_rule=True)
    n = len(fi)
    g = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    ncc, lab = connected_components(g, directed=False)
    if ncc <= 1:
        return trav, 0
    sizes = np.bincount(lab)
    keep = lab == int(np.argmax(sizes))
    out = np.zeros_like(trav)
    out[fi[keep], fj[keep]] = True
    return out, int((~keep).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", nargs="+", required=True,
                    help="TRAINING day npz files; their detections delineate the "
                         "instrumented-coverage region (never pass test days)")
    ap.add_argument("--map-pgm", required=True)
    ap.add_argument("--map-yaml", required=True)
    ap.add_argument("--out", required=True, help="output TSDF-voxel graph.json")
    ap.add_argument("--voxel-size", type=float, default=0.4,
                    help="TSDF voxel size (m): one voxel-graph node per supported "
                         "cell at this resolution")
    ap.add_argument("--min-coverage-visits", type=int, default=1,
                    help="min detections for a cell to count toward the "
                         "instrumented-coverage region")
    ap.add_argument("--coverage-closing-cells", type=int, default=5,
                    help="binary-closing structure width (cells) that smooths the "
                         "coverage region (5 cells = 2 m at 0.4 m)")
    ap.add_argument("--wall-block-px", type=int, default=8,
                    help="wall pixels (of 64) in a cell that block traversal")
    ap.add_argument("--wall-solid-px", type=int, default=33,
                    help="wall pixels (of 64) that exclude a cell from support "
                         "(majority-solid wall)")
    ap.add_argument("--room-block", type=float, default=12.0,
                    help="side of the square block that defines one room (m)")
    ap.add_argument("--single-room", action="store_true", default=True,
                    help="one concourse room over all voxels (default). ATC is an "
                         "open concourse with no natural rooms; the meaningful "
                         "mid-level layer is the flow-archetype layer, added later.")
    ap.add_argument("--block-rooms", dest="single_room", action="store_false",
                    help="instead partition into --room-block spatial bins (not "
                         "semantic; visualization eye-candy only)")
    ap.add_argument("--out-occupancy", default=None,
                    help="also emit the occupancy layer here (.npz: traversable "
                         "cells + walls) for the z0 environment-map render")
    ap.add_argument("--coverage-cache", default=None,
                    help="npz cache for the per-cell training-day detection "
                         "counts (reused only if days and voxel size match)")
    args = ap.parse_args()

    ext, _, wall_counts = load_map(args.map_yaml, args.map_pgm)
    x0, x1, y0, y1 = ext
    vs = args.voxel_size
    nx = int(np.ceil((x1 - x0) / vs))
    ny = int(np.ceil((y1 - y0) / vs))
    wall_counts = wall_counts[:nx, :ny]
    wall_block = wall_counts >= args.wall_block_px
    wall_solid = wall_counts >= args.wall_solid_px
    print(f"[author] map extent x=({x0},{x1}) y=({y0},{y1}) m | "
          f"wall cells: block={int(wall_block.sum())} solid={int(wall_solid.sum())}")

    counts = coverage_counts(args.days, ext, vs, cache=args.coverage_cache)
    k = args.coverage_closing_cells
    coverage = ndimage.binary_closing(counts >= args.min_coverage_visits,
                                      structure=np.ones((k, k)))
    print(f"[author] instrumented coverage: {int(coverage.sum()):,} cells "
          f"({len(args.days)} training days)")

    # traversable floor: covered, wall-free, one component under the routing
    # connectivity (no corner-cutting), so every waypoint is Dijkstra-reachable
    trav, dropped = edge_connected_floor(coverage & ~wall_block)
    if dropped:
        print(f"[author] traversable: dropped {dropped} cells not "
              f"routing-connected to the main floor (kept {int(trav.sum())})")
    # support: traversable + wall-straddling covered cells touching it
    ring = coverage & ~wall_solid & ~trav & \
        ndimage.binary_dilation(trav, structure=np.ones((3, 3)))
    support = trav | ring
    dist_field = ndimage.distance_transform_edt(trav) * vs
    cov_dets = int(counts[support].sum())
    all_dets = int(counts.sum())
    print(f"[author] traversable {int(trav.sum()):,} + observable ring "
          f"{int(ring.sum()):,} = support {int(support.sum()):,} cells; "
          f"support holds {100 * cov_dets / max(all_dets, 1):.2f}% of "
          f"in-bounds training detections")

    # --- TSDF voxels: one node per supported cell at the voxel size. These are
    #     the cells the flow model lives on (created incrementally by the TSDF in
    #     the live system). attrs.distance is the traversable clearance: > 0 on
    #     the floor the robot may route over, == 0 on wall-straddling support.
    vi, vj = np.where(support)
    voxels = {}          # (i,j) -> dict(idx, x, y, dist, trav)
    for i, j in zip(vi.tolist(), vj.tolist()):
        voxels[(i, j)] = dict(
            idx=len(voxels),
            x=x0 + (i + 0.5) * vs, y=y0 + (j + 0.5) * vs,
            trav=bool(trav[i, j]),
            dist=max(float(dist_field[i, j]), vs) if trav[i, j] else 0.0)
    print(f"[author] TSDF voxels: {len(voxels)}")

    G = dsg.DynamicSceneGraph()
    for (i, j), vx in voxels.items():
        a = dsg.PlaceNodeAttributes()
        a.position = np.array([vx["x"], vx["y"], 0.0], dtype=np.float64)
        a.distance = vx["dist"]
        a.num_basis_points = 0
        G.add_node(dsg.DsgLayers.PLACES, NodeSymbol("p", vx["idx"]).value, a)

    # --- voxel-voxel edges (8-adjacency), weight = euclidean distance.
    #     Diagonals between traversable cells require both shared orthogonal
    #     neighbours traversable (no corner-cutting through a wall); support-only
    #     cells connect to traversable neighbours but never to each other.
    nbrs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    n_vedges = 0
    for (i, j), vx in voxels.items():
        for di, dj in nbrs:
            nb = voxels.get((i + di, j + dj))
            if nb is None or nb["idx"] <= vx["idx"]:
                continue
            if not (vx["trav"] or nb["trav"]):
                continue
            diagonal = di != 0 and dj != 0
            if diagonal and vx["trav"] and nb["trav"]:
                if not (trav[i + di, j] and trav[i, j + dj]):
                    continue
            e = dsg.EdgeAttributes()
            e.weight = float(np.hypot(vx["x"] - nb["x"], vx["y"] - nb["y"]))
            e.weighted = True
            G.insert_edge(NodeSymbol("p", vx["idx"]).value,
                          NodeSymbol("p", nb["idx"]).value, e)
            n_vedges += 1

    # --- rooms ---
    if args.single_room:
        # One concourse room parenting all voxels. ATC has no natural rooms; the
        # meaningful mid-level layer is the flow-archetype layer, added later.
        cx = float(np.mean([v["x"] for v in voxels.values()]))
        cy = float(np.mean([v["y"] for v in voxels.values()]))
        a = dsg.RoomNodeAttributes()
        a.position = np.array([cx, cy, 0.0], dtype=np.float64)
        a.name = "R(0) concourse"
        G.add_node(dsg.DsgLayers.ROOMS, NodeSymbol("R", 0).value, a)
        for vx in voxels.values():
            G.insert_edge(NodeSymbol("R", 0).value,
                          NodeSymbol("p", vx["idx"]).value, dsg.EdgeAttributes())
        n_rooms = 1
    else:
        # Spatial bins (not semantic; visualization eye-candy only).
        def block_of(vx):
            return (int((vx["x"] - x0) // args.room_block),
                    int((vx["y"] - y0) // args.room_block))

        blocks = {}
        for vx in voxels.values():
            blocks.setdefault(block_of(vx), []).append(vx)
        room_id = {b: i for i, b in enumerate(sorted(blocks))}
        for b, members in blocks.items():
            rid = room_id[b]
            cx = float(np.mean([v["x"] for v in members]))
            cy = float(np.mean([v["y"] for v in members]))
            a = dsg.RoomNodeAttributes()
            a.position = np.array([cx, cy, 0.0], dtype=np.float64)
            a.name = f"R({rid})"
            G.add_node(dsg.DsgLayers.ROOMS, NodeSymbol("R", rid).value, a)
            for vx in members:
                G.insert_edge(NodeSymbol("R", rid).value,
                              NodeSymbol("p", vx["idx"]).value, dsg.EdgeAttributes())
        room_pairs = set()
        for (i, j), vx in voxels.items():
            ra = room_id[block_of(vx)]
            for di, dj in nbrs:
                nb = voxels.get((i + di, j + dj))
                if nb is None:
                    continue
                rb = room_id[block_of(nb)]
                if ra != rb:
                    room_pairs.add(tuple(sorted((ra, rb))))
        for a, b in room_pairs:
            G.insert_edge(NodeSymbol("R", a).value, NodeSymbol("R", b).value,
                          dsg.EdgeAttributes())
        n_rooms = len(blocks)

    G.save(args.out)
    print(f"[author] saved TSDF-voxel graph {args.out}: {G.num_nodes()} nodes, "
          f"{G.num_edges()} edges | {len(voxels)} voxels, {n_vedges} voxel-edges")

    prov = dict(
        days=[pathlib.Path(d).stem for d in args.days],
        map_pgm=str(args.map_pgm), map_yaml=str(args.map_yaml),
        voxel_size=vs, min_coverage_visits=args.min_coverage_visits,
        coverage_closing_cells=args.coverage_closing_cells,
        wall_block_px=args.wall_block_px, wall_solid_px=args.wall_solid_px,
        cells=dict(coverage=int(coverage.sum()), traversable=int(trav.sum()),
                   ring=int(ring.sum()), support=int(support.sum()),
                   wall_block=int(wall_block.sum())),
        support_detection_fraction=cov_dets / max(all_dets, 1),
        edges=n_vedges, rooms=n_rooms)
    prov_path = pathlib.Path(args.out).with_suffix(".provenance.json")
    prov_path.write_text(json.dumps(prov, indent=2))
    print(f"[author] provenance -> {prov_path}")

    if args.out_occupancy:
        fi, fj = np.where(trav)
        free_xy = np.column_stack([x0 + (fi + 0.5) * vs, y0 + (fj + 0.5) * vs])
        # walls near the concourse only (within the traversable bounding box)
        wi, wj = np.where(wall_block)
        bb = (fi.min(), fi.max(), fj.min(), fj.max())
        inbb = (wi >= bb[0]) & (wi <= bb[1]) & (wj >= bb[2]) & (wj <= bb[3])
        wall_xy = np.column_stack([x0 + (wi[inbb] + 0.5) * vs, y0 + (wj[inbb] + 0.5) * vs])
        np.savez_compressed(args.out_occupancy, free_xy=free_xy, wall_xy=wall_xy,
                            voxel_size=vs)
        print(f"[author] saved occupancy {args.out_occupancy}: {len(free_xy)} free, "
              f"{len(wall_xy)} wall cells")


if __name__ == "__main__":
    main()
