#!/usr/bin/env python3
"""Simulated roaming robot over the given ATC 3DSG.

The robot patrols the traversable floor of the voxel graph (authored from the
building's occupancy map), and at each keyframe reports the active window the
live system would maintain: an omnidirectional sphere of radius
`map_window.max_radius_m` (5 m) around its pose. Graph voxels within that
sphere -- traversable floor and wall-straddling support cells alike -- are the
observed/active set (fed to setVisibleVoxels); ATC detections within it are
what the robot sees this frame (fed to push_detections). This recreates the
sparse, incremental, per-cell-irregular sampling regime, rather than a
god's-eye feed. Routing never crosses walls: waypoints and lane joins stay on
the traversable subgraph.

Patrol path: a depth-first tour of a spanning tree of the place graph, so the
robot stays on the walkable floor and covers the whole environment; over a long
session it loops the tour many times, revisiting every area at varied times of
day. The tour is a modelling choice — vary the seed/root for robustness checks.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
try:
    import spark_dsg as dsg
except ImportError:      # host-side tools use only the geometric helpers
    dsg = None

ACTIVE_WINDOW_RADIUS_M = 5.0   # map_window.max_radius_m in the TBD/kairos configs
ROBOT_SPEED_MPS = 1.5
KEYFRAME_DT_S = 0.1            # 10 Hz


def load_nav_graph(graph_path):
    """Load the voxel-node layer as (positions Nx2, node_ids, csr adjacency
    with edge-length weights, traversable mask). Nodes are the 0.4 m TSDF
    voxels (spark_dsg layer id PLACES internally). `traversable` is the
    authored clearance flag (attrs.distance > 0): the floor cells the robot
    may route over; the remaining nodes are wall-straddling observable support
    (visible, carry flow state, never driven through)."""
    G = dsg.DynamicSceneGraph.load(graph_path)
    layer = G.get_layer(dsg.DsgLayers.PLACES)
    nodes = list(layer.nodes)
    # node.id is a NodeSymbol; .value is the uint64 that edge.source/target use.
    node_ids = [n.id.value for n in nodes]
    idx_of = {v: i for i, v in enumerate(node_ids)}
    pos = np.array([[n.attributes.position[0], n.attributes.position[1]]
                    for n in nodes])
    traversable = np.array([n.attributes.distance > 0 for n in nodes], bool)
    rows, cols, w = [], [], []
    for e in layer.edges:
        if e.source in idx_of and e.target in idx_of:
            i, j = idx_of[e.source], idx_of[e.target]
            d = float(np.hypot(*(pos[i] - pos[j])))
            rows += [i, j]; cols += [j, i]; w += [d, d]
    adj = csr_matrix((w, (rows, cols)), shape=(len(node_ids), len(node_ids)))
    return pos, node_ids, adj, traversable


def load_nav_clearance(graph_path):
    """Per-node authored clearance (PlaceNodeAttributes::distance, metres), in
    the same order as load_nav_graph's positions. This is the distance from the
    node to the nearest reconstructed surface, and it bounds how far that
    node's flow-voxel support set may extend."""
    G = dsg.DynamicSceneGraph.load(graph_path)
    layer = G.get_layer(dsg.DsgLayers.PLACES)
    return np.array([float(n.attributes.distance) for n in layer.nodes], float)


def install_nav_layer(pipeline, graph_path):
    """Install the authored navigational layer on a pipeline holding a loaded
    flow state, and run one place pass so evidence sharing has its
    neighbourhoods.

    A pipeline that loads a trained state onto a stub scene builds no scene
    graph of its own, so its navigational layer is empty until this is called:
    the same layer the trainer installed (train_atc.py). The voxel grid the
    place pass needs is restored by load_flow_state, so set_visible_voxels is
    deliberately not called here. Returns (n_nodes, n_edges)."""
    pos, _node_ids, adj, _traversable = load_nav_graph(graph_path)
    clearance = load_nav_clearance(graph_path)
    nav_edges = sorted({(min(i, j), max(i, j))
                        for i, j in zip(*adj.nonzero()) if i != j})
    pipeline.set_navigational_layer(
        [[float(pos[i, 0]), float(pos[i, 1]), 0.0] for i in range(len(pos))],
        [float(c) for c in clearance],
        [[int(i), int(j)] for i, j in nav_edges])
    # The pass logs the published topology ([EvidenceSharing.topology]); that
    # line is the check that sharing has a neighbourhood to draw on.
    pipeline.update_place_summaries(0.0)
    return len(pos), len(nav_edges)


def _dfs_tour(adj_tree, root):
    """Iterative DFS over a tree (csr); returns node order including backtracks,
    so consecutive entries are always tree-adjacent (robot stays on the floor)."""
    n = adj_tree.shape[0]
    children = [[] for _ in range(n)]
    coo = adj_tree.tocoo()
    for i, j in zip(coo.row, coo.col):
        children[i].append(j); children[j].append(i)
    order, visited, stack = [], np.zeros(n, bool), [(root, -1)]
    # explicit stack storing (node, parent, next-child-cursor)
    frames = [[root, -1, 0]]
    visited[root] = True
    order.append(root)
    while frames:
        node, parent, k = frames[-1]
        nxt = None
        while k < len(children[node]):
            c = children[node][k]; k += 1
            if not visited[c]:
                nxt = c; break
        frames[-1][2] = k
        if nxt is None:
            frames.pop()
            if frames:
                order.append(frames[-1][0])   # backtrack step
        else:
            visited[nxt] = True
            order.append(nxt)
            frames.append([nxt, node, 0])
    return order


def build_patrol(pos, adj, root=0):
    """Build a walkable patrol polyline: DFS tour of the graph's spanning tree.
    Works per connected component (largest first), stitching components by a
    straight hop (rare on a connected concourse)."""
    mst = minimum_spanning_tree(adj)           # forest if disconnected
    mst = (mst + mst.T)
    order = _dfs_tour(mst, root)
    poly = pos[order]
    return poly, order


def build_coverage_loop(pos, adj, traversable=None,
                        radius=ACTIVE_WINDOW_RADIUS_M, lane_width=8.0):
    """A closed serpentine (boustrophedon) coverage loop over the traversable
    floor.

    A patrol/cleaning robot that snakes across the whole environment once per loop:
    coarse waypoints spaced ~`lane_width` in a grid over the traversable cells (so
    the moving `radius`-metre sphere, a 2*radius-wide swath, tiles the area),
    ordered in a boustrophedon sweep along the environment's long axis (run the
    length, step one lane across, run back), and consecutive waypoints joined by
    shortest paths ON the traversable subgraph so the robot never crosses walls or
    non-walkable space. `traversable` is the mask from load_nav_graph; wall-
    straddling support nodes are observable but never routed over (None = all
    nodes traversable). The loop is closed (returns to the first waypoint) and is
    repeated identically every session, from the same start pose. Returns
    (polyline Mx2, order in global node indices)."""
    from scipy.sparse.csgraph import dijkstra
    if traversable is None:
        traversable = np.ones(len(pos), bool)
    g_idx = np.flatnonzero(traversable)       # traversable -> global index map
    tpos = pos[g_idx]
    tadj = adj[g_idx][:, g_idx]
    n = len(tpos)
    c = tpos.mean(axis=0)
    X = tpos - c
    _, V = np.linalg.eigh(X.T @ X)
    u, v = V[:, -1], V[:, -2]                 # long axis, short axis
    uu, vv = X @ u, X @ v
    lane = np.floor((vv - vv.min()) / lane_width).astype(int)
    ubin = np.floor((uu - uu.min()) / lane_width).astype(int)
    # one representative traversable cell per (lane, along-axis) grid bin, nearest
    # the bin centre so waypoints sit mid-corridor.
    reps = {}
    for i in range(n):
        key = (int(lane[i]), int(ubin[i]))
        cu = (ubin[i] + 0.5) * lane_width + uu.min()
        cv = (lane[i] + 0.5) * lane_width + vv.min()
        d2 = (uu[i] - cu) ** 2 + (vv[i] - cv) ** 2
        if key not in reps or d2 < reps[key][1]:
            reps[key] = (i, d2)
    # boustrophedon order of bins: lanes low->high; within a lane alternate the
    # along-axis direction so consecutive lanes join end-to-end (the snake).
    keys = sorted(reps, key=lambda k: (k[0], k[1] if k[0] % 2 == 0 else -k[1]))
    wp = [reps[k][0] for k in keys]
    if len(wp) < 2:
        return tpos[wp], g_idx[np.array(wp)]
    # join consecutive waypoints (and close the loop) via graph shortest paths.
    order = []
    seq = wp + [wp[0]]
    for a, b in zip(seq[:-1], seq[1:]):
        _, pred = dijkstra(tadj, indices=a, return_predecessors=True)
        path, j = [], b
        while j != a and j >= 0:
            path.append(j); j = int(pred[j])
        path.append(a); path.reverse()
        if j < 0:                             # disconnected (shouldn't happen: 1 component)
            path = [a, b]
        order.extend(path[1:] if order and order[-1] == path[0] else path)
    order = np.array(order)
    return tpos[order], g_idx[order]


class RobotSim:
    def __init__(self, poly, place_pos, radius=ACTIVE_WINDOW_RADIUS_M,
                 speed=ROBOT_SPEED_MPS):
        self.poly = poly
        seg = np.diff(poly, axis=0)
        self.seglen = np.hypot(seg[:, 0], seg[:, 1])
        self.cum = np.concatenate([[0.0], np.cumsum(self.seglen)])
        self.total = float(self.cum[-1])
        self.radius = radius
        self.speed = speed
        self.place_pos = place_pos
        self.place_tree = cKDTree(place_pos)

    def pose_at_arclength(self, s):
        s = s % self.total
        k = np.searchsorted(self.cum, s, "right") - 1
        k = min(max(k, 0), len(self.seglen) - 1)
        if self.seglen[k] <= 0:
            return self.poly[k]
        frac = (s - self.cum[k]) / self.seglen[k]
        return self.poly[k] + frac * (self.poly[k + 1] - self.poly[k])

    def visible_places(self, pose):
        """Indices of walkable places within the active-window sphere of pose."""
        return self.place_tree.query_ball_point(pose, self.radius)
