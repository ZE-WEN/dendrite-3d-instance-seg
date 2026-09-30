import numpy as np


class UnionFind:
    """Keeps track of which track ids are really the same track. The smaller id wins."""

    def __init__(self):
        self.parent = {}

    def find(self, x):
        root = x
        while self.parent.setdefault(root, root) != root:
            root = self.parent[root]
        # point everything on the path straight at the root
        while x != root:
            nxt = self.parent[x]
            self.parent[x] = root
            x = nxt
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        keep, drop = (ra, rb) if ra < rb else (rb, ra)
        self.parent[drop] = keep
        return keep


def greedy_linking(df_edges, comps, counts, p_thresh=0.5):
    """
    Turn scored edges into a volume of track ids.

    Every edge with p_link >= p_thresh is kept. There is no one-to-one rule, so a
    component can join several others (branches are fine). Edges between
    neighbouring slices (dz=1) are handled before skip edges (dz=2), highest
    probability first. Components that never get linked get their own id.
    """
    Z = len(comps)
    uf = UnionFind()

    # track_of[z][c] is the track id of component c on slice z (0 = not assigned yet)
    track_of = [np.zeros(k + 1, dtype=np.uint32) for k in counts]
    next_id = 1

    edges = df_edges
    if not edges.empty:
        edges = edges[edges["p_link"] >= p_thresh]

    if not edges.empty:
        for (dz, z), group in edges.groupby(["dz", "z"]):
            group = group.sort_values("p_link", ascending=False)

            for c0, c1 in zip(group["c_src"].astype(int), group["c_tgt"].astype(int)):
                t0 = track_of[z][c0]
                t1 = track_of[z + dz][c1]

                if t0 and t1:
                    keep = uf.union(int(t0), int(t1))
                    track_of[z][c0] = keep
                    track_of[z + dz][c1] = keep
                elif t0:
                    track_of[z + dz][c1] = uf.find(int(t0))
                elif t1:
                    track_of[z][c0] = uf.find(int(t1))
                else:
                    track_of[z][c0] = next_id
                    track_of[z + dz][c1] = next_id
                    uf.find(next_id)
                    next_id += 1

    # anything left over is its own track
    for z in range(Z):
        for c in range(1, counts[z] + 1):
            if track_of[z][c] == 0:
                track_of[z][c] = next_id
                uf.find(next_id)
                next_id += 1

    # resolve merged ids and paint the volume
    out = np.zeros((Z,) + comps[0].shape, dtype=np.uint32)
    for z in range(Z):
        ids = track_of[z]
        for i in np.flatnonzero(ids):
            ids[i] = uf.find(int(ids[i]))
        out[z] = ids[comps[z]]

    return out
