from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from tifffile import imread, imwrite

# change these to your own data
GT_PATH = "/path/to/ground_truth_instances.tif"
COMPONENTS_OUT = "/path/to/output/components_global.tif"
EDGES_OUT = "/path/to/output/train_edges.csv"

MIN_OVERLAP_PX = 8      # ignore pairs that overlap less than this
CONNECTIVITY = 1        # 1 = 4-connected, 2 = 8-connected
USE_SKIP_LINKS = False  # also build edges between slice z and z+2


def overlap_pairs(C0, C1):
    """all (label in C0, label in C1, overlap in pixels) between two slices"""
    both = (C0 > 0) & (C1 > 0)
    if not both.any():
        return [], [], []
    a = C0[both].astype(np.int64)
    b = C1[both].astype(np.int64)
    base = b.max() + 1
    keys, counts = np.unique(a * base + b, return_counts=True)
    keep = counts >= MIN_OVERLAP_PX
    return keys[keep] // base, keys[keep] % base, counts[keep]


G = imread(GT_PATH)
if G.dtype.kind == "f":
    G = G.astype(np.int64)
assert G.ndim == 3, "expected a (Z, Y, X) volume"
Z = G.shape[0]

structure = np.ones((3, 3)) if CONNECTIVITY == 2 else None

# split every instance into its 2D connected components, slice by slice
comps = []      # comps[z] is a slice where each component has its own id (1..n)
n_comps = []    # number of components per slice
parents = []    # parents[z][i] is the instance id that component i+1 came from

for z in range(Z):
    Gz = G[z]
    Cz = np.zeros(Gz.shape, dtype=np.int32)
    parent_of = []
    n = 0

    for gid in np.unique(Gz):
        if gid == 0:
            continue
        ys, xs = np.nonzero(Gz == gid)
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1

        labs, k = ndi.label(Gz[y0:y1, x0:x1] == gid, structure=structure)
        window = Cz[y0:y1, x0:x1]
        window[labs > 0] = labs[labs > 0] + n

        parent_of += [int(gid)] * k
        n += k

    comps.append(Cz)
    n_comps.append(n)
    parents.append(parent_of)

# area and centroid of every component
areas, centroids = [], []
for Cz, n in zip(comps, n_comps):
    if n == 0:
        areas.append(np.zeros(0))
        centroids.append(np.zeros((0, 2)))
        continue
    ys, xs = np.nonzero(Cz)
    ids = Cz[ys, xs] - 1
    area = np.bincount(ids, minlength=n).astype(float)
    cy = np.bincount(ids, weights=ys, minlength=n) / area
    cx = np.bincount(ids, weights=xs, minlength=n) / area
    areas.append(area)
    centroids.append(np.stack([cy, cx], axis=1))

# one volume where every component in the whole stack has a unique id
offsets = np.concatenate([[0], np.cumsum(n_comps)[:-1]])
comp_vol = np.zeros(G.shape, dtype=np.int64)
for z in range(Z):
    m = comps[z] > 0
    comp_vol[z][m] = comps[z][m] + offsets[z]

Path(COMPONENTS_OUT).parent.mkdir(parents=True, exist_ok=True)
imwrite(COMPONENTS_OUT, comp_vol, dtype=np.int64)

# candidate edges between slices, with the features the RF uses
rows = []
dzs = [1, 2] if USE_SKIP_LINKS else [1]

for dz in dzs:
    for z in range(Z - dz):
        if n_comps[z] == 0 or n_comps[z + dz] == 0:
            continue

        src, tgt, overlap = overlap_pairs(comps[z], comps[z + dz])

        for c0, c1, o in zip(src, tgt, overlap):
            a0 = areas[z][c0 - 1]
            a1 = areas[z + dz][c1 - 1]
            dy, dx = centroids[z + dz][c1 - 1] - centroids[z][c0 - 1]

            inst_src = parents[z][c0 - 1]
            inst_tgt = parents[z + dz][c1 - 1]

            rows.append({
                "z": z,
                "dz": dz,
                "inst_src": inst_src,
                "inst_tgt": inst_tgt,
                "c_src": int(c0),
                "c_tgt": int(c1),
                "label_same": int(inst_src == inst_tgt),
                "overlap_px": int(o),
                "IoU": o / (a0 + a1 - o),
                "area_src": a0,
                "area_tgt": a1,
                "area_ratio": min(a0, a1) / max(a0, a1),
                "dy": dy,
                "dx": dx,
                "d_centroid": np.hypot(dy, dx),
            })

df = pd.DataFrame(rows)
Path(EDGES_OUT).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(EDGES_OUT, index=False)

print(f"{len(df)} edges, volume shape {comp_vol.shape}")
