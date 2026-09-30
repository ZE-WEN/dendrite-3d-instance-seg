from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from tifffile import imread, imwrite

from .greedy import UnionFind, greedy_linking

# These must be the same columns, in the same order, as FEATURES in train_rf.py.
# The forest picks features by position, so a mismatch does not raise an error.
FEATURES = ["overlap_px", "IoU", "area_src", "area_tgt",
            "area_ratio", "dy", "dx", "d_centroid"]

EDGE_COLUMNS = ["z", "dz", "c_src", "c_tgt", "gid_src", "gid_tgt",
                "overlap_px", "IoU", "area_src", "area_tgt",
                "area_ratio", "dy", "dx", "d_centroid"]


# ---------------------------------------------------------------
# splitting the labels into 2D components and building edges
# ---------------------------------------------------------------

def split_into_components(volume, connectivity=1):
    """
    On every slice, split each instance into its connected pieces so that two
    separate blobs sharing one label are treated as two components.
    Returns one int32 slice per z (components numbered 1..k) and the counts k.
    """
    structure = np.ones((3, 3)) if connectivity == 2 else None
    comps, counts = [], []

    for sl in volume:
        out = np.zeros(sl.shape, dtype=np.int32)
        n = 0
        for gid in np.unique(sl):
            if gid == 0:
                continue
            ys, xs = np.nonzero(sl == gid)
            y0, y1 = ys.min(), ys.max() + 1
            x0, x1 = xs.min(), xs.max() + 1

            pieces, k = ndi.label(sl[y0:y1, x0:x1] == gid, structure=structure)
            window = out[y0:y1, x0:x1]
            window[pieces > 0] = pieces[pieces > 0] + n
            n += k

        comps.append(out)
        counts.append(n)

    return comps, counts


def component_geometry(Cz, k):
    """area and centroid (y, x) of every component on one slice"""
    if k == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    ys, xs = np.nonzero(Cz)
    ids = Cz[ys, xs]
    area = np.bincount(ids, minlength=k + 1)[1:]
    denom = np.maximum(area, 1)
    cy = np.bincount(ids, weights=ys, minlength=k + 1)[1:] / denom
    cx = np.bincount(ids, weights=xs, minlength=k + 1)[1:] / denom
    return area, cy, cx


def overlapping_pairs(C0, k0, C1, k1, min_px):
    """rows of (component on slice 0, component on slice 1, overlap in pixels)"""
    empty = np.empty((0, 3), dtype=np.int64)
    if k0 == 0 or k1 == 0:
        return empty
    both = (C0 > 0) & (C1 > 0)
    if not both.any():
        return empty

    pair_id = C0[both].astype(np.int64) * (k1 + 1) + C1[both]
    counts = np.bincount(pair_id)
    idx = np.flatnonzero(counts)
    overlap = counts[idx]
    keep = overlap >= min_px
    if not keep.any():
        return empty

    c0 = idx // (k1 + 1)
    c1 = idx % (k1 + 1)
    return np.stack([c0[keep], c1[keep], overlap[keep]], axis=1)


def build_edges(volume, min_overlap_px=8, connectivity=1, use_skip_links=False):
    """
    Candidate links between components on slice z and z+1 (and z+2 if
    use_skip_links) together with the features the forest needs.
    """
    comps, counts = split_into_components(volume, connectivity)
    geometry = [component_geometry(c, k) for c, k in zip(comps, counts)]
    offsets = np.concatenate([[0], np.cumsum(counts)[:-1]])

    Z = len(comps)
    rows = []

    for dz in ([1, 2] if use_skip_links else [1]):
        for z in range(Z - dz):
            pairs = overlapping_pairs(comps[z], counts[z],
                                      comps[z + dz], counts[z + dz], min_overlap_px)
            area0, cy0, cx0 = geometry[z]
            area1, cy1, cx1 = geometry[z + dz]

            for c0, c1, ov in pairs:
                a0, a1 = int(area0[c0 - 1]), int(area1[c1 - 1])
                dy = float(cy1[c1 - 1] - cy0[c0 - 1])
                dx = float(cx1[c1 - 1] - cx0[c0 - 1])

                rows.append({
                    "z": z,
                    "dz": dz,
                    "c_src": int(c0),
                    "c_tgt": int(c1),
                    "gid_src": int(offsets[z] + c0),
                    "gid_tgt": int(offsets[z + dz] + c1),
                    "overlap_px": int(ov),
                    "IoU": ov / (a0 + a1 - ov),
                    "area_src": a0,
                    "area_tgt": a1,
                    "area_ratio": min(a0, a1) / max(a0, a1),
                    "dy": dy,
                    "dx": dx,
                    "d_centroid": float(np.hypot(dy, dx)),
                })

    return pd.DataFrame(rows, columns=EDGE_COLUMNS), comps, counts


def score_edges(df_edges, rf_model_path):
    """add a p_link column with the forest's probability that two components belong together"""
    df = df_edges.copy()
    if df.empty:
        df["p_link"] = np.zeros(0)
        return df
    rf = joblib.load(rf_model_path)
    df["p_link"] = rf.predict_proba(df[FEATURES].values)[:, 1]
    return df


def load_volume(volume_or_path):
    if isinstance(volume_or_path, (str, Path)):
        return imread(str(volume_or_path)).astype(np.uint32)
    return np.asarray(volume_or_path).astype(np.uint32)


# ---------------------------------------------------------------
# single pass linking
# ---------------------------------------------------------------

def run_full_inference(
    proposal_tif_path,
    rf_model_path,
    out_edges_csv,
    out_tracks_tif,
    min_overlap_px=8,
    connectivity=1,
    use_skip_links=False,
    p_thresh=0.6,
):
    """
    Link a 3D instance volume (ZYX tif) across slices.
    Saves the scored edges as csv and the linked volume as tif.
    """
    volume = load_volume(proposal_tif_path)
    df_edges, comps, counts = build_edges(volume, min_overlap_px, connectivity, use_skip_links)
    df_edges = score_edges(df_edges, rf_model_path)

    Path(out_edges_csv).parent.mkdir(parents=True, exist_ok=True)
    df_edges.to_csv(out_edges_csv, index=False)

    tracks = greedy_linking(df_edges, comps, counts, p_thresh=p_thresh)

    Path(out_tracks_tif).parent.mkdir(parents=True, exist_ok=True)
    imwrite(out_tracks_tif, tracks, metadata={"axes": "ZYX"})
    return df_edges, tracks


# ---------------------------------------------------------------
# second pass, filling gaps of one slice (dz=2)
# ---------------------------------------------------------------

def track_of_components(track_slice, Cz, k):
    """the track id that most pixels of each component carry (index 0 unused)"""
    out = np.zeros(k + 1, dtype=np.uint32)
    if k == 0:
        return out
    ys, xs = np.nonzero(Cz)
    table = pd.DataFrame({"comp": Cz[ys, xs], "tid": track_slice[ys, xs]})
    mode = table.groupby("comp")["tid"].agg(lambda s: np.bincount(s).argmax())
    out[mode.index.values] = mode.values
    return out


def find_endpoints(tracks, comps, counts):
    """
    fwd[z] holds the components with no same-track continuation on slice z+1,
    bwd[z] the ones with no continuation on slice z-1.
    """
    Z = len(comps)
    fwd = [set() for _ in range(Z)]
    bwd = [set() for _ in range(Z)]

    for z in range(Z):
        Cz, k = comps[z], counts[z]
        if k == 0:
            continue

        own_track = track_of_components(tracks[z], Cz, k)
        ys, xs = np.nonzero(Cz)
        ids = Cz[ys, xs]
        own = own_track[ids]

        for nz, ends in ((z + 1, fwd[z]), (z - 1, bwd[z])):
            if not 0 <= nz < Z:
                ends.update(range(1, k + 1))  # edge of the stack
                continue
            same = (tracks[nz][ys, xs] == own) & (own > 0)
            continues = np.bincount(ids, weights=same.astype(float), minlength=k + 1) > 0
            ends.update(c for c in range(1, k + 1) if not continues[c])

    return fwd, bwd


def gap_fill_dz2(df_edges, comps, counts, tracks,
                 p_thresh_dz2=0.8, d_centroid_max=3.0, area_ratio_min=0.5):
    """
    Bridge one missing slice. Only track ends can be linked (the forward end on
    slice z to the backward end on slice z+2), one partner each, and the pair
    also has to be close and similar in size. Existing dz=1 links stay as they are.
    """
    Z = len(comps)
    track_of = [track_of_components(tracks[z], comps[z], counts[z]) for z in range(Z)]
    fwd, bwd = find_endpoints(tracks, comps, counts)

    edges = df_edges[(df_edges["dz"] == 2)
                     & (df_edges["p_link"] >= p_thresh_dz2)
                     & (df_edges["d_centroid"] <= d_centroid_max)
                     & (df_edges["area_ratio"] >= area_ratio_min)]

    uf = UnionFind()
    next_id = int(tracks.max()) + 1

    for z in range(Z - 2):
        group = edges[edges["z"] == z].sort_values("p_link", ascending=False)
        used_src, used_tgt = set(), set()

        for c0, c1 in zip(group["c_src"].astype(int), group["c_tgt"].astype(int)):
            if c0 in used_src or c1 in used_tgt:
                continue
            if c0 not in fwd[z] or c1 not in bwd[z + 2]:
                continue

            t0 = int(track_of[z][c0])
            t1 = int(track_of[z + 2][c1])

            if t0 and t1:
                if t0 != t1:
                    keep = uf.union(t0, t1)
                    track_of[z][c0] = keep
                    track_of[z + 2][c1] = keep
            elif t0:
                track_of[z + 2][c1] = uf.find(t0)
            elif t1:
                track_of[z][c0] = uf.find(t1)
            else:
                track_of[z][c0] = next_id
                track_of[z + 2][c1] = next_id
                next_id += 1

            used_src.add(c0)
            used_tgt.add(c1)

    out = np.zeros(tracks.shape, dtype=np.uint32)
    for z in range(Z):
        ids = track_of[z]
        for i in np.flatnonzero(ids):
            ids[i] = uf.find(int(ids[i]))
        out[z] = ids[comps[z]]
    return out


def run_iterative_inference(
    proposal_tif_path,
    rf_model_path,
    out_edges_csv_A,
    out_tracks_tif_A,
    out_edges_csv_B,
    out_tracks_tif_final,
    min_overlap_px_A=8,
    p_thresh_dz1=0.6,
    p_thresh_dz2=0.7,
    d_centroid_max=3.0,
    area_ratio_min=0.5,
):
    """
    Pass A links neighbouring slices only (dz=1).
    Pass B then bridges single missing slices (dz=2) at track ends.
    """
    volume = load_volume(proposal_tif_path)
    df_all, comps, counts = build_edges(volume, min_overlap_px_A, connectivity=1, use_skip_links=True)
    df_all = score_edges(df_all, rf_model_path)

    # pass A
    df_A = df_all[df_all["dz"] == 1]
    Path(out_edges_csv_A).parent.mkdir(parents=True, exist_ok=True)
    df_A.to_csv(out_edges_csv_A, index=False)
    tracks_A = greedy_linking(df_A, comps, counts, p_thresh=p_thresh_dz1)
    imwrite(out_tracks_tif_A, tracks_A, metadata={"axes": "ZYX"})

    # pass B
    df_all.to_csv(out_edges_csv_B, index=False)
    tracks_final = gap_fill_dz2(df_all, comps, counts, tracks_A,
                                p_thresh_dz2=p_thresh_dz2,
                                d_centroid_max=d_centroid_max,
                                area_ratio_min=area_ratio_min)
    imwrite(out_tracks_tif_final, tracks_final, metadata={"axes": "ZYX"})

    return df_A, tracks_A, df_all, tracks_final
