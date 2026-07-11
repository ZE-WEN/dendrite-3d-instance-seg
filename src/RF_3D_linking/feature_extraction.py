from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from tifffile import imread, imwrite


# ---------------------------------------------------------------------
# Connected components per slice, within each 3D instance id
# ---------------------------------------------------------------------

def _componentize_instance_slice(Gz: np.ndarray, gid: int, structure: Optional[np.ndarray]) -> Optional[Tuple[np.ndarray, int, int, int, int, int]]:
    """
    Find connected components of a given GT instance id gid on a single slice Gz.
    Returns labeled subarray and its bbox in the full slice coordinates.
    """
    ys, xs = np.nonzero(Gz == gid)
    if ys.size == 0:
        return None
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    sub = (Gz[y0:y1, x0:x1] == gid)
    labs, K = ndi.label(sub.astype(np.uint8), structure=structure)
    if K == 0:
        return None
    return labs, K, y0, y1, x0, x1


def per_slice_components_within_instances(G: np.ndarray, conn: int = 1) -> Tuple[List[np.ndarray], List[int], List[np.ndarray]]:
    """
    For each slice z, compute 2D connected components *within each GT instance id* on that slice.
    Returns:
      C_list[z]  : int32 map of component ids 1..Kz on slice z (0 background)
      counts[z]  : number of components Kz on slice z
      parents[z] : uint32 array of length Kz, parent GT instance id for each component (1-indexed)
    """
    Z, Y, X = G.shape
    C_list: List[np.ndarray] = []
    counts: List[int] = []
    parents: List[np.ndarray] = []

    structure = np.ones((3, 3), dtype=np.uint8) if conn == 2 else None  # 4-conn if None; 8-conn if 3x3

    for z in range(Z):
        Gz = G[z]
        Cz = np.zeros_like(Gz, dtype=np.int32)
        parent_ids: List[int] = []
        next_c = 1

        gids = np.unique(Gz)
        gids = gids[gids > 0]  # ignore background

        for gid in gids:
            out = _componentize_instance_slice(Gz, gid, structure)
            if out is None:
                continue
            labs, K, y0, y1, x0, x1 = out
            if K <= 0:
                continue
            # paste with offset
            block = labs.astype(np.int32)
            block[block > 0] += (next_c - 1)
            Cz[y0:y1, x0:x1] = np.where(block > 0, block, Cz[y0:y1, x0:x1])
            # record parents (1-indexed components)
            parent_ids.extend([int(gid)] * K)
            next_c += K

        Kz = next_c - 1
        C_list.append(Cz)
        counts.append(Kz)
        parents.append(np.asarray(parent_ids, dtype=np.uint32))

    return C_list, counts, parents


# ---------------------------------------------------------------------
# Per-component sizes and centroids for each slice
# ---------------------------------------------------------------------

def component_sizes_centroids(C_list: List[np.ndarray], counts: List[int]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    For each slice z, compute:
      sizes[z]     : float64 array, length Kz, pixel area per component
      centroids[z] : float64 array, shape (Kz, 2) with (y, x) per component
    """
    sizes: List[np.ndarray] = []
    centroids: List[np.ndarray] = []
    for z, (Cz, Kz) in enumerate(zip(C_list, counts)):
        if Kz == 0:
            sizes.append(np.zeros((0,), dtype=np.float64))
            centroids.append(np.zeros((0, 2), dtype=np.float64))
            continue
        flat_counts = np.bincount(Cz.ravel(), minlength=Kz + 1).astype(np.float64)
        areas = flat_counts[1:]  # drop background
        # Centroids
        ys, xs = np.nonzero(Cz)
        ids = Cz[ys, xs] - 1  # 0-based component indices
        sumy = np.bincount(ids, weights=ys, minlength=Kz).astype(np.float64)
        sumx = np.bincount(ids, weights=xs, minlength=Kz).astype(np.float64)
        with np.errstate(divide='ignore', invalid='ignore'):
            cy = np.where(areas > 0, sumy / areas, 0.0)
            cx = np.where(areas > 0, sumx / areas, 0.0)
        centroids.append(np.stack([cy, cx], axis=1))
        sizes.append(areas)
    return sizes, centroids


# ---------------------------------------------------------------------
# Overlap pairs across slices
# ---------------------------------------------------------------------

def overlap_triplets(C0: np.ndarray, C1: np.ndarray, min_overlap_px: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute all (c0, c1, overlap_px) pairs between two labeled slices C0 and C1 (labels in 0..K),
    keeping only pairs with overlap >= min_overlap_px.
    Returns arrays of c0 (1-based), c1 (1-based), and overlap counts.
    """
    mask = (C0 > 0) & (C1 > 0)
    if not np.any(mask):
        return (np.zeros(0, dtype=np.int32),
                np.zeros(0, dtype=np.int32),
                np.zeros(0, dtype=np.int32))
    a = C0[mask].astype(np.int64)
    b = C1[mask].astype(np.int64)
    key = a * (b.max() + 1) + b  # perfect hashing
    uniq, counts = np.unique(key, return_counts=True)
    c0 = (uniq // (b.max() + 1)).astype(np.int32)
    c1 = (uniq % (b.max() + 1)).astype(np.int32)
    keep = counts >= int(min_overlap_px)
    return c0[keep], c1[keep], counts[keep].astype(np.int32)


# ---------------------------------------------------------------------
# Global component volume (for visualization/bookkeeping)
# ---------------------------------------------------------------------

def global_offsets(counts: List[int]) -> np.ndarray:
    offs = np.zeros(len(counts), dtype=np.int64)
    total = 0
    for i, k in enumerate(counts):
        offs[i] = total
        total += k
    return offs


def build_components_global_volume(C_list: List[np.ndarray], counts: List[int]) -> np.ndarray:
    """
    Build a ZYX volume where each component in the entire stack has a unique global id.
    """
    Z = len(C_list)
    Y, X = (C_list[0].shape if Z > 0 else (0, 0))
    out = np.zeros((Z, Y, X), dtype=np.int64)
    offs = global_offsets(counts)
    for z, Cz in enumerate(C_list):
        Kz = counts[z]
        if Kz == 0:
            continue
        block = Cz.copy().astype(np.int64)
        nonzero = block > 0
        block[nonzero] += offs[z]
        out[z] = block
    return out


# ---------------------------------------------------------------------
# Main: derive edges + features (+ optional CSV/TIF I/O)
# ---------------------------------------------------------------------

def derive_edges_and_features(
    gt_path_or_array: str | Path | np.ndarray,
    min_overlap_px: int = 15,
    connectivity: int = 1,
    use_skip_links: bool = True,
    max_skip_dz: int = 2,
    save_components_tif: Optional[str | Path] = None,
    save_edges_csv: Optional[str | Path] = None
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Build an edge table of candidate links across slices with features and labels,
    and a global-components volume for visualization.

    CSV columns (ordered as requested):
      z, dz, inst_src, inst_tgt, c_src, c_tgt, label_same,
      overlap_px, IoU, area_src, area_tgt, area_ratio, dy, dx, d_centroid
    """
    # Load GT
    if isinstance(gt_path_or_array, (str, Path)):
        G = imread(str(gt_path_or_array))
    else:
        G = gt_path_or_array
    G = np.asarray(G)
    assert G.ndim == 3, "GT must be a 3D volume (Z,Y,X)"
    if G.dtype.kind in ('f',):
        G = G.astype(np.int64)
    Z, Y, X = G.shape

    # 1) Components per slice (within each instance id)
    C_list, counts, parents = per_slice_components_within_instances(G, conn=connectivity)

    # 2) Per-component size & centroid per slice
    sizes, cents = component_sizes_centroids(C_list, counts)

    # 3) Build global components volume (for TIF output)
    comp_vol = build_components_global_volume(C_list, counts)

    # Optionally save the components volume
    if save_components_tif:
        Path(save_components_tif).parent.mkdir(parents=True, exist_ok=True)
        imwrite(str(save_components_tif), comp_vol.astype(np.int64), dtype=np.int64)

    # 4) Build edges
    rows: List[Dict] = []

    def add_edges(z: int, dz: int):
        C0, C1 = C_list[z], C_list[z + dz]
        K0, K1 = counts[z], counts[z + dz]
        if K0 == 0 or K1 == 0:
            return
        c0s, c1s, overlaps = overlap_triplets(C0, C1, min_overlap_px=min_overlap_px)
        if c0s.size == 0:
            return

        # parents arrays are 1-indexed by component id
        p0 = parents[z]
        p1 = parents[z + dz]

        areas0 = sizes[z]
        areas1 = sizes[z + dz]
        cents0 = cents[z]
        cents1 = cents[z + dz]

        for c0, c1, o in zip(c0s, c1s, overlaps):
            a0 = float(areas0[c0 - 1])
            a1 = float(areas1[c1 - 1])
            if a0 <= 0 or a1 <= 0:
                continue
            # IoU
            denom = (a0 + a1 - o)
            iou = float(o) / float(denom) if denom > 0 else 0.0
            # centroid deltas
            y0, x0 = cents0[c0 - 1]
            y1, x1 = cents1[c1 - 1]
            dy = float(y1 - y0)
            dx = float(x1 - x0)
            dcent = float(np.hypot(dy, dx))
            # area ratio
            area_ratio = float(min(a0, a1) / max(a0, a1))

            inst_src = int(p0[c0 - 1]) if (c0 - 1) < len(p0) else 0
            inst_tgt = int(p1[c1 - 1]) if (c1 - 1) < len(p1) else 0
            label_same = int(inst_src == inst_tgt)

            rows.append({
                # Topology/meta
                "z": int(z),
                "dz": int(dz),
                "inst_src": inst_src,
                "inst_tgt": inst_tgt,
                "c_src": int(c0),
                "c_tgt": int(c1),
                "label_same": label_same,
                # Geometry
                "overlap_px": int(o),
                "IoU": float(iou),
                "area_src": float(a0),
                "area_tgt": float(a1),
                "area_ratio": float(area_ratio),
                "dy": float(dy),
                "dx": float(dx),
                "d_centroid": float(dcent),
            })

    for z in range(Z - 1):
        add_edges(z, 1)
        if use_skip_links and (max_skip_dz >= 2) and (z + 2 < Z):
            add_edges(z, 2)

    # 5) Assemble DataFrame in requested column order
    df = pd.DataFrame(rows, columns=[
        "z", "dz", "inst_src", "inst_tgt",
        "c_src", "c_tgt", "label_same",
        "overlap_px", "IoU", "area_src", "area_tgt",
        "area_ratio", "dy", "dx", "d_centroid"
    ])

    # Optionally save CSV (NOTE: no gid columns by design)
    if save_edges_csv:
        Path(save_edges_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_edges_csv, index=False)

    return df, comp_vol


if __name__ == "__main__":
    # Simple CLI passthrough (optional)
    import argparse
    ap = argparse.ArgumentParser(description="Derive edge features and labels from a 3D instance volume.")
    ap.add_argument("--gt", required=True, help="Path to 3D instance-labeled TIFF")
    ap.add_argument("--min_overlap_px", type=int, default=15)
    ap.add_argument("--connectivity", type=int, default=1, choices=[1, 2])
    ap.add_argument("--use_skip_links", action="store_true")
    ap.add_argument("--max_skip_dz", type=int, default=2)
    ap.add_argument("--save_components_tif", default=None)
    ap.add_argument("--save_edges_csv", default=None)
    args = ap.parse_args()

    df, comp_vol = derive_edges_and_features(
        gt_path_or_array=args.gt,
        min_overlap_px=args.min_overlap_px,
        connectivity=args.connectivity,
        use_skip_links=args.use_skip_links,
        max_skip_dz=args.max_skip_dz,
        save_components_tif=args.save_components_tif,
        save_edges_csv=args.save_edges_csv
    )
    print(f"Derived {len(df)} edges. Volume shape: {comp_vol.shape}")
