import csv
import multiprocessing as mp
from multiprocessing import shared_memory

import numpy as np
import tifffile as tiff
from scipy import ndimage as ndi
from skimage.measure import label, regionprops
from skimage.morphology import binary_closing, disk, remove_small_objects
from skimage.transform import resize
from tqdm import tqdm

# looser check used when only one candidate id is found
SINGLE_COVERAGE_MIN = 0.4
SINGLE_BBOX_IOU_MIN = 0.2

FOUR_CONN = ndi.generate_binary_structure(2, 1)


# ---------------------------------------------------------------
# reading, writing, resizing
# ---------------------------------------------------------------

def read_tiff_stack(path, memmap=False):
    return tiff.memmap(path, mode="r") if memmap else tiff.imread(path)


def save_tiff_stack(path, arr):
    tiff.imwrite(path, arr, photometric="minisblack", metadata={"axes": "ZYX"})


def downsample_nearest(vol, target_xy=1024):
    """shrink every slice to target_xy x target_xy, nearest neighbour so the ids survive"""
    out = np.zeros((vol.shape[0], target_xy, target_xy), dtype=vol.dtype)
    for z in range(vol.shape[0]):
        out[z] = resize(vol[z], (target_xy, target_xy), order=0,
                        preserve_range=True, anti_aliasing=False).astype(vol.dtype)
    return out


# ---------------------------------------------------------------
# small helpers for the repair
# ---------------------------------------------------------------

def largest_region(mask):
    regions = regionprops(label(mask.astype(np.uint8), connectivity=1))
    return max(regions, key=lambda p: p.area) if regions else None


def bbox_iou(b1, b2):
    h = max(0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
    w = max(0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
    inter = h * w
    area1 = max(0, b1[2] - b1[0]) * max(0, b1[3] - b1[1])
    area2 = max(0, b2[2] - b2[0]) * max(0, b2[3] - b2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def centroid_distance(a, b):
    pa, pb = largest_region(a), largest_region(b)
    if pa is None or pb is None:
        return np.inf
    return float(np.hypot(pa.centroid[0] - pb.centroid[0], pa.centroid[1] - pb.centroid[1]))


def coverage(mask, other):
    """fraction of mask that is covered by other"""
    n = int(mask.sum())
    return int((mask & other).sum()) / n if n else 0.0


def clean_mask(mask, cfg):
    if not mask.any():
        return mask.copy()
    out = remove_small_objects(mask, min_size=cfg["min_area"])
    if cfg["closing_radius"] > 0:
        out = binary_closing(out, footprint=disk(cfg["closing_radius"]))
    if cfg["fill_holes"]:
        out = ndi.binary_fill_holes(out)
    return out


def good_shape(mask, cfg):
    """dendrites cut across look like round, solid blobs"""
    p = largest_region(mask)
    return (p is not None
            and p.area >= cfg["min_area"]
            and p.solidity >= cfg["solidity_min"]
            and p.eccentricity <= cfg["eccentricity_max"])


def candidate_ids(prev_mask, prev_bbox, curr, cfg):
    """ids on the current slice that could continue prev_mask"""
    vals = curr[prev_mask]
    vals = vals[vals > 0]
    if vals.size:
        ids, counts = np.unique(vals, return_counts=True)
        ids = ids[counts >= cfg["min_overlap_pixels"]]
        if ids.size:
            return ids.tolist()

    # nothing overlaps enough, so look around the previous box instead
    m = cfg["bbox_margin"]
    h, w = curr.shape
    r0, c0, r1, c1 = prev_bbox
    crop = curr[max(0, r0 - m):min(h, r1 + m), max(0, c0 - m):min(w, c1 + m)]
    ids = np.unique(crop)
    return ids[ids > 0].tolist()


# ---------------------------------------------------------------
# step 1, repair in the x view
# ---------------------------------------------------------------

def repair_transition(prev, curr, cfg, z=None, log=None):
    """
    For every id on the previous slice, find the pieces on the current slice that
    continue it. If those pieces together form one good looking object, they are
    merged and take the previous id.
    """
    out = curr.copy()
    used = set()

    for pid in np.unique(prev):
        if pid == 0:
            continue
        prev_mask = prev == pid
        prev_region = largest_region(prev_mask)
        if prev_region is None or prev_region.area < cfg["min_area"]:
            continue

        cands = [c for c in candidate_ids(prev_mask, prev_region.bbox, out, cfg) if c not in used]
        if not cands:
            continue

        union = clean_mask(np.isin(out, cands), cfg)
        region = largest_region(union)
        if region is None:
            continue

        cov = coverage(prev_mask, union)
        dist = centroid_distance(prev_mask, union)
        biou = bbox_iou(prev_region.bbox, region.bbox)
        mode = None

        if (cov >= cfg["coverage_min"]
                and dist <= cfg["centroid_dist_max"]
                and biou >= cfg["bbox_iou_min"]
                and good_shape(union, cfg)):
            out[np.isin(out, cands)] = 0
            out[union] = pid
            used.update(cands)
            mode = "merge_union"

        elif len(cands) == 1:
            only = out == cands[0]
            only_region = largest_region(only)
            if (only_region is not None
                    and coverage(prev_mask, only) >= SINGLE_COVERAGE_MIN
                    and centroid_distance(prev_mask, only) <= cfg["centroid_dist_max"]
                    and bbox_iou(prev_region.bbox, only_region.bbox) >= SINGLE_BBOX_IOU_MIN):
                out[only] = pid
                used.add(cands[0])
                mode = "single_relabel"

        if mode and log is not None:
            log.append({
                "slice_index": z,
                "prev_id": int(pid),
                "candidate_ids": ",".join(map(str, cands)),
                "coverage": cov,
                "centroid_dist": dist,
                "bbox_iou": biou,
                "mode": mode,
            })

    return out


_worker = {}


def _init_worker(name, shape, dtype):
    shm = shared_memory.SharedMemory(name=name)
    _worker["shm"] = shm
    _worker["vol"] = np.ndarray(shape, dtype=np.dtype(dtype), buffer=shm.buf)


def _repair_chunk(task):
    start, end, cfg = task
    sub = _worker["vol"][start:end].copy()
    log = []
    for s in range(1, len(sub)):
        sub[s] = repair_transition(sub[s - 1], sub[s], cfg, z=start + s, log=log)
    return start, end, sub, log


def forward_pass(vol, cfg):
    """
    Repair from front to back. The stack is cut into chunks that run in parallel,
    then the first slice of every chunk is repaired against the slice before it.
    """
    n = vol.shape[0]
    step = cfg["chunk_size"]
    chunks = [(s, min(s + step, n)) for s in range(0, n, step)]

    shm = shared_memory.SharedMemory(create=True, size=vol.nbytes)
    try:
        shared = np.ndarray(vol.shape, dtype=vol.dtype, buffer=shm.buf)
        shared[:] = vol
        del shared

        out = np.array(vol)
        log = []
        tasks = [(s, e, cfg) for s, e in chunks]

        ctx = mp.get_context("spawn")
        with ctx.Pool(cfg["num_workers"], initializer=_init_worker,
                      initargs=(shm.name, vol.shape, vol.dtype.str)) as pool:
            for s, e, sub, chunk_log in tqdm(pool.imap_unordered(_repair_chunk, tasks),
                                             total=len(tasks), desc="forward pass"):
                out[s:e] = sub
                log += chunk_log

        # the chunks never saw each other, so fix the seams
        for s, _ in chunks[1:]:
            out[s] = repair_transition(out[s - 1], out[s], cfg)

        return out, log
    finally:
        shm.close()
        shm.unlink()


def backward_pass(vol, cfg):
    out = vol.copy()
    log = []
    for s in tqdm(range(len(out) - 2, -1, -1), desc="backward pass"):
        out[s] = repair_transition(out[s + 1], out[s], cfg, z=s, log=log)
    return out, log


def run_repair(input_tif, output_tif, log_csv, cfg, memmap=False):
    """
    Look at the stack from the side. The volume is turned so that x becomes the
    slice axis, repaired forward (and backward), then turned back.
    """
    vol = read_tiff_stack(input_tif, memmap=memmap)
    if vol.ndim != 3:
        raise ValueError(f"expected a (Z, Y, X) volume, got {vol.shape}")

    side = np.transpose(vol, (2, 0, 1))   # (X, Z, Y)
    side, log = forward_pass(side, cfg)
    if cfg["run_backward_pass"]:
        side, back_log = backward_pass(side, cfg)
        log += back_log

    out = np.transpose(side, (1, 2, 0)).astype(vol.dtype, copy=False)
    save_tiff_stack(output_tif, out)

    with open(log_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["slice_index", "prev_id", "candidate_ids",
                                               "coverage", "centroid_dist", "bbox_iou", "mode"])
        writer.writeheader()
        writer.writerows(log)

    print(f"repair: {len(log)} changes, saved to {output_tif}")


# ---------------------------------------------------------------
# step 2, merge and absorb small pieces
# ---------------------------------------------------------------

def absorb_small_pieces(sl, min_area=500):
    """
    Pieces of an instance smaller than min_area go to the neighbouring instance
    they touch the most. Pieces that only touch background stay as they are.
    """
    out = sl.copy()
    for iid in np.unique(out):
        if iid == 0:
            continue
        pieces, n = ndi.label(out == iid, structure=FOUR_CONN)
        for k in range(1, n + 1):
            piece = pieces == k
            if piece.sum() >= min_area:
                continue
            ring = ndi.binary_dilation(piece, structure=FOUR_CONN) & ~piece
            nb = out[ring]
            nb = nb[(nb != 0) & (nb != iid)]
            if nb.size == 0:
                continue
            ids, counts = np.unique(nb, return_counts=True)
            out[piece] = ids[np.argmax(counts)]
    return out


def run_mg(vol1_path, vol2_path, output_path, min_component_area=500):
    """
    Combine the original and the repaired volume. The original is kept, and the
    repair only fills voxels that were background. Then small pieces are absorbed.
    """
    original = tiff.imread(str(vol1_path))
    repaired = tiff.imread(str(vol2_path))
    if original.shape != repaired.shape:
        raise ValueError(f"shapes differ: {original.shape} vs {repaired.shape}")

    merged = original.copy()
    new = (original == 0) & (repaired != 0)
    merged[new] = repaired[new]

    for z in tqdm(range(merged.shape[0]), desc="absorbing small pieces"):
        merged[z] = absorb_small_pieces(merged[z], min_area=min_component_area)

    tiff.imwrite(str(output_path), merged)
    print("merge: saved to", output_path)


# ---------------------------------------------------------------
# step 3, make ids consistent over a 5-slice window
# ---------------------------------------------------------------

def harmonize_ids(vol, overlap_thresh=0.95):
    """
    Slide a window of 5 slices through the stack. For every instance on the first
    slice, find the id on each of the next 4 slices that covers it best. If that
    id covers at least overlap_thresh of it, it takes the first slice's id, and
    empty slices in between are filled with the instance's shape.
    """
    vol = vol.copy()
    filled = relabelled = 0

    for start in tqdm(range(vol.shape[0] - 4), desc="harmonizing ids"):
        win = vol[start:start + 5].copy()
        anchor = win[0]

        for aid in np.unique(anchor):
            if aid == 0:
                continue
            amask = anchor == aid
            gaps = []

            for k in range(1, 5):
                ids = np.unique(win[k][amask])
                ids = ids[ids != 0]
                if ids.size == 0:
                    gaps.append(k)
                    continue

                best, best_ov = max(((int(c), coverage(amask, win[k] == c)) for c in ids),
                                    key=lambda t: t[1])
                if best_ov < overlap_thresh:
                    continue

                for g in gaps:
                    fill = amask & (win[g] == 0)
                    win[g][fill] = aid
                    filled += int(fill.sum())
                gaps = []

                if best != aid:
                    m = win[k] == best
                    win[k][m] = aid
                    relabelled += int(m.sum())

        vol[start:start + 5] = win

    return vol, filled, relabelled


def run_after_mg(input_path, output_path, overlap_thresh=0.95):
    vol = tiff.imread(str(input_path))
    if vol.ndim == 4 and vol.shape[-1] == 1:
        vol = vol[..., 0]
    if vol.ndim != 3:
        raise ValueError(f"expected a (Z, Y, X) volume, got {vol.shape}")
    if not np.issubdtype(vol.dtype, np.integer):
        vol = vol.astype(np.int32)

    fixed, filled, relabelled = harmonize_ids(vol, overlap_thresh=overlap_thresh)
    tiff.imwrite(str(output_path), fixed)

    print(f"harmonize: {filled} gap voxels filled, {relabelled} voxels relabelled")
    print("saved to", output_path)
