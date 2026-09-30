import glob
import os

import imageio.v3 as iio
import numpy as np
import tifffile as tiff
from scipy import ndimage
from skimage.measure import label, regionprops, regionprops_table
from skimage.morphology import binary_dilation, disk


# ---------------------------------------------------------------
# boxes
# skimage gives boxes as (row0, col0, row1, col1), SAM wants (x0, y0, x1, y1)
# ---------------------------------------------------------------

def rc_to_xy(box):
    r0, c0, r1, c1 = box
    return [c0, r0, c1, r1]


def yolo_to_corners(x_center, y_center, width, height, img_size):
    """normalized YOLO box -> pixel corners (x_min, y_min, x_max, y_max)"""
    x_c, y_c = x_center * img_size, y_center * img_size
    w, h = width * img_size, height * img_size

    x_min = max(0, int(x_c - w / 2))
    y_min = max(0, int(y_c - h / 2))
    x_max = min(img_size, int(x_c + w / 2))
    y_max = min(img_size, int(y_c + h / 2))
    return [x_min, y_min, x_max, y_max]


def add_boxes(store, key, new_boxes):
    """append boxes to store[key], which is an (n, 4) array"""
    new = np.asarray(list(new_boxes), dtype=int)
    if new.size == 0:
        return
    old = store.get(key)
    if old is None or np.asarray(old).size == 0:
        store[key] = new
    else:
        store[key] = np.vstack([np.asarray(old, dtype=int), new])


def boxes_from_mask(mask, min_area=800, connectivity=1):
    lbl = label(mask, connectivity=connectivity)
    return [p.bbox for p in regionprops(lbl) if p.area >= min_area]


def box_iou(a, b):
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r1 <= r0 or c1 <= c0:
        return 0.0
    inter = (r1 - r0) * (c1 - c0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / float(area_a + area_b - inter + 1e-9)


def drop_duplicate_boxes(boxes, iou_thresh=0.8):
    kept = []
    for b in boxes:
        if all(box_iou(b, k) < iou_thresh for k in kept):
            kept.append(b)
    return kept


def find_missing_boxes(m1, m2, m3, m4, min_area=800, connectivity=1, iou_thresh=0.8):
    """
    m1..m4 are boolean masks of four neighbouring slices.
    Returns boxes to add to slice 2, and boxes to add to slice 3
    (the second list is only for objects missing in both 2 and 3).
    """
    # things that show up later but are missing in slice 2
    missing_2_from_3 = m3 & ~m2
    missing_2_from_4 = m4 & ~m2
    # things in slice 4 that are missing in both 2 and 3
    missing_2_and_3 = m4 & ~m3 & ~m2

    boxes_2 = drop_duplicate_boxes(
        boxes_from_mask(missing_2_from_3, min_area, connectivity)
        + boxes_from_mask(missing_2_from_4, min_area, connectivity),
        iou_thresh,
    )

    boxes_both = boxes_from_mask(missing_2_and_3, min_area, connectivity)
    boxes_2 = drop_duplicate_boxes(boxes_2 + boxes_both, iou_thresh)
    boxes_3 = drop_duplicate_boxes(boxes_both, iou_thresh)
    return boxes_2, boxes_3


# ---------------------------------------------------------------
# reading images
# ---------------------------------------------------------------

def read_volume(path_or_dir):
    """read a .tif stack, or a folder (or glob) of .png slices, as a (Z, Y, X) array"""
    if os.path.isfile(path_or_dir) and path_or_dir.lower().endswith((".tif", ".tiff")):
        return tiff.imread(path_or_dir)

    if os.path.isdir(path_or_dir):
        paths = glob.glob(os.path.join(path_or_dir, "*.png"))
    else:
        paths = glob.glob(path_or_dir)
    if not paths:
        raise FileNotFoundError(f"no png files found at {path_or_dir}")

    try:
        from natsort import natsorted
        paths = natsorted(paths)
    except ImportError:
        paths = sorted(paths, key=lambda p: (len(p), p))

    slices = []
    for p in paths:
        img = iio.imread(p)
        slices.append(img if img.ndim == 2 else img[..., 0])
    return np.stack(slices, axis=0).astype(slices[0].dtype)


# ---------------------------------------------------------------
# merging entangled pieces inside one slice
# ---------------------------------------------------------------

def harmonize_entangled_components_2d(lbl2d, r=2, min_score=0.6):
    """
    Sometimes SAM gives two labels to what is really one object, so the pieces sit
    right next to each other. This looks at touching pieces from different labels,
    scores how well they belong together, and gives merged pieces the label of
    the biggest one.
    """
    lbl2d = np.asarray(lbl2d)
    H, W = lbl2d.shape
    labels = np.unique(lbl2d)
    labels = labels[labels != 0]
    if labels.size == 0:
        return lbl2d.copy()

    # 1) split every label into its 4-connected pieces
    comp_map = np.zeros(lbl2d.shape, dtype=np.int32)
    comp_label, comp_area, comp_bbox = [], [], []
    offset = 0
    for L in labels:
        pieces = label(lbl2d == L, connectivity=1)
        n = pieces.max()
        if n == 0:
            continue
        comp_map[pieces > 0] = pieces[pieces > 0] + offset
        props = regionprops_table(pieces, properties=("label", "area", "bbox"))
        for i in range(n):
            comp_label.append(int(L))
            comp_area.append(int(props["area"][i]))
            comp_bbox.append((int(props["bbox-0"][i]), int(props["bbox-1"][i]),
                              int(props["bbox-2"][i]), int(props["bbox-3"][i])))
        offset += n

    N = offset
    if N == 0:
        return lbl2d.copy()
    comp_label = np.array(comp_label, dtype=np.int32)
    comp_area = np.array(comp_area, dtype=np.int32)

    # small helpers, masks and dilations are cached because pieces get reused a lot
    footprint = disk(r) if r > 0 else None
    bg_dist = ndimage.distance_transform_edt(lbl2d == 0)
    nb8 = ndimage.generate_binary_structure(2, 2)
    masks, dilated = {}, {}

    def mask_of(i):
        if i not in masks:
            masks[i] = comp_map == (i + 1)
        return masks[i]

    def dilated_of(i):
        if i not in dilated:
            m = mask_of(i)
            dilated[i] = binary_dilation(m, footprint=footprint) if r > 0 else m
        return dilated[i]

    def contact_length(mA, mB):
        border = mA & ndimage.binary_dilation(mB, structure=nb8)
        perimeter = np.count_nonzero(ndimage.binary_dilation(mA, structure=nb8) & ~mA) + 1
        return border.sum() / perimeter

    def min_distance(mA, mB):
        d = ndimage.distance_transform_edt(~mB)[mA]
        return float(d.min()) if d.size else 999.0

    def gap_width(dA, dB):
        corridor = (dA & dB) & (bg_dist > 0)
        if not corridor.any():
            return np.inf
        return float(np.median(bg_dist[corridor]))

    # 2) candidate pairs, pruned with the (expanded) bounding boxes
    boxes = [(max(0, a - r), max(0, b - r), min(H, c + r), min(W, d + r))
             for (a, b, c, d) in comp_bbox]

    candidates = [[] for _ in range(N)]
    for i in range(N):
        r1, c1, r2, c2 = boxes[i]
        for j in range(i + 1, N):
            s1, t1, s2, t2 = boxes[j]
            if r1 >= t2 or s1 >= c2 or r2 <= t1 or s2 <= c1:
                continue
            if comp_label[i] != comp_label[j]:  # only pieces from different labels
                candidates[i].append(j)
                candidates[j].append(i)

    # 3) every piece picks its best partner, merge if the score is high enough
    parent = np.arange(N, dtype=np.int32)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if comp_area[ra] >= comp_area[rb]:
            parent[rb] = ra
        else:
            parent[ra] = rb

    for i in range(N):
        mA = mask_of(i)
        best_j, best_score = None, -np.inf

        for j in candidates[i]:
            if find(i) == find(j):
                continue
            mB = mask_of(j)
            dA, dB = dilated_of(i), dilated_of(j)

            union_area = (dA | dB).sum()
            iou_dil = (dA & dB).sum() / union_area if union_area > 0 else 0.0
            contact = contact_length(mA, mB)
            dmin = min_distance(mA, mB)
            gap = gap_width(dA, dB)

            score = (0.2 * iou_dil
                     + 0.8 * contact
                     + 0.5 * np.exp(-dmin / 2.0)
                     - 0.25 * (0.0 if np.isinf(gap) else gap / (r + 1e-6)))
            if score > best_score:
                best_score, best_j = score, j

        if best_j is not None and best_score >= min_score:
            union(i, best_j)

    # 4) every merged group takes the label of its biggest piece
    groups = {}
    for i in range(N):
        groups.setdefault(find(i), []).append(i)

    out = np.zeros_like(lbl2d, dtype=lbl2d.dtype)
    for members in groups.values():
        target = int(comp_label[members][np.argmax(comp_area[members])])
        for cid in members:
            out[comp_map == (cid + 1)] = target
    return out
