from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tifffile
from scipy.optimize import linear_sum_assignment

# change these to your own data
GT_PATH = "/path/to/ground_truth.tif"   # annotated slices only, (Z, Y, X) instance labels
PRED_DIR = "/path/to/predictions"       # predicted slices at the same positions, one tif each
OUT_DIR = "/path/to/eval_output"

THRESHOLDS = np.round(np.arange(0.10, 0.91, 0.05), 2)   # IoU thresholds for S_inst
PQ_THRESHOLD = 0.5


def iou(a, b):
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum()) / union if union else 0.0


def dice(a, b):
    total = a.sum() + b.sum()
    return 2.0 * float(np.logical_and(a, b).sum()) / total if total else 0.0


def match(ious, t):
    """one-to-one matching between gt and predictions, only pairs with IoU >= t count"""
    ok = ious >= t
    if not ok.any():
        return []
    rows, cols = linear_sum_assignment(-ious * ok)
    return [(r, c) for r, c in zip(rows, cols) if ious[r, c] >= t]


def load_predictions(pred_dir, gt_shape):
    # skip the "._" sidecar files macOS leaves on external drives
    files = sorted(f for f in Path(pred_dir).glob("*.tif") if not f.name.startswith("._"))
    if not files:
        raise FileNotFoundError(f"no tif files in {pred_dir}")

    h, w = gt_shape[1:]
    slices = []
    for f in files:
        sl = tifffile.imread(f)
        if sl.shape != (h, w):
            # nearest neighbour so the ids survive, cv2 can't resize uint32 so go through int32
            if sl.dtype == np.uint32:
                sl = sl.astype(np.int32)
            sl = cv2.resize(sl, (w, h), interpolation=cv2.INTER_NEAREST)
        slices.append(sl)
    return np.stack(slices)


def collect_instances(gt, pred):
    gt_ids = [i for i in np.unique(gt) if i != 0]

    # predictions that never touch an annotated object are ignored, because the
    # annotation is sparse and they may be real dendrites nobody labelled
    gt_fg = gt > 0
    pred_ids = [i for i in np.unique(pred) if i != 0 and (gt_fg & (pred == i)).any()]

    gt_masks = {g: gt == g for g in gt_ids}
    pred_masks = {p: pred == p for p in pred_ids}

    ious = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float32)
    for i, g in enumerate(gt_ids):
        for j, p in enumerate(pred_ids):
            ious[i, j] = iou(gt_masks[g], pred_masks[p])
    return gt_ids, pred_ids, gt_masks, pred_masks, ious


def threshold_sweep(ious):
    """S = TP / (TP + FP + FN) at every threshold, S_inst is the mean over thresholds"""
    n_gt, n_pred = ious.shape
    rows = []
    for t in THRESHOLDS:
        tp = len(match(ious, t))
        fp, fn = n_pred - tp, n_gt - tp
        rows.append({
            "IoU_threshold": t, "TP": tp, "FP": fp, "FN": fn,
            "S": round(tp / (tp + fp + fn), 4) if tp + fp + fn else 0.0,
            "Precision": round(tp / (tp + fp), 4) if tp + fp else 0.0,
            "Recall": round(tp / (tp + fn), 4) if tp + fn else 0.0,
        })
    return pd.DataFrame(rows)


def match_table(pairs, gt_ids, pred_ids, gt_masks, pred_masks):
    rows = []
    for r, c in pairs:
        g, p = gt_ids[r], pred_ids[c]
        rows.append({
            "GT_id": g, "Pred_id": p,
            "IoU": round(iou(gt_masks[g], pred_masks[p]), 4),
            "Dice": round(dice(gt_masks[g], pred_masks[p]), 4),
            "GT_voxels": int(gt_masks[g].sum()), "Pred_voxels": int(pred_masks[p].sum()),
        })
    return pd.DataFrame(rows)


def panoptic(gt_ids, pred_ids, gt_masks, pred_masks, ious):
    pairs = match(ious, PQ_THRESHOLD)
    tp = len(pairs)
    fn = len(gt_ids) - tp
    fp = len(pred_ids) - tp

    sq = float(np.mean([ious[r, c] for r, c in pairs])) if tp else 0.0
    dq = tp / (tp + 0.5 * fp + 0.5 * fn) if tp + fp + fn else 0.0

    table = match_table(pairs, gt_ids, pred_ids, gt_masks, pred_masks)
    table.insert(2, "Match", "TP")
    matched_gt = {gt_ids[r] for r, _ in pairs}
    matched_pred = {pred_ids[c] for _, c in pairs}

    extra = []
    for g in gt_ids:
        if g not in matched_gt:
            extra.append({"GT_id": g, "Pred_id": None, "Match": "FN", "IoU": 0.0, "Dice": 0.0,
                          "GT_voxels": int(gt_masks[g].sum()), "Pred_voxels": 0})
    for p in pred_ids:
        if p not in matched_pred:
            extra.append({"GT_id": None, "Pred_id": p, "Match": "FP", "IoU": 0.0, "Dice": 0.0,
                          "GT_voxels": 0, "Pred_voxels": int(pred_masks[p].sum())})
    table = pd.concat([table, pd.DataFrame(extra)], ignore_index=True)

    return {"PQ": sq * dq, "SQ": sq, "DQ": dq, "TP": tp, "FP": fp, "FN": fn, "table": table}


def semantic(gt, pred):
    g, p = gt > 0, pred > 0
    tp = int((g & p).sum())
    fp = int((~g & p).sum())
    fn = int((g & ~p).sum())

    def scores(tp, fp, fn):
        return {
            "dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "iou": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
            "prec": tp / (tp + fp) if tp + fp else 0.0,
            "rec": tp / (tp + fn) if tp + fn else 0.0,
        }

    # "full" counts every predicted voxel, including unannotated dendrites.
    # "sparse" only counts prediction inside annotated objects, so precision is 1 by construction.
    return scores(tp, fp, fn), scores(tp, 0, fn)


def main():
    out = Path(OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    gt = tifffile.imread(GT_PATH)
    pred = load_predictions(PRED_DIR, gt.shape)
    print("gt:", gt.shape, gt.dtype, "| pred:", pred.shape, pred.dtype)
    if pred.shape != gt.shape:
        raise ValueError(f"shapes differ: gt {gt.shape} vs pred {pred.shape}")

    gt_ids, pred_ids, gt_masks, pred_masks, ious = collect_instances(gt, pred)

    sweep = threshold_sweep(ious)
    sweep.to_csv(out / "instance_dsb_thresholds.csv", index=False)
    for t, name in [(0.5, "iou05"), (0.75, "iou075")]:
        match_table(match(ious, t), gt_ids, pred_ids, gt_masks, pred_masks) \
            .to_csv(out / f"instance_matches_{name}.csv", index=False)

    pan = panoptic(gt_ids, pred_ids, gt_masks, pred_masks, ious)
    pan["table"].to_csv(out / "panoptic_per_instance.csv", index=False)

    full, sparse = semantic(gt, pred)

    def s_at(t):
        row = sweep[sweep["IoU_threshold"] == t]
        return float(row["S"].iloc[0]) if not row.empty else 0.0

    summary = {
        "GT_path": GT_PATH, "Pred_path": PRED_DIR,
        "Instance_S_mean": round(float(sweep["S"].mean()), 4),
        "Instance_S_at_0.5": round(s_at(0.5), 4),
        "Instance_S_at_0.75": round(s_at(0.75), 4),
        "PQ": round(pan["PQ"], 4), "SQ": round(pan["SQ"], 4), "DQ": round(pan["DQ"], 4),
        "TP_instances": pan["TP"], "FP_instances": pan["FP"], "FN_instances": pan["FN"],
        "GT_instances": len(gt_ids), "Valid_predictions": len(pred_ids),
    }
    for kind, s in [("sparse", sparse), ("full", full)]:
        summary.update({
            f"Semantic_Dice_{kind}": round(s["dice"], 4), f"Semantic_IoU_{kind}": round(s["iou"], 4),
            f"Semantic_Prec_{kind}": round(s["prec"], 4), f"Semantic_Rec_{kind}": round(s["rec"], 4),
        })
    pd.DataFrame([summary]).to_csv(out / "eval_summary.csv", index=False)

    print(f"S_inst {summary['Instance_S_mean']}, PQ {summary['PQ']}")
    print("saved to", out)


if __name__ == "__main__":
    main()
