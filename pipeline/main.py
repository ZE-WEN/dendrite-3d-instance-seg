import gc
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import tifffile as tiff
import torch
from scipy import ndimage
from micro_sam.util import get_sam_model

from functions import (
    add_boxes,
    find_missing_boxes,
    harmonize_entangled_components_2d,
    rc_to_xy,
    read_volume,
    yolo_to_corners,
)
from inference_lib import run_inference_with_boxes

# ---------------------------------------------------------------
# change these to your own data
# ---------------------------------------------------------------
SEQUENCE_DIR = Path("/path/to/image_sequence")          # 2D slices as .png
YOLO_LABELS_DIR = Path("/path/to/yolo/labels")          # .txt files from the YOLOv6 run
EMBEDDINGS_DIR = Path("/path/to/sam_embeddings")        # SAM embeddings are cached here
SAM_CHECKPOINT = Path("/path/to/finetuned_sam_vit_l.pth")
RF_MODEL = Path("/path/to/rf_edge_classifier.pkl")
OUTPUT_DIR = Path("/path/to/output")

# folder that contains infer/, no need to change it if you keep the repo layout
RF_LINKER_DIR = Path(__file__).resolve().parent.parent / "training" / "rf_linker"

IMAGE_SIZE = 1024            # images are assumed to be square
NUM_ITERATIONS = 3
UPDATE_BOXES_AFTER = [1, 2]  # iterations that add missing boxes for the next one
MIN_AREA = 500               # smallest piece of a mask we keep (pixels)
HARMONIZE_RADIUS = 2
HARMONIZE_MIN_SCORE = 0.45
MIN_OVERLAP_PX = 8           # RF linking
LINK_THRESHOLD = 0.6         # RF linking probability
# ---------------------------------------------------------------

sys.path.insert(0, str(RF_LINKER_DIR))
from infer.infer import run_full_inference  # noqa: E402


def yolo_labels_to_pickle(labels_dir, out_pkl):
    """Turn the YOLO txt files into one pickle {image_name: [[x1, y1, x2, y2], ...]}."""
    boxes = {}
    for txt in sorted(labels_dir.glob("*.txt")):
        found = []
        for line in txt.read_text().splitlines():
            parts = line.split()
            if len(parts) < 6:  # expects class, x, y, w, h, conf
                continue
            x, y, w, h = map(float, parts[1:5])
            found.append(yolo_to_corners(x, y, w, h, IMAGE_SIZE))
        boxes[txt.with_suffix(".png").name] = np.array(found, dtype=np.int32)

    with open(out_pkl, "wb") as f:
        pickle.dump(boxes, f)
    print(f"saved boxes for {len(boxes)} images to {out_pkl}")


def clean_slices(iter_dir):
    """Per slice: keep big pieces, fill holes, then merge entangled pieces. Saves a tif."""
    vol = read_volume(str(iter_dir))
    four_conn = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    eight_conn = np.ones((3, 3), dtype=bool)
    out_vol = np.zeros_like(vol)

    for z in range(vol.shape[0]):
        sl = vol[z]
        cleaned = np.zeros_like(sl)

        for obj_id in np.unique(sl):
            if obj_id == 0:
                continue
            cc, n = ndimage.label(sl == obj_id, structure=four_conn)
            if n == 0:
                continue
            sizes = np.bincount(cc.ravel())
            big = np.flatnonzero(sizes >= MIN_AREA)
            big = big[big != 0]
            if big.size == 0:
                continue
            filled = ndimage.binary_fill_holes(np.isin(cc, big), structure=eight_conn)
            cleaned[filled] = obj_id

        out_vol[z] = harmonize_entangled_components_2d(
            cleaned, r=HARMONIZE_RADIUS, min_score=HARMONIZE_MIN_SCORE
        )

    tiff.imwrite(str(iter_dir / "postprocessed_2d.tif"), out_vol.astype(np.uint16))
    return out_vol


def update_boxes(volume, boxes_pkl):
    """Look at 4 neighbouring slices and add boxes for objects that vanished in slice 2."""
    with open(boxes_pkl, "rb") as f:
        boxes = pickle.load(f)

    if len(boxes) != volume.shape[0]:
        print("warning: number of boxes entries and slices differ, continuing anyway")

    names = list(boxes.keys())
    for i in range(volume.shape[0] - 3):
        m1, m2, m3, m4 = (volume[i + k] > 0 for k in range(4))
        new_boxes, _ = find_missing_boxes(m1, m2, m3, m4, min_area=MIN_AREA)
        if len(new_boxes) > 0:
            add_boxes(boxes, names[i + 1], [rc_to_xy(b) for b in new_boxes])

    with open(boxes_pkl, "wb") as f:
        pickle.dump(boxes, f)


def link_tracks(iter_dir):
    run_full_inference(
        proposal_tif_path=str(iter_dir / "postprocessed_2d.tif"),
        rf_model_path=str(RF_MODEL),
        out_edges_csv=str(iter_dir / "edges.csv"),
        out_tracks_tif=str(iter_dir / "linked.tif"),
        min_overlap_px=MIN_OVERLAP_PX,
        connectivity=1,
        use_skip_links=False,
        p_thresh=LINK_THRESHOLD,
    )


def main():
    # iteration 1 starts from the YOLO detections
    first_dir = OUTPUT_DIR / "segmentation_infer_iter1"
    first_dir.mkdir(parents=True, exist_ok=True)
    yolo_labels_to_pickle(YOLO_LABELS_DIR, first_dir / "boxes.pkl")

    predictor = get_sam_model(model_type="vit_l", checkpoint_path=str(SAM_CHECKPOINT))
    image_paths = [str(p) for p in sorted(SEQUENCE_DIR.glob("*.png"))]

    for i in range(1, NUM_ITERATIONS + 1):
        print(f"\n=== iteration {i} ===")
        iter_dir = OUTPUT_DIR / f"segmentation_infer_iter{i}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        boxes_pkl = iter_dir / "boxes.pkl"

        run_inference_with_boxes(
            predictor,
            image_paths,
            embedding_dir=str(EMBEDDINGS_DIR),
            prediction_dir=str(iter_dir),
            boxes_pkl=str(boxes_pkl),
        )

        volume = clean_slices(iter_dir)

        if i in UPDATE_BOXES_AFTER:
            update_boxes(volume, boxes_pkl)

        if i < NUM_ITERATIONS:
            next_dir = OUTPUT_DIR / f"segmentation_infer_iter{i + 1}"
            next_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(boxes_pkl, next_dir / "boxes.pkl")
        else:
            print("linking slices with the random forest")
            link_tracks(iter_dir)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\ndone")


if __name__ == "__main__":
    main()
