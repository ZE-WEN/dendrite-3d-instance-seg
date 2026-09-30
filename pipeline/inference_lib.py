"""
Runs SAM on every slice using cached box prompts.

Adapted from the inference code in micro-sam
(https://github.com/computational-cell-analytics/micro-sam), MIT license.
"""
import os
import pickle

import imageio.v3 as imageio
import numpy as np
from tqdm import tqdm
from micro_sam.inference import batched_inference


def run_inference_with_boxes(predictor, image_paths, embedding_dir, prediction_dir,
                             boxes_pkl, batch_size=512):
    with open(boxes_pkl, "rb") as f:
        all_boxes = pickle.load(f)

    os.makedirs(prediction_dir, exist_ok=True)

    for image_path in tqdm(image_paths, desc="SAM"):
        name = os.path.basename(image_path)
        out_path = os.path.join(prediction_dir, name)

        # already done in an earlier run, so a crashed job can just be restarted
        if os.path.exists(out_path):
            continue

        image = imageio.imread(image_path)
        boxes = np.asarray(all_boxes[name])
        embedding_path = os.path.join(embedding_dir, os.path.splitext(name)[0] + ".zarr")

        if len(boxes) == 0:
            labels = np.zeros(image.shape[:2], dtype=np.uint16)
        else:
            labels = batched_inference(
                predictor, image, batch_size,
                boxes=boxes, points=None, point_labels=None,
                multimasking=False, embedding_path=embedding_path,
                return_instance_segmentation=True,
            )

        imageio.imwrite(out_path, np.ascontiguousarray(labels).astype(np.uint16), compression=5)
