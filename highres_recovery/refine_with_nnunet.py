import numpy as np
import tifffile as tiff
from scipy import ndimage as ndi
from tqdm import tqdm

# change these to your own data
INSTANCES_PATH = "/path/to/upscaled.tif"       # high-res instance labels
PROB_PATH = "/path/to/nnunet_prob_3D.npy"      # nnU-Net dendrite probability, same shape
OUTPUT_PATH = "/path/to/refined.tif"

PROB_THRESHOLD = 0.98

inst_vol = tiff.imread(INSTANCES_PATH)
prob_vol = np.load(PROB_PATH, mmap_mode="r")   # stays on disk, we only read the crops
assert inst_vol.shape == prob_vol.shape, "instances and probabilities have different shapes"

# bounding box of every instance in one pass, boxes[i] belongs to label i + 1
boxes = ndi.find_objects(inst_vol)
print(f"found {sum(b is not None for b in boxes)} instances")

out = np.zeros(inst_vol.shape, dtype=np.uint32)

for inst_id, box in enumerate(tqdm(boxes, desc="refining instances"), start=1):
    if box is None:
        continue

    inst_roi = inst_vol[box] == inst_id
    prob_roi = np.asarray(prob_vol[box])
    refined = inst_roi.copy()

    for z in range(inst_roi.shape[0]):
        if not inst_roi[z].any():
            continue

        # confident nnU-Net regions on this slice
        comps, n = ndi.label(prob_roi[z] > PROB_THRESHOLD)
        if n == 0:
            continue

        # keep the ones that touch the instance and add them to it
        touching = np.unique(comps[inst_roi[z]])
        touching = touching[touching != 0]
        refined[z] |= np.isin(comps, touching)

    out[box][refined] = inst_id

print("saving...")
tiff.imwrite(OUTPUT_PATH, out, bigtiff=True,
             compression="zlib", compressionargs={"level": 6})
print("saved to", OUTPUT_PATH)
