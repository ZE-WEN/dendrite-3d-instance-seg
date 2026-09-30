import numpy as np
import tifffile as tiff
from skimage.measure import label as cc_label

# change these to your own data
INPUT_PATH = "/path/to/linked.tif"
OUTPUT_PATH = "/path/to/linked_filtered.tif"

MIN_VOXELS = 90000   # objects smaller than this are removed, you need to adjust according to your data

# True keeps the ids from the linking step.
# False treats the volume as binary and relabels it with 3D connected components.
KEEP_IDS = True
CONNECTIVITY = 1     # only used when KEEP_IDS is False (1 = 6, 2 = 18, 3 = 26 neighbours)

vol = tiff.imread(INPUT_PATH)
assert vol.ndim == 3, f"expected a (Z, Y, X) volume, got {vol.shape}"

if KEEP_IDS:
    ids, sizes = np.unique(vol, return_counts=True)
    big = ids[(ids != 0) & (sizes >= MIN_VOXELS)]
    out = np.where(np.isin(vol, big), vol, 0).astype(vol.dtype)
else:
    lab = cc_label(vol > 0, connectivity=CONNECTIVITY)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    big = np.flatnonzero(sizes >= MIN_VOXELS)
    out = np.where(np.isin(lab, big), lab, 0).astype(np.uint32)

tiff.imwrite(OUTPUT_PATH, out)

before = np.count_nonzero(vol)
after = np.count_nonzero(out)
print(f"kept {len(big)} objects")
print(f"foreground voxels: {after} of {before} ({100 * after / max(before, 1):.1f}%)")
print("saved to", OUTPUT_PATH)
