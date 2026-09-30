import numpy as np
import tifffile as tiff
from scipy import ndimage as ndi
from skimage.transform import resize
from tqdm import tqdm

# change these to your own data
LOW_RES_MASK = "/path/to/linked_filtered.tif"      # instance labels at low resolution
HIGH_RES_IMAGE = "/path/to/high_res/aligned.tif"   # only used to read the target shape
OUTPUT_PATH = "/path/to/upscaled.tif"

low = tiff.imread(LOW_RES_MASK)
assert low.ndim == 3, f"expected a (Z, Y, X) volume, got {low.shape}"

with tiff.TiffFile(HIGH_RES_IMAGE) as f:
    high_shape = np.array(f.series[0].shape)
low_shape = np.array(low.shape)

# scale per axis, z is usually 1 since both stacks have the same slices
scale = high_shape / low_shape
print("scale (z, y, x):", scale)

out = np.zeros(high_shape, dtype=np.uint32)

# bounding box of every label in one pass, boxes[i] belongs to label i + 1
boxes = ndi.find_objects(low)

for label, box in enumerate(tqdm(boxes, desc="upscaling instances"), start=1):
    if box is None:
        continue

    start = np.array([s.start for s in box])
    stop = np.array([s.stop for s in box])

    # same box in high-res coordinates
    h_start = (start * scale).astype(int)
    h_stop = np.minimum((stop * scale).astype(int), high_shape)
    roi_shape = tuple(h_stop - h_start)
    if any(d <= 1 for d in roi_shape):
        continue

    small = low[box] == label
    big = resize(small.astype(np.float32), roi_shape, order=0,
                 anti_aliasing=False, preserve_range=True).astype(bool)

    z0, y0, x0 = h_start
    out[z0:z0 + roi_shape[0], y0:y0 + roi_shape[1], x0:x0 + roi_shape[2]][big] = label

print("saving...")
tiff.imwrite(OUTPUT_PATH, out, bigtiff=True,
             compression="zlib", compressionargs={"level": 6})
print("saved to", OUTPUT_PATH)
