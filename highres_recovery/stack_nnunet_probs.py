import numpy as np
from pathlib import Path
from tqdm import tqdm

# change these to your own data
NPZ_DIR = "/path/to/nnunet_predictions"       # one .npz per slice, from --save_probabilities
OUTPUT_PATH = "/path/to/nnunet_prob_3D.npy"

FOREGROUND_CHANNEL = 1   # channel 0 is background, channel 1 is dendrite

# slices are stacked in sorted file name order, so the names must sort in z order
files = sorted(Path(NPZ_DIR).glob("*.npz"))
print(f"found {len(files)} files")

# nnU-Net stores the probabilities under the first key, shape (classes, 1, H, W)
with np.load(files[0]) as first:
    key = first.files[0]
    h, w = first[key].shape[-2:]
print("key:", key, "| slice size:", (h, w))

# write straight to disk so the whole stack never has to fit in RAM
prob = np.lib.format.open_memmap(OUTPUT_PATH, mode="w+", dtype=np.float32,
                                 shape=(len(files), h, w))

lo, hi, total = np.inf, -np.inf, 0.0
for z, f in enumerate(tqdm(files, desc="stacking slices")):
    with np.load(f) as data:
        sl = np.squeeze(data[key][FOREGROUND_CHANNEL]).astype(np.float32)
    prob[z] = sl
    lo, hi, total = min(lo, sl.min()), max(hi, sl.max()), total + sl.sum()

prob.flush()
print("shape:", prob.shape)
print(f"min {lo:.4f}, max {hi:.4f}, mean {total / prob.size:.5f}")
print("saved to", OUTPUT_PATH)
