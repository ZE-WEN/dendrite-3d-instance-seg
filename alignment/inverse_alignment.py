import numpy as np
import tifffile as tiff
from scipy.ndimage import shift
from tqdm import tqdm
import imreg_dft as ird

# paths, change these to your own data
L_UNALIGNED = "/path/to/low_res/unaligned.tif"
L_ALIGNED = "/path/to/low_res/aligned.tif"
H_UNALIGNED = "/path/to/high_res/unaligned.tif"
H_ALIGNED_OUT = "/path/to/high_res/aligned.tif"
SHIFTS_OUT = "/path/to/low_res/shifts.npy"

L_un = tiff.imread(L_UNALIGNED)
L_al = tiff.imread(L_ALIGNED)
H_un = tiff.imread(H_UNALIGNED)

assert L_un.shape == L_al.shape, "low-res stacks have different shapes"
assert L_un.shape[0] == H_un.shape[0], "low and high res have different number of slices"

n_slices = L_un.shape[0]
scale = np.array([H_un.shape[1] / L_un.shape[1],
                  H_un.shape[2] / L_un.shape[2]])
print("scale (y, x):", scale)

shifts = np.zeros((n_slices, 2), dtype=np.float32)
H_al = np.zeros_like(H_un)

for i in tqdm(range(n_slices)):
    # shift that takes the unaligned low-res slice to the aligned one
    shifts[i] = ird.translation(L_al[i], L_un[i])["tvec"]

    # same shift, but in high-res pixels
    H_al[i] = shift(H_un[i], shifts[i] * scale, order=1, mode="constant", cval=0)

tiff.imwrite(H_ALIGNED_OUT, H_al.astype(H_un.dtype))
np.save(SHIFTS_OUT, shifts)
print("done")
