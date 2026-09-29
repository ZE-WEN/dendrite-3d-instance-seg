## Acknowledgements
The slice alignment step uses the StackReg plugin from the Biomedical Imaging Group at EPFL (https://bigwww.epfl.ch/thevenaz/stackreg/). If you use this it, please also cite the original registration method.

```bibtex
@article{thevenaz1998pyramid,
  author  = {Thevenaz, P. and Ruttimann, U.E. and Unser, M.},
  title   = {A Pyramid Approach to Subpixel Registration Based on Intensity},
  journal = {IEEE Transactions on Image Processing},
  volume  = {7},
  number  = {1},
  pages   = {27--41},
  year    = {1998},
  doi     = {10.1109/83.650848}
}
```

## Inverse alignment
`inverse_alignment.py` takes the alignment done on a low-res stack and applies it to the high-res version of the same volume.

### Why this exists
The pipeline uses the same volume at two resolutions, so both stacks have to sit in exactly the same space. I aligned the low-res stack with StackReg first. Running StackReg again on the high-res data would give slightly different shifts, and then the two stacks would no longer match. So instead I measure the shift each low-res slice went through, scale it up by the resolution ratio, and apply it to the matching high-res slice.

### What it does
For every slice, `imreg_dft` finds the translation between the unaligned and aligned low-res image. That shift gets multiplied by the ratio between the two resolutions and used to move the high-res slice. The low-res shifts are also saved as a `.npy` file in case you want to look at them or reuse them.

You need to change the paths in the script


L_UNALIGNED = "/path/to/low_res/unaligned.tif"
L_ALIGNED = "/path/to/low_res/aligned.tif"
H_UNALIGNED = "/path/to/high_res/unaligned.tif"
H_ALIGNED_OUT = "/path/to/high_res/aligned.tif"
SHIFTS_OUT = "/path/to/low_res/shifts.npy"

