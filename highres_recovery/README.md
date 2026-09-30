# High-res recovery

The pipeline segments and links at low resolution, which is fast but loses fine detail. This folder brings the result back to full resolution and lets nnU-Net fill in the detail. The instance ids and the 3D identity come from the low-res linking. nnU-Net only adds pixels to them and never removes any.

## Steps

1. Run nnU-Net on the full-res slices and save the probabilities.
2. `stack_nnunet_probs.py` stacks those probabilities into one 3D array.
3. `upscale_instances.py` brings the linked low-res instances up to full resolution.
4. `refine_with_nnunet.py` grows each instance into the confident nnU-Net regions that touch it.

## Files

- `stack_nnunet_probs.py` turns the per-slice `.npz` files from nnU-Net into one `(Z, Y, X)` float32 `.npy` file with the dendrite probability.
- `upscale_instances.py` upscales the instance labels to the high-res shape with nearest neighbor, one instance at a time.
- `refine_with_nnunet.py` does the refinement and saves the final instance volume.
