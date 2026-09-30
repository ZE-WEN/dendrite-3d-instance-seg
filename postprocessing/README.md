# Postprocessing

A final 3D cleanup that runs on the refined result from [`highres_recovery`](../highres_recovery). It fixes instances that are broken or split across slices, which is easier to see from the side than from the slices the pipeline was segmented in.

## Why the x view

Segmentation and linking both work slice by slice in z, so their mistakes line up along z. When you turn the volume and look at it along x, a dendrite cut across should look like a round, solid blob that moves only a little from one slice to the next. Pieces that break this pattern are likely errors, and the neighboring slice tells us how to fix them.

## Steps

1. **Downsample.** The volume is shrunk to 1024 by 1024 per slice with nearest neighbor, so the ids are kept.
2. **Repair in the x view.** The volume is turned so that x becomes the slice axis. For every instance on one slice, the script looks for the pieces that continue it on the next slice. If those pieces together are close, overlap enough and have a good shape, they are merged and take the same id. This runs front to back and then back to front. Then the volume is turned back.
3. **Merge and clean.** The original volume is kept, and the repair only fills voxels that were background. Small pieces of an instance are then given to the neighboring instance they touch the most.
4. **Harmonize ids.** A window of 5 slices slides along z. If an instance on the first slice is almost fully covered by one id on a later slice, that id is renamed to match, and empty slices in between are filled.
5. **NOTE:** You will need to upscale for evaluation

## Files

- `run_postprocessing.py` runs the four steps. Paths and settings are at the top.
- `pipeline_utils.py` holds all the functions.

