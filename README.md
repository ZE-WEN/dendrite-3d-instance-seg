# 3D dendrite instance segmentation in SBF-SEM

A fully automatic pipeline that segments individual dendrites in 3D from serial block-face electron microscopy (SBF-SEM) stacks. A YOLOv6 detector finds dendrites on every slice. 
![Overview of the pipeline](fig1.png)

1. **Align the stack.** Align the low-resolution stack with the StackReg plugin in Fiji, then use `alignment/inverse_alignment.py` to apply the same shifts to the high-resolution stack. Both resolutions are used later, so they have to stay in the same space.
2. **Detect dendrites.** Run the trained YOLOv6 model on the low-res slices and save the boxes as txt files. See `training/yolov6` and `pipeline/README.md`.
3. **Segment and link.** Run `pipeline/main.py`. It prompts SAM with the boxes, cleans the masks, recovers missed objects over three iterations and links everything into 3D with the random forest.
4. **Remove tiny objects.** Run `pipeline/filter_small_objects.py` on the linked volume.
5. **Recover detail at full resolution.** Run nnU-Net on the high-res slices, then the three scripts in `highres_recovery/`.
6. **Clean up in 3D.** Run `postprocessing/run_postprocessing.py`. 
7. **Evaluate.** If you have annotated slices, run `evaluation/evaluate.py`.
