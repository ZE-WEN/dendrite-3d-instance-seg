# Pipeline

This folder runs the iterative 3D dendrite segmentation on a stack of 2D slices. It combines the YOLOv6 detector, the fine-tuned SAM and the random forest linker.

## Overview

1. **Detect.** Before you run the pipeline, run the trained YOLOv6 model on your 2D image sequence yourself. This gives one txt file of bounding boxes per slice, and `main.py` reads those files as its starting point. See [`training/yolov6`](../training/yolov6) for the model and how it was trained.
2. **Segment.** SAM is prompted with the boxes and produces an instance mask for every slice.
3. **Clean.** Each slice is cleaned in 2D. Small pieces are dropped, holes are filled, and neighboring pieces that carry different labels but really belong together are merged.
4. **Recover missing objects.** Objects that show up in the neighboring slices but are missing in a slice get new boxes. Those boxes are added to the prompts for the next iteration.
5. **Repeat** steps 2 to 4. I use three iterations, and boxes are only updated after the first two.
6. **Link.** After the last iteration, the random forest links the 2D instances across slices into 3D instances, see [`training/rf_linker`](../training/rf_linker).
7. **Filter.** Very small 3D objects are removed from the linked volume. This is a separate script that you run after `main.py`.
   
`main.py` runs steps 2 to 6. The nnU-Net refinement is trained separately (see [`training/nnunet`](../training/nnunet)) and is not part of `main.py`.

## Files

- `main.py` is the pipeline itself. Paths and parameters are set at the top.
- `functions.py` holds the helpers for boxes, reading slices and the 2D cleanup.
- `inference_lib.py` runs SAM with box prompts.
- `filter_small_objects.py` removes 3D objects below a voxel count from the linked volume. Set the paths and `MIN_VOXELS` at the top of the script. The right threshold depends on your resolution, so adjust it for your data. 

## Acknowledgements

`inference_lib.py` is adapted from [micro-sam](https://github.com/computational-cell-analytics/micro-sam). Please cite it if you use this code.
```bibtex
@article{archit2025segment,
  title={Segment anything for microscopy},
  author={Archit, Anwai and Freckmann, Luca and Nair, Sushmita and Khalid, Nabeel and Hilt, Paul and Rajashekar, Vikas and Freitag, Marei and Teuber, Carolin and Spitzner, Melanie and Tapia Contreras, Constanza and others},
  journal={Nature methods},
  volume={22},
  number={3},
  pages={579--591},
  year={2025},
  publisher={Nature Publishing Group US New York}
}
```
