# nnU-Net

The high-resolution refinement model is trained with [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet). I used the standard training pipeline from their repository and did not change the code.

## Data

The images and masks are prepared as 2D TIF slices. `dataset.json` is in this folder, so you can use it as a reference for how the dataset is set up. For the folder layout and file naming, please follow the nnU-Net documentation.

## Training

I first ran preprocessing and planning.

```bash
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```

Then I trained for 100 epochs, using the built-in `nnUNetTrainer_100epochs` trainer.

```bash
nnUNetv2_train DATASET_ID 2d FOLD -tr nnUNetTrainer_100epochs
```

Replace `DATASET_ID` and `FOLD` with your own values. The default training length in nnU-Net is 1000 epochs, so the trainer flag is what sets it to 100.

## Citation

Please cite nnU-Net if you use this.

```bibtex
@article{isensee2021nnu,
  title={nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation},
  author={Isensee, Fabian and Jaeger, Paul F and Kohl, Simon AA and Petersen, Jens and Maier-Hein, Klaus H},
  journal={Nature Methods},
  volume={18},
  number={2},
  pages={203--211},
  year={2021}
}
```
Please cite our paper if you use our trained weights
TBD
