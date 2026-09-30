# RF linker

A random forest that decides whether two 2D instances on neighboring slices belong to the same 3D instance. The pipeline uses it to link segmentations across slices. The trained model is available for [download](https://zenodo.org/records/23054050).

## Files

- `extract_features.py` builds training data from a 3D ground truth instance volume.
- `rf_training.py` trains the random forest on that data.
- `train_edges.csv` is the edge table I used for training.
- `run_linking.py` links a segmentation with a trained model.
- `infer/` holds the linking code that `run_linking.py` and the main pipeline both use.

## How it works

`extract_features.py` splits every instance into its 2D connected components on each slice. For each pair of overlapping components on neighboring slices it writes one row with these features.

`overlap_px`, `IoU`, `area_src`, `area_tgt`, `area_ratio`, `dy`, `dx`, `d_centroid`

`label_same` is 1 if both components come from the same ground truth instance and 0 otherwise. That is what the forest learns to predict. 

At inference time the same edges are built from your segmentation and the forest gives each one a link probability. Every edge above the threshold is kept and connected components are merged into tracks. There is no one-to-one rule, so branching is allowed.
