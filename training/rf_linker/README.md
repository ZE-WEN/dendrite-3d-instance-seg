
# RF linker

A random forest that decides whether two 2D instances on neighboring slices belong to the same 3D instance. The pipeline uses it to link segmentations across slices.

## Files

- `extract_features.py` builds training data from a 3D ground truth instance volume.
- `rf_training.py` trains the random forest on that data.
- `train_edges.csv` is the edge table I used for training.
