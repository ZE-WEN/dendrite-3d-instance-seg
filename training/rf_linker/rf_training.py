import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

# change these to your own data
EDGES_CSV = "/path/to/train_edges.csv"
MODEL_OUT = "/path/to/weights/rf_edge_classifier.pkl"

FEATURES = [
    "overlap_px", "IoU", "area_src", "area_tgt",
    "area_ratio", "dy", "dx", "d_centroid",
]

# set to True to keep whole slices together in the split (less leakage)
SPLIT_BY_SLICE = False

df = pd.read_csv(EDGES_CSV)
X = df[FEATURES].values
y = df["label_same"].values.astype(int)

if SPLIT_BY_SLICE:
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(X, y, groups=df["z"].values))
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
else:
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

rf = RandomForestClassifier(
    n_estimators=400,
    min_samples_leaf=2,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
)
rf.fit(X_train, y_train)

pred = rf.predict(X_val)
prob = rf.predict_proba(X_val)[:, 1]
print(classification_report(y_val, pred))

try:
    print("ROC AUC:", roc_auc_score(y_val, prob))
    print("PR AUC: ", average_precision_score(y_val, prob))
except ValueError:
    pass  # happens if the validation set only has one class

print("\nfeature importances")
print(pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False))

joblib.dump(rf, MODEL_OUT)
print("\nsaved to", MODEL_OUT)
