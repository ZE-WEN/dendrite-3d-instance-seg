import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
import joblib

# 1) Load CSV
df = pd.read_csv(r"")

feature_cols = [
    "overlap_px", "IoU", "area_src", "area_tgt",
    "area_ratio", "dy", "dx", "d_centroid"
]
X = df[feature_cols].values
y = df["label_same"].values.astype(int)

# Optional: choose a grouping to reduce leakage (pick ONE)

groups = None

# 3) Train/val split
if groups is None:
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
else:
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(X, y, groups))
    X_train, X_val, y_train, y_val = X[train_idx], X[val_idx], y[train_idx], y[val_idx]

# 4) Train RF
rf = RandomForestClassifier(
    n_estimators=400,          # a bit larger is cheap & stable
    max_depth=None,
    min_samples_leaf=2,        # mild regularization
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train, y_train)

# 5) Validation metrics
y_pred = rf.predict(X_val)
y_prob = rf.predict_proba(X_val)[:, 1]
print(classification_report(y_val, y_pred))
try:
    print("ROC AUC:", roc_auc_score(y_val, y_prob))
    print("PR AUC :", average_precision_score(y_val, y_prob))
except ValueError:
    pass 

# (Optional) quick feature importance sanity check
imp = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nFeature importances:\n", imp)

# 6) Save
joblib.dump(rf, r"")
