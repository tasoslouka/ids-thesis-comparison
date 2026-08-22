from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "MachineLearningCSV"
    / "MachineLearningCVE"
    / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
)

RESULTS_DIR = PROJECT_ROOT / "results" / "tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
PRECISIONS = [4, 3, 2, 1]  # standardised-space decimal precision


def metric_dict(y_true, y_pred):
    if len(y_true) == 0:
        return {
            "rows": 0,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "fpr": None,
            "false_positives": None,
            "false_negatives": None,
        }

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    return {
        "rows": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "fpr": float(fpr),
        "false_positives": int(fp),
        "false_negatives": int(fn),
    }


def quantised_hashes(arr, decimals):
    q = np.round(arr, decimals=decimals)
    # Avoid treating -0.0 and +0.0 as different binary values.
    q[q == 0] = 0.0
    qdf = pd.DataFrame(q)
    return pd.util.hash_pandas_object(qdf, index=False).to_numpy()


print("=" * 78)
print("CICIDS2017 FRIDAY DDoS - NEAR-DUPLICATE SENSITIVITY AUDIT")
print("=" * 78)

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

if "Label" not in df.columns:
    raise ValueError("Expected 'Label' column not found.")

df["Label"] = df["Label"].astype(str).str.strip()
df = df[df["Label"].isin(["BENIGN", "DDoS"])].copy()

y_text = df["Label"].copy()
X = df.drop(columns=["Label"]).copy()
X = X.apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)

valid_mask = X.notna().all(axis=1)
X = X.loc[valid_mask].reset_index(drop=True)
y_text = y_text.loc[valid_mask].reset_index(drop=True)
y = y_text.map({"BENIGN": 0, "DDoS": 1}).astype(int)

print(f"Cleaned rows: {len(X):,}")
print(f"Features:     {X.shape[1]:,}")

# Remove exact duplicate feature vectors BEFORE the split.
# This ensures the near-duplicate audit is not merely rediscovering exact duplicates.
keep_mask = ~X.duplicated(keep="first")
X_dedup = X.loc[keep_mask].reset_index(drop=True)
y_dedup = y.loc[keep_mask].reset_index(drop=True)

print(f"Rows after exact deduplication: {len(X_dedup):,}")

X_train, X_test, y_train, y_test = train_test_split(
    X_dedup,
    y_dedup,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_dedup,
)

print(f"Train rows: {len(X_train):,}")
print(f"Test rows:  {len(X_test):,}")

# Verify exact overlap really is zero.
train_exact = set(pd.util.hash_pandas_object(X_train, index=False).to_numpy())
test_exact_hashes = pd.util.hash_pandas_object(X_test, index=False).to_numpy()
exact_overlap_mask = np.isin(test_exact_hashes, list(train_exact))
exact_overlap_count = int(exact_overlap_mask.sum())

print(f"Exact train-test overlap after deduplication: {exact_overlap_count:,}")

if exact_overlap_count != 0:
    raise RuntimeError(
        "Exact overlap was expected to be zero after deduplication. "
        "Stop and inspect the data before interpreting near-duplicate results."
    )

# Train the same baseline RF used in the thesis.
rf = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

baseline_metrics = metric_dict(y_test.to_numpy(), y_pred)

print("\nBaseline metrics on deduplicated split:")
for k, v in baseline_metrics.items():
    print(f"  {k}: {v}")

# Standardisation is used ONLY to define a scale-normalised similarity audit.
# It is not used to train the Random Forest.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
X_test_scaled = scaler.transform(X_test).astype(np.float32)

rows = []

for decimals in PRECISIONS:
    train_hashes = quantised_hashes(X_train_scaled, decimals)
    test_hashes = quantised_hashes(X_test_scaled, decimals)

    train_hash_set = set(train_hashes.tolist())
    overlap_mask = np.isin(test_hashes, list(train_hash_set))

    overlap_count = int(overlap_mask.sum())
    overlap_pct = float(overlap_count / len(X_test) * 100)

    # Build training-label sets for each quantised bucket.
    train_bucket_df = pd.DataFrame({
        "qhash": train_hashes,
        "label": y_train.to_numpy(),
    })
    label_sets = (
        train_bucket_df
        .groupby("qhash")["label"]
        .agg(lambda s: frozenset(int(x) for x in s))
        .to_dict()
    )

    same_label_available = 0
    opposite_label_available = 0
    mixed_label_bucket = 0

    y_test_np = y_test.to_numpy()
    for h, label, is_overlap in zip(test_hashes, y_test_np, overlap_mask):
        if not is_overlap:
            continue
        labels = label_sets.get(int(h), frozenset())
        if int(label) in labels:
            same_label_available += 1
        if any(x != int(label) for x in labels):
            opposite_label_available += 1
        if len(labels) > 1:
            mixed_label_bucket += 1

    overlap_metrics = metric_dict(
        y_test_np[overlap_mask],
        y_pred[overlap_mask],
    )
    non_overlap_metrics = metric_dict(
        y_test_np[~overlap_mask],
        y_pred[~overlap_mask],
    )

    row = {
        "standardised_decimals": decimals,
        "approx_standardised_bin_width": 10 ** (-decimals),
        "test_rows": int(len(X_test)),
        "near_overlap_test_rows": overlap_count,
        "near_overlap_test_percentage": overlap_pct,
        "same_label_available_rows": same_label_available,
        "opposite_label_available_rows": opposite_label_available,
        "mixed_label_bucket_rows": mixed_label_bucket,
        "overlap_accuracy": overlap_metrics["accuracy"],
        "overlap_f1": overlap_metrics["f1"],
        "overlap_false_positives": overlap_metrics["false_positives"],
        "overlap_false_negatives": overlap_metrics["false_negatives"],
        "non_overlap_rows": non_overlap_metrics["rows"],
        "non_overlap_accuracy": non_overlap_metrics["accuracy"],
        "non_overlap_f1": non_overlap_metrics["f1"],
        "non_overlap_false_positives": non_overlap_metrics["false_positives"],
        "non_overlap_false_negatives": non_overlap_metrics["false_negatives"],
    }
    rows.append(row)

    print("\n" + "-" * 78)
    print(f"Precision: {decimals} decimals in standardised feature space")
    print(f"Near-overlap test rows: {overlap_count:,} ({overlap_pct:.6f}%)")
    print(f"Same-label training bucket available: {same_label_available:,}")
    print(f"Opposite-label training bucket available: {opposite_label_available:,}")
    print(f"Mixed-label training buckets: {mixed_label_bucket:,}")
    print(f"Non-overlap test rows: {non_overlap_metrics['rows']:,}")
    print(f"Non-overlap accuracy: {non_overlap_metrics['accuracy']}")
    print(f"Non-overlap F1: {non_overlap_metrics['f1']}")
    print(f"Non-overlap FP/FN: {non_overlap_metrics['false_positives']}/"
          f"{non_overlap_metrics['false_negatives']}")

results_df = pd.DataFrame(rows)

csv_path = RESULTS_DIR / "rf_ddos_near_duplicate_sensitivity.csv"
results_df.to_csv(csv_path, index=False)

json_path = RESULTS_DIR / "rf_ddos_near_duplicate_summary.json"
summary = {
    "method": (
        "Exact duplicate feature vectors were removed before splitting. "
        "StandardScaler was fitted on the training partition only. "
        "Train and test rows were then quantised in standardised feature space "
        "at several decimal precisions. A test row was counted as a near-overlap "
        "when its quantised 78-feature vector matched at least one training row."
    ),
    "important_interpretation_note": (
        "This is a quantised near-duplicate sensitivity audit, not proof that "
        "matched rows belong to the same original network session."
    ),
    "random_state": RANDOM_STATE,
    "test_size": TEST_SIZE,
    "cleaned_rows": int(len(X)),
    "deduplicated_rows": int(len(X_dedup)),
    "training_rows": int(len(X_train)),
    "testing_rows": int(len(X_test)),
    "exact_overlap_after_deduplication": exact_overlap_count,
    "baseline_metrics": baseline_metrics,
    "precisions_tested": PRECISIONS,
}
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4)

print("\n" + "=" * 78)
print("NEAR-DUPLICATE AUDIT COMPLETE")
print("=" * 78)
print(f"CSV:  {csv_path}")
print(f"JSON: {json_path}")
print(
    "\nInterpretation: use the multi-precision results as a sensitivity analysis. "
    "Do not call every quantised match a proven duplicate network session."
)
