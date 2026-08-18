from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

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


# ============================================================
# LOAD RAW FRIDAY DDoS CSV
# ============================================================

print("=" * 72)
print("CICIDS2017 FRIDAY DDoS - RANDOM FOREST DUPLICATE AUDIT")
print("=" * 72)

print(f"\nInput file:\n{DATA_PATH}")

df = pd.read_csv(DATA_PATH)

# CICIDS2017 column names sometimes contain leading/trailing spaces.
df.columns = df.columns.str.strip()

print(f"\nRaw rows:    {len(df):,}")
print(f"Raw columns: {len(df.columns):,}")


# ============================================================
# IDENTIFY LABEL COLUMN
# ============================================================

if "Label" not in df.columns:
    raise ValueError(
        "The expected 'Label' column was not found.\n"
        f"Columns available:\n{list(df.columns)}"
    )

print("\nRaw label distribution:")
print(df["Label"].astype(str).str.strip().value_counts(dropna=False))


# ============================================================
# CLEAN DATA USING SAME GENERAL LOGIC AS RF EXPERIMENT
# ============================================================

df["Label"] = df["Label"].astype(str).str.strip()

# This Friday afternoon file should contain BENIGN and DDoS.
df = df[df["Label"].isin(["BENIGN", "DDoS"])].copy()

# Separate labels from features.
y_text = df["Label"].copy()
X = df.drop(columns=["Label"]).copy()

# Convert all feature columns to numeric.
# Anything that cannot be converted becomes NaN.
X = X.apply(pd.to_numeric, errors="coerce")

# Replace positive/negative infinity with NaN.
X = X.replace([np.inf, -np.inf], np.nan)

rows_before_invalid_removal = len(X)

# Keep only rows with complete finite feature values.
valid_mask = X.notna().all(axis=1)

X = X.loc[valid_mask].reset_index(drop=True)
y_text = y_text.loc[valid_mask].reset_index(drop=True)

rows_removed_invalid = rows_before_invalid_removal - len(X)

# Binary labels:
# BENIGN = 0
# DDoS  = 1
y = y_text.map({
    "BENIGN": 0,
    "DDoS": 1,
})

print("\n" + "=" * 72)
print("CLEANING SUMMARY")
print("=" * 72)

print(f"Rows before invalid-value removal: {rows_before_invalid_removal:,}")
print(f"Rows removed as NaN/inf/invalid:   {rows_removed_invalid:,}")
print(f"Cleaned rows:                      {len(X):,}")
print(f"Feature columns:                   {X.shape[1]:,}")

print("\nCleaned class distribution:")
print(y.value_counts().sort_index().rename(index={0: "BENIGN", 1: "DDoS"}))


# ============================================================
# 1. EXACT DUPLICATES IN THE COMPLETE CLEANED DATASET
# ============================================================

# Duplicate FEATURE vectors are important because two flows
# can be separate CSV rows but contain exactly the same feature values.

duplicated_any_mask = X.duplicated(keep=False)
duplicated_after_first_mask = X.duplicated(keep="first")

rows_in_duplicate_groups = int(duplicated_any_mask.sum())
removable_duplicates = int(duplicated_after_first_mask.sum())
unique_feature_rows = int(len(X) - removable_duplicates)

print("\n" + "=" * 72)
print("1. EXACT DUPLICATES IN CLEANED FEATURE DATA")
print("=" * 72)

print(f"Total cleaned feature rows:       {len(X):,}")
print(f"Rows belonging to duplicate sets:{rows_in_duplicate_groups:>12,}")
print(f"Removable exact duplicates:       {removable_duplicates:>12,}")
print(f"Unique feature rows:              {unique_feature_rows:>12,}")

duplicate_percentage = (
    removable_duplicates / len(X) * 100
    if len(X)
    else 0
)

print(f"Removable duplicate percentage:   {duplicate_percentage:.4f}%")


# ============================================================
# 2. IDENTICAL FEATURES WITH DIFFERENT LABELS
# ============================================================

feature_hashes = pd.util.hash_pandas_object(
    X,
    index=False,
)

hash_and_label = pd.DataFrame({
    "feature_hash": feature_hashes.to_numpy(),
    "label": y.to_numpy(),
})

labels_per_feature = (
    hash_and_label
    .groupby("feature_hash")["label"]
    .nunique()
)

conflicting_hashes = labels_per_feature[
    labels_per_feature > 1
].index

conflicting_pattern_count = len(conflicting_hashes)

conflicting_rows = int(
    hash_and_label["feature_hash"]
    .isin(conflicting_hashes)
    .sum()
)

print("\n" + "=" * 72)
print("2. IDENTICAL FEATURE VECTORS WITH DIFFERENT LABELS")
print("=" * 72)

print(
    "Feature patterns associated with both BENIGN and DDoS: "
    f"{conflicting_pattern_count:,}"
)

print(
    "Rows involved in conflicting-label feature patterns: "
    f"{conflicting_rows:,}"
)


# ============================================================
# 3. RECREATE ORIGINAL STRATIFIED 80/20 RANDOM SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("\n" + "=" * 72)
print("3. STRATIFIED RANDOM 80/20 SPLIT")
print("=" * 72)

print(f"random_state: {RANDOM_STATE}")
print(f"Training rows: {len(X_train):,}")
print(f"Testing rows:  {len(X_test):,}")

print("\nTraining class distribution:")
print(
    y_train.value_counts()
    .sort_index()
    .rename(index={0: "BENIGN", 1: "DDoS"})
)

print("\nTesting class distribution:")
print(
    y_test.value_counts()
    .sort_index()
    .rename(index={0: "BENIGN", 1: "DDoS"})
)


# ============================================================
# 4. EXACT FEATURE OVERLAP BETWEEN TRAIN AND TEST
# ============================================================

train_hashes = pd.util.hash_pandas_object(
    X_train,
    index=False,
)

test_hashes = pd.util.hash_pandas_object(
    X_test,
    index=False,
)

train_hash_set = set(train_hashes.to_numpy())
test_hash_set = set(test_hashes.to_numpy())

shared_feature_hashes = train_hash_set.intersection(
    test_hash_set
)

shared_feature_patterns = len(shared_feature_hashes)

train_rows_also_in_test = int(
    train_hashes.isin(shared_feature_hashes).sum()
)

test_rows_also_in_train = int(
    test_hashes.isin(shared_feature_hashes).sum()
)

test_overlap_percentage = (
    test_rows_also_in_train / len(X_test) * 100
    if len(X_test)
    else 0
)

print("\n" + "=" * 72)
print("4. EXACT FEATURE OVERLAP BETWEEN TRAIN AND TEST")
print("=" * 72)

print(
    "Unique feature patterns appearing in BOTH train and test: "
    f"{shared_feature_patterns:,}"
)

print(
    "Training rows with an identical feature vector in test: "
    f"{train_rows_also_in_test:,}"
)

print(
    "Test rows with an identical feature vector in training: "
    f"{test_rows_also_in_train:,}"
)

print(
    "Percentage of test rows duplicated in training: "
    f"{test_overlap_percentage:.4f}%"
)


# ============================================================
# 5. DEDUPLICATED DATASET SUMMARY
# ============================================================

# IMPORTANT:
# We do NOT save or overwrite the original data here.
# This section only calculates what the dataset would look like
# after removing repeated exact feature vectors.

keep_mask = ~X.duplicated(keep="first")

X_dedup = X.loc[keep_mask].reset_index(drop=True)
y_dedup = y.loc[keep_mask].reset_index(drop=True)

print("\n" + "=" * 72)
print("5. DATASET AFTER EXACT FEATURE DEDUPLICATION")
print("=" * 72)

print(f"Rows before deduplication: {len(X):,}")
print(f"Rows after deduplication:  {len(X_dedup):,}")
print(f"Rows removed:              {len(X) - len(X_dedup):,}")

print("\nClass distribution after deduplication:")
print(
    y_dedup.value_counts()
    .sort_index()
    .rename(index={0: "BENIGN", 1: "DDoS"})
)


# ============================================================
# 6. SAVE AUDIT SUMMARY
# ============================================================

summary = {
    "input_file": str(DATA_PATH),
    "raw_rows": int(len(df) + 0),
    "cleaned_rows": int(len(X)),
    "feature_columns": int(X.shape[1]),
    "invalid_rows_removed": int(rows_removed_invalid),
    "benign_rows": int((y == 0).sum()),
    "ddos_rows": int((y == 1).sum()),
    "random_state": RANDOM_STATE,
    "test_size": TEST_SIZE,
    "training_rows": int(len(X_train)),
    "testing_rows": int(len(X_test)),
    "rows_in_duplicate_groups": rows_in_duplicate_groups,
    "removable_exact_duplicates": removable_duplicates,
    "unique_feature_rows": unique_feature_rows,
    "duplicate_percentage": duplicate_percentage,
    "conflicting_feature_patterns": int(conflicting_pattern_count),
    "conflicting_rows": conflicting_rows,
    "shared_train_test_feature_patterns": int(shared_feature_patterns),
    "train_rows_duplicated_in_test": train_rows_also_in_test,
    "test_rows_duplicated_in_train": test_rows_also_in_train,
    "test_overlap_percentage": test_overlap_percentage,
    "deduplicated_rows": int(len(X_dedup)),
}

json_path = RESULTS_DIR / "rf_ddos_duplicate_audit.json"

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4)

summary_csv_path = RESULTS_DIR / "rf_ddos_duplicate_audit.csv"

pd.DataFrame(
    list(summary.items()),
    columns=["Metric", "Value"],
).to_csv(
    summary_csv_path,
    index=False,
)

print("\n" + "=" * 72)
print("AUDIT COMPLETE")
print("=" * 72)

print(f"\nJSON results saved to:")
print(json_path)

print(f"\nCSV results saved to:")
print(summary_csv_path)

print("\nIMPORTANT VALUES FOR THESIS REVISION:")
print(f"Cleaned rows:                                {len(X):,}")
print(f"Removable exact duplicates:                 {removable_duplicates:,}")
print(f"Shared train/test feature patterns:         {shared_feature_patterns:,}")
print(f"Test rows duplicated in training:           {test_rows_also_in_train:,}")
print(f"Test-set overlap percentage:                {test_overlap_percentage:.4f}%")
print(f"Conflicting-label feature patterns:         {conflicting_pattern_count:,}")