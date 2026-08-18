from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


DATA_PATH = Path("data/raw/MachineLearningCSV/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
OUT_DIR = Path("results/tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Loading: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

if "Label" not in df.columns:
    raise ValueError("Label column not found.")

df["Binary_Label"] = (df["Label"].astype(str).str.strip() != "BENIGN").astype(int)

drop_cols = [
    "Flow ID",
    "Source IP",
    "Source Port",
    "Destination IP",
    "Timestamp",
    "Label",
    "Binary_Label",
]

feature_cols = [c for c in df.columns if c not in drop_cols]

X = df[feature_cols].copy()
y = df["Binary_Label"].copy()

X = X.apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)

valid_mask = X.notna().all(axis=1)
X = X.loc[valid_mask].copy()
y = y.loc[valid_mask].copy()
df_clean = df.loc[valid_mask].copy()

print(f"Cleaned rows: {len(df_clean)}")
print("Label distribution:")
print(y.value_counts().to_string())

# Keep indices so we can map predictions back to original rows
train_idx, test_idx = train_test_split(
    X.index,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train = X.loc[train_idx]
X_test = X.loc[test_idx]
y_train = y.loc[train_idx]
y_test = y.loc[test_idx]

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

print("Training Random Forest...")
model.fit(X_train, y_train)

print("Predicting...")
y_pred = model.predict(X_test)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

print("\nConfusion matrix:")
print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

results = pd.DataFrame({
    "original_index": X_test.index,
    "true_label": y_test.values,
    "predicted_label": y_pred
}, index=X_test.index)

results["error_type"] = "correct"
results.loc[(results["true_label"] == 0) & (results["predicted_label"] == 1), "error_type"] = "false_positive"
results.loc[(results["true_label"] == 1) & (results["predicted_label"] == 0), "error_type"] = "false_negative"

# Add useful original columns if present
useful_original_cols = [
    "Flow ID",
    "Source IP",
    "Source Port",
    "Destination IP",
    "Destination Port",
    "Protocol",
    "Timestamp",
    "Label"
]

available_original_cols = [c for c in useful_original_cols if c in df_clean.columns]

misclassified_idx = results[results["error_type"] != "correct"].index

misclassified = pd.concat(
    [
        results.loc[misclassified_idx],
        df_clean.loc[misclassified_idx, available_original_cols],
        X.loc[misclassified_idx]
    ],
    axis=1
)

false_positives = misclassified[misclassified["error_type"] == "false_positive"]
false_negatives = misclassified[misclassified["error_type"] == "false_negative"]

# Key features for interpretation
key_features = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Min",
    "Bwd Packet Length Max",
    "Bwd Packet Length Mean",
    "Max Packet Length",
    "Min Packet Length",
    "Average Packet Size",
    "Packet Length Std",
    "Packet Length Variance",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward"
]

key_features = [c for c in key_features if c in X.columns]

summary_rows = []

groups = {
    "correct_benign": results[(results["true_label"] == 0) & (results["predicted_label"] == 0)].index,
    "false_positive": results[results["error_type"] == "false_positive"].index,
    "false_negative": results[results["error_type"] == "false_negative"].index,
    "correct_attack": results[(results["true_label"] == 1) & (results["predicted_label"] == 1)].index,
}

for group_name, idx in groups.items():
    row = {
        "group": group_name,
        "rows": len(idx)
    }

    if len(idx) > 0:
        for feature in key_features:
            row[f"{feature}_mean"] = X.loc[idx, feature].mean()
            row[f"{feature}_median"] = X.loc[idx, feature].median()
    else:
        for feature in key_features:
            row[f"{feature}_mean"] = np.nan
            row[f"{feature}_median"] = np.nan

    summary_rows.append(row)

summary = pd.DataFrame(summary_rows)

out_misclassified = OUT_DIR / "rf_ddos_misclassified_rows.csv"
out_fp = OUT_DIR / "rf_ddos_false_positive_rows.csv"
out_fn = OUT_DIR / "rf_ddos_false_negative_rows.csv"
out_summary = OUT_DIR / "rf_ddos_misclassification_feature_summary.csv"

misclassified.to_csv(out_misclassified, index=False)
false_positives.to_csv(out_fp, index=False)
false_negatives.to_csv(out_fn, index=False)
summary.to_csv(out_summary, index=False)

print("\nSaved:")
print(out_misclassified)
print(out_fp)
print(out_fn)
print(out_summary)

print("\nMisclassified rows:")
print(misclassified[["original_index", "true_label", "predicted_label", "error_type"] + available_original_cols].to_string(index=False))

print("\nSummary rows:")
print(summary[["group", "rows"]].to_string(index=False))