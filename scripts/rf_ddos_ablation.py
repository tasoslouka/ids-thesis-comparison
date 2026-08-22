from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split


DATA_PATH = Path("data/raw/MachineLearningCSV/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")

OUT_DIR = Path("results/tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Loading: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

# Clean column names
df.columns = df.columns.str.strip()

print(f"Original shape: {df.shape}")
print("Columns:")
print(df.columns.tolist())

if "Label" not in df.columns:
    raise ValueError("Could not find Label column after stripping column names.")

# Binary label: BENIGN = 0, attack/DDoS = 1
df["Binary_Label"] = (df["Label"].astype(str).str.strip() != "BENIGN").astype(int)

# Drop non-feature columns
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

# Convert everything to numeric
X = X.apply(pd.to_numeric, errors="coerce")

# Replace inf with nan, then drop invalid rows
X = X.replace([np.inf, -np.inf], np.nan)
valid_mask = X.notna().all(axis=1)

X = X.loc[valid_mask].copy()
y = y.loc[valid_mask].copy()

print(f"Cleaned shape: {X.shape}")
print("Label distribution:")
print(y.value_counts().to_string())

# Same split for all ablation tests
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

experiments = {
    "A_all_features": [],
    "B_without_destination_port": ["Destination Port"],
    "C_without_port_and_tcp_window": [
        "Destination Port",
        "Init_Win_bytes_forward",
        "Init_Win_bytes_backward",
    ],
    "D_without_top5_features": [
        "Destination Port",
        "Max Packet Length",
        "Init_Win_bytes_forward",
        "Average Packet Size",
        "Min Packet Length",
    ],
}

results = []

for exp_name, removed_features in experiments.items():
    existing_removed = [f for f in removed_features if f in X_train.columns]
    missing_removed = [f for f in removed_features if f not in X_train.columns]

    selected_features = [c for c in X_train.columns if c not in existing_removed]

    print("\n" + "=" * 80)
    print(f"Experiment: {exp_name}")
    print(f"Removed existing features: {existing_removed}")
    print(f"Missing requested features: {missing_removed}")
    print(f"Number of features used: {len(selected_features)}")

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    model.fit(X_train[selected_features], y_train)
    y_pred = model.predict(X_test[selected_features])

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    row = {
        "experiment": exp_name,
        "features_removed": "; ".join(existing_removed) if existing_removed else "None",
        "missing_requested_features": "; ".join(missing_removed) if missing_removed else "None",
        "features_used": len(selected_features),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "false_positive_rate": fpr,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }

    results.append(row)

    print(json.dumps(row, indent=2))

results_df = pd.DataFrame(results)

out_csv = OUT_DIR / "rf_ddos_ablation_results.csv"
results_df.to_csv(out_csv, index=False)

print("\nSaved results to:")
print(out_csv)
print("\nFinal ablation results:")
print(results_df.to_string(index=False))