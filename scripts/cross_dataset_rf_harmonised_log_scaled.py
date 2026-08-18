from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

project_path = Path(r"C:\Users\tasos\Desktop\thesis-ids-comparison")

cicids_file = project_path / "data" / "processed" / "cicids2017_harmonised_stratified_sample.csv.gz"
unsw_file = project_path / "data" / "processed" / "unsw_nb15_harmonised.csv.gz"

results_tables_path = project_path / "results" / "tables"
confusion_matrices_path = project_path / "results" / "confusion_matrices"

results_tables_path.mkdir(parents=True, exist_ok=True)
confusion_matrices_path.mkdir(parents=True, exist_ok=True)

print("Loading harmonised datasets...")

cicids = pd.read_csv(cicids_file)
unsw = pd.read_csv(unsw_file)

print("Original CICIDS2017 shape:", cicids.shape)
print("Original UNSW-NB15 shape:", unsw.shape)

features = [col for col in cicids.columns if col != "Binary_Label"]

# Remove invalid negative values
for feature in features:
    cicids = cicids[cicids[feature] >= 0]
    unsw = unsw[unsw[feature] >= 0]

print("\nAfter removing negative values:")
print("CICIDS2017 shape:", cicids.shape)
print("UNSW-NB15 shape:", unsw.shape)

# Unit conversion for CICIDS2017 time-based features
# CICIDS Flow Duration is converted from microseconds to seconds
# CICIDS IAT means are converted from microseconds to milliseconds
cicids["duration"] = cicids["duration"] / 1_000_000
cicids["fwd_iat_mean"] = cicids["fwd_iat_mean"] / 1_000
cicids["bwd_iat_mean"] = cicids["bwd_iat_mean"] / 1_000

# Apply log1p transformation to reduce extreme skew
# log1p(x) = log(1 + x), safe for zero values
for feature in features:
    cicids[feature] = np.log1p(cicids[feature])
    unsw[feature] = np.log1p(unsw[feature])

print("\nAfter unit conversion and log transformation:")
print("CICIDS2017 feature summary:")
print(cicids[features].describe().T[["mean", "std", "min", "max"]])

print("\nUNSW-NB15 feature summary:")
print(unsw[features].describe().T[["mean", "std", "min", "max"]])

print("\nCICIDS2017 label distribution:")
print(cicids["Binary_Label"].value_counts())

print("\nUNSW-NB15 label distribution:")
print(unsw["Binary_Label"].value_counts())

def evaluate_cross_dataset(train_df, test_df, train_name, test_name):
    print("\n" + "=" * 70)
    print(f"Training on {train_name}, testing on {test_name}")
    print("=" * 70)

    X_train = train_df[features]
    y_train = train_df["Binary_Label"]

    X_test = test_df[features]
    y_test = test_df["Binary_Label"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    print("Training model...")
    model.fit(X_train_scaled, y_train)

    print("Making predictions...")
    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn)

    print("\nResults:")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    print("False Positive Rate:", fpr)

    print("\nConfusion Matrix:")
    print(cm)

    print("\nTN:", tn)
    print("FP:", fp)
    print("FN:", fn)
    print("TP:", tp)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Benign", "Attack"]
    )

    disp.plot()
    plt.title(f"Log-Scaled Cross-Dataset RF: Train {train_name}, Test {test_name}")
    plt.tight_layout()

    cm_filename = f"rf_cross_log_scaled_train_{train_name}_test_{test_name}_confusion_matrix.png"
    cm_file = confusion_matrices_path / cm_filename
    plt.savefig(cm_file, dpi=300, bbox_inches="tight")
    plt.close()

    print("\nConfusion matrix saved:")
    print(cm_file)

    return {
        "Train_Dataset": train_name,
        "Test_Dataset": test_name,
        "Training_Rows": len(train_df),
        "Testing_Rows": len(test_df),
        "Number_of_Features": len(features),
        "Transformation": "unit_conversion_log1p_standard_scaling",
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "False Positive Rate": fpr,
        "True Negatives": tn,
        "False Positives": fp,
        "False Negatives": fn,
        "True Positives": tp
    }

results = []

results.append(
    evaluate_cross_dataset(
        cicids,
        unsw,
        "CICIDS2017",
        "UNSW-NB15"
    )
)

results.append(
    evaluate_cross_dataset(
        unsw,
        cicids,
        "UNSW-NB15",
        "CICIDS2017"
    )
)

results_df = pd.DataFrame(results)

results_file = results_tables_path / "rf_cross_dataset_harmonised_log_scaled_results.csv"
results_df.to_csv(results_file, index=False)

print("\nLog-scaled cross-dataset results saved successfully:")
print(results_file)

print("\nFinal log-scaled cross-dataset results:")
print(results_df)