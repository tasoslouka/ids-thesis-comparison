from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

RANDOM_STATE = 42

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "raw" / "MachineLearningCSV" / "MachineLearningCVE" / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

RESULTS_TABLES = BASE_DIR / "results" / "tables"
RESULTS_FIGURES = BASE_DIR / "results" / "confusion_matrices"

RESULTS_TABLES.mkdir(parents=True, exist_ok=True)
RESULTS_FIGURES.mkdir(parents=True, exist_ok=True)


def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1_Score": f1_score(y_true, y_pred, zero_division=0),
        "False_Positive_Rate": fp / (fp + tn) if (fp + tn) > 0 else 0,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


def save_confusion_matrix(cm, title, output_path):
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Benign", "DDoS"]
    )
    display.plot(values_format="d")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    print("Loading CICIDS2017 DDoS dataset...")
    df = pd.read_csv(DATA_PATH)

    print(f"Original shape: {df.shape}")

    df.columns = df.columns.str.strip()

    missing_values = df.isna().sum().sum()
    infinite_values = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()

    print(f"Missing values: {missing_values}")
    print(f"Infinite values: {infinite_values}")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    print(f"Cleaned shape: {df.shape}")
    print("Label distribution:")
    print(df["Label"].value_counts())

    df["Binary_Label"] = df["Label"].apply(lambda x: 0 if x == "BENIGN" else 1)

    X = df.drop(columns=["Label", "Binary_Label"])
    y = df["Binary_Label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f"Training rows: {X_train.shape[0]}")
    print(f"Testing rows: {X_test.shape[0]}")

    print("\nTraining baseline Random Forest...")
    baseline_model = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced"
    )

    baseline_model.fit(X_train, y_train)
    baseline_pred = baseline_model.predict(X_test)
    baseline_metrics = calculate_metrics(y_test, baseline_pred)
    baseline_metrics["Model"] = "Baseline Random Forest"

    print("Baseline metrics:")
    print(baseline_metrics)

    print("\nPreparing smaller tuning sample from training data...")
    tuning_size = min(80000, len(X_train))

    X_tune, _, y_tune, _ = train_test_split(
        X_train,
        y_train,
        train_size=tuning_size,
        random_state=RANDOM_STATE,
        stratify=y_train
    )

    print(f"Tuning sample size: {X_tune.shape[0]}")

    param_distributions = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20, 30, 40],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", 0.5],
        "class_weight": ["balanced", "balanced_subsample", None],
    }

    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    rf_for_search = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    search = RandomizedSearchCV(
        estimator=rf_for_search,
        param_distributions=param_distributions,
        n_iter=12,
        scoring="f1",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=2
    )

    print("\nRunning RandomizedSearchCV...")
    search.fit(X_tune, y_tune)

    print("\nBest parameters:")
    print(search.best_params_)
    print(f"Best CV F1 score: {search.best_score_}")

    best_params = search.best_params_

    print("\nTraining tuned Random Forest on full training set...")
    tuned_model = RandomForestClassifier(
        **best_params,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    tuned_model.fit(X_train, y_train)
    tuned_pred = tuned_model.predict(X_test)
    tuned_metrics = calculate_metrics(y_test, tuned_pred)
    tuned_metrics["Model"] = "Tuned Random Forest"

    print("Tuned metrics:")
    print(tuned_metrics)

    results_df = pd.DataFrame([baseline_metrics, tuned_metrics])
    results_df = results_df[
        [
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1_Score",
            "False_Positive_Rate",
            "TN",
            "FP",
            "FN",
            "TP",
        ]
    ]

    results_path = RESULTS_TABLES / "rf_cicids2017_ddos_baseline_vs_tuned.csv"
    results_df.to_csv(results_path, index=False)

    best_params_df = pd.DataFrame([best_params])
    best_params_df["Best_CV_F1"] = search.best_score_
    best_params_path = RESULTS_TABLES / "rf_cicids2017_ddos_best_params.csv"
    best_params_df.to_csv(best_params_path, index=False)

    baseline_cm = confusion_matrix(y_test, baseline_pred)
    tuned_cm = confusion_matrix(y_test, tuned_pred)

    save_confusion_matrix(
        baseline_cm,
        "Baseline Random Forest - CICIDS2017 DDoS",
        RESULTS_FIGURES / "rf_cicids2017_ddos_baseline_confusion_matrix.png"
    )

    save_confusion_matrix(
        tuned_cm,
        "Tuned Random Forest - CICIDS2017 DDoS",
        RESULTS_FIGURES / "rf_cicids2017_ddos_tuned_confusion_matrix.png"
    )

    print("\nSaved outputs:")
    print(results_path)
    print(best_params_path)
    print(RESULTS_FIGURES / "rf_cicids2017_ddos_baseline_confusion_matrix.png")
    print(RESULTS_FIGURES / "rf_cicids2017_ddos_tuned_confusion_matrix.png")


if __name__ == "__main__":
    main()