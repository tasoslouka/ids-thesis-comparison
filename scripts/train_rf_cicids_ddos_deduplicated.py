from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold,
)
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

DATA_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "MachineLearningCSV"
    / "MachineLearningCVE"
    / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
)

RESULTS_TABLES = BASE_DIR / "results" / "tables"
RESULTS_FIGURES = BASE_DIR / "results" / "confusion_matrices"

RESULTS_TABLES.mkdir(parents=True, exist_ok=True)
RESULTS_FIGURES.mkdir(parents=True, exist_ok=True)


def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "Recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "F1_Score": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "False_Positive_Rate": (
            fp / (fp + tn)
            if (fp + tn) > 0
            else 0
        ),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


def save_confusion_matrix(cm, title, output_path):
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Benign", "DDoS"],
    )

    display.plot(values_format="d")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():

    print("=" * 72)
    print("CICIDS2017 DDoS - DEDUPLICATED RANDOM FOREST EXPERIMENT")
    print("=" * 72)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print("\nLoading CICIDS2017 DDoS dataset...")

    df = pd.read_csv(DATA_PATH)

    print(f"Original shape: {df.shape}")

    df.columns = df.columns.str.strip()

    missing_values = df.isna().sum().sum()

    infinite_values = (
        np.isinf(
            df.select_dtypes(include=[np.number])
        )
        .sum()
        .sum()
    )

    print(f"Missing values: {missing_values}")
    print(f"Infinite values: {infinite_values}")

    # --------------------------------------------------------
    # ORIGINAL CLEANING
    # --------------------------------------------------------

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    df = df.dropna()

    print(f"\nCleaned shape before deduplication: {df.shape}")

    print("\nLabel distribution before deduplication:")
    print(df["Label"].value_counts())

    df["Binary_Label"] = df["Label"].apply(
        lambda x: 0 if x == "BENIGN" else 1
    )

    X = df.drop(
        columns=["Label", "Binary_Label"]
    )

    y = df["Binary_Label"]

    # --------------------------------------------------------
    # EXACT FEATURE DUPLICATE REMOVAL
    # --------------------------------------------------------

    duplicate_mask = X.duplicated(
        keep="first"
    )

    duplicate_count = int(
        duplicate_mask.sum()
    )

    print("\n" + "=" * 72)
    print("EXACT FEATURE DEDUPLICATION")
    print("=" * 72)

    print(
        f"Rows before deduplication: {len(X):,}"
    )

    print(
        f"Exact duplicate rows removed: "
        f"{duplicate_count:,}"
    )

    # Remove duplicate feature vectors and their corresponding
    # labels, keeping the first occurrence of each unique vector.

    X = X.loc[
        ~duplicate_mask
    ].reset_index(drop=True)

    y = y.loc[
        ~duplicate_mask
    ].reset_index(drop=True)

    print(
        f"Rows after deduplication: {len(X):,}"
    )

    print("\nClass distribution after deduplication:")

    print(
        y.value_counts()
        .sort_index()
        .rename(
            index={
                0: "BENIGN",
                1: "DDoS",
            }
        )
    )

    # --------------------------------------------------------
    # TRAIN / TEST SPLIT
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\n" + "=" * 72)
    print("DEDUPLICATED STRATIFIED 80/20 SPLIT")
    print("=" * 72)

    print(f"random_state: {RANDOM_STATE}")
    print(f"Training rows: {X_train.shape[0]:,}")
    print(f"Testing rows:  {X_test.shape[0]:,}")

    print("\nTraining class distribution:")
    print(
        y_train.value_counts()
        .sort_index()
        .rename(
            index={
                0: "BENIGN",
                1: "DDoS",
            }
        )
    )

    print("\nTesting class distribution:")
    print(
        y_test.value_counts()
        .sort_index()
        .rename(
            index={
                0: "BENIGN",
                1: "DDoS",
            }
        )
    )

    # --------------------------------------------------------
    # SAFETY CHECK:
    # TRAIN / TEST SHOULD NOW HAVE NO EXACT FEATURE OVERLAP
    # --------------------------------------------------------

    train_hashes = pd.util.hash_pandas_object(
        X_train,
        index=False,
    )

    test_hashes = pd.util.hash_pandas_object(
        X_test,
        index=False,
    )

    shared_hashes = set(
        train_hashes.to_numpy()
    ).intersection(
        set(test_hashes.to_numpy())
    )

    print("\nTrain/test exact feature overlap after deduplication:")
    print(
        f"Shared feature patterns: {len(shared_hashes):,}"
    )

    if len(shared_hashes) != 0:
        raise RuntimeError(
            "Exact train/test feature overlap still exists "
            "after deduplication."
        )

    # --------------------------------------------------------
    # BASELINE RANDOM FOREST
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("TRAINING BASELINE RANDOM FOREST")
    print("=" * 72)

    baseline_model = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )

    baseline_model.fit(
        X_train,
        y_train,
    )

    baseline_pred = baseline_model.predict(
        X_test
    )

    baseline_metrics = calculate_metrics(
        y_test,
        baseline_pred,
    )

    baseline_metrics["Model"] = (
        "Baseline Random Forest - Deduplicated"
    )

    print("\nBaseline metrics:")

    for key, value in baseline_metrics.items():
        print(f"{key}: {value}")

    # --------------------------------------------------------
    # TUNING SAMPLE
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("PREPARING TUNING SAMPLE")
    print("=" * 72)

    tuning_size = min(
        80000,
        len(X_train),
    )

    X_tune, _, y_tune, _ = train_test_split(
        X_train,
        y_train,
        train_size=tuning_size,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    print(
        f"Tuning sample size: "
        f"{X_tune.shape[0]:,}"
    )

    # Same search space as original experiment.

    param_distributions = {
        "n_estimators": [
            100,
            200,
            300,
        ],
        "max_depth": [
            None,
            10,
            20,
            30,
            40,
        ],
        "min_samples_split": [
            2,
            5,
            10,
        ],
        "min_samples_leaf": [
            1,
            2,
            4,
        ],
        "max_features": [
            "sqrt",
            "log2",
            0.5,
        ],
        "class_weight": [
            "balanced",
            "balanced_subsample",
            None,
        ],
    }

    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    rf_for_search = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    search = RandomizedSearchCV(
        estimator=rf_for_search,
        param_distributions=param_distributions,
        n_iter=12,
        scoring="f1",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=2,
    )

    print("\nRunning RandomizedSearchCV...")

    search.fit(
        X_tune,
        y_tune,
    )

    print("\nBest parameters:")
    print(search.best_params_)

    print(
        f"Best CV F1 score: "
        f"{search.best_score_}"
    )

    best_params = search.best_params_

    # --------------------------------------------------------
    # TUNED MODEL
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("TRAINING TUNED RANDOM FOREST")
    print("=" * 72)

    tuned_model = RandomForestClassifier(
        **best_params,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    tuned_model.fit(
        X_train,
        y_train,
    )

    tuned_pred = tuned_model.predict(
        X_test
    )

    tuned_metrics = calculate_metrics(
        y_test,
        tuned_pred,
    )

    tuned_metrics["Model"] = (
        "Tuned Random Forest - Deduplicated"
    )

    print("\nTuned metrics:")

    for key, value in tuned_metrics.items():
        print(f"{key}: {value}")

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        [
            baseline_metrics,
            tuned_metrics,
        ]
    )

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

    results_path = (
        RESULTS_TABLES
        / "rf_cicids2017_ddos_deduplicated_baseline_vs_tuned.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    # --------------------------------------------------------
    # SAVE BEST PARAMETERS
    # --------------------------------------------------------

    best_params_df = pd.DataFrame(
        [best_params]
    )

    best_params_df[
        "Best_CV_F1"
    ] = search.best_score_

    best_params_path = (
        RESULTS_TABLES
        / "rf_cicids2017_ddos_deduplicated_best_params.csv"
    )

    best_params_df.to_csv(
        best_params_path,
        index=False,
    )

    # --------------------------------------------------------
    # SAVE DEDUPLICATION SUMMARY
    # --------------------------------------------------------

    dedup_summary = pd.DataFrame(
        [
            {
                "Original_Cleaned_Rows": (
                    len(X) + duplicate_count
                ),
                "Exact_Duplicates_Removed": (
                    duplicate_count
                ),
                "Deduplicated_Rows": len(X),
                "Training_Rows": len(X_train),
                "Testing_Rows": len(X_test),
                "Shared_Train_Test_Feature_Patterns": (
                    len(shared_hashes)
                ),
            }
        ]
    )

    dedup_summary_path = (
        RESULTS_TABLES
        / "rf_cicids2017_ddos_deduplicated_summary.csv"
    )

    dedup_summary.to_csv(
        dedup_summary_path,
        index=False,
    )

    # --------------------------------------------------------
    # CONFUSION MATRICES
    # --------------------------------------------------------

    baseline_cm = confusion_matrix(
        y_test,
        baseline_pred,
    )

    tuned_cm = confusion_matrix(
        y_test,
        tuned_pred,
    )

    baseline_cm_path = (
        RESULTS_FIGURES
        / "rf_cicids2017_ddos_deduplicated_baseline_confusion_matrix.png"
    )

    tuned_cm_path = (
        RESULTS_FIGURES
        / "rf_cicids2017_ddos_deduplicated_tuned_confusion_matrix.png"
    )

    save_confusion_matrix(
        baseline_cm,
        "Baseline Random Forest - Deduplicated CICIDS2017 DDoS",
        baseline_cm_path,
    )

    save_confusion_matrix(
        tuned_cm,
        "Tuned Random Forest - Deduplicated CICIDS2017 DDoS",
        tuned_cm_path,
    )

    # --------------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("DEDUPLICATED RF EXPERIMENT COMPLETE")
    print("=" * 72)

    print("\nSaved outputs:")
    print(results_path)
    print(best_params_path)
    print(dedup_summary_path)
    print(baseline_cm_path)
    print(tuned_cm_path)

    print("\n" + "=" * 72)
    print("IMPORTANT RESULTS FOR THESIS")
    print("=" * 72)

    print(
        f"Exact duplicates removed: "
        f"{duplicate_count:,}"
    )

    print(
        f"Deduplicated dataset rows: "
        f"{len(X):,}"
    )

    print(
        f"Shared feature patterns after split: "
        f"{len(shared_hashes):,}"
    )

    print("\nBASELINE:")
    print(
        f"Accuracy:  "
        f"{baseline_metrics['Accuracy']:.6f}"
    )
    print(
        f"Precision: "
        f"{baseline_metrics['Precision']:.6f}"
    )
    print(
        f"Recall:    "
        f"{baseline_metrics['Recall']:.6f}"
    )
    print(
        f"F1:        "
        f"{baseline_metrics['F1_Score']:.6f}"
    )
    print(
        f"FPR:       "
        f"{baseline_metrics['False_Positive_Rate']:.8f}"
    )
    print(
        "Confusion matrix: "
        f"TN={baseline_metrics['TN']}, "
        f"FP={baseline_metrics['FP']}, "
        f"FN={baseline_metrics['FN']}, "
        f"TP={baseline_metrics['TP']}"
    )

    print("\nTUNED:")
    print(
        f"Accuracy:  "
        f"{tuned_metrics['Accuracy']:.6f}"
    )
    print(
        f"Precision: "
        f"{tuned_metrics['Precision']:.6f}"
    )
    print(
        f"Recall:    "
        f"{tuned_metrics['Recall']:.6f}"
    )
    print(
        f"F1:        "
        f"{tuned_metrics['F1_Score']:.6f}"
    )
    print(
        f"FPR:       "
        f"{tuned_metrics['False_Positive_Rate']:.8f}"
    )
    print(
        "Confusion matrix: "
        f"TN={tuned_metrics['TN']}, "
        f"FP={tuned_metrics['FP']}, "
        f"FN={tuned_metrics['FN']}, "
        f"TP={tuned_metrics['TP']}"
    )

    print("\nBest tuned parameters:")
    print(best_params)

    print(
        f"Best tuning CV F1: "
        f"{search.best_score_:.6f}"
    )


if __name__ == "__main__":
    main()