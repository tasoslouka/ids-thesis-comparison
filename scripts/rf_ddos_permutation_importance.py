from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score


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
RESULTS_FIGURES = BASE_DIR / "results" / "figures"

RESULTS_TABLES.mkdir(parents=True, exist_ok=True)
RESULTS_FIGURES.mkdir(parents=True, exist_ok=True)


def main():

    print("=" * 72)
    print("FRIDAY DDoS RANDOM FOREST - PERMUTATION IMPORTANCE")
    print("=" * 72)

    # --------------------------------------------------------
    # LOAD AND CLEAN DATA
    # --------------------------------------------------------

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    print(f"\nRaw shape: {df.shape}")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    print(f"Cleaned shape: {df.shape}")

    df["Binary_Label"] = df["Label"].apply(
        lambda x: 0 if x == "BENIGN" else 1
    )

    X = df.drop(columns=["Label", "Binary_Label"])
    y = df["Binary_Label"]

    # --------------------------------------------------------
    # REMOVE EXACT FEATURE DUPLICATES
    # --------------------------------------------------------

    duplicate_mask = X.duplicated(keep="first")
    duplicate_count = int(duplicate_mask.sum())

    X = X.loc[~duplicate_mask].reset_index(drop=True)
    y = y.loc[~duplicate_mask].reset_index(drop=True)

    print(f"Exact duplicates removed: {duplicate_count:,}")
    print(f"Deduplicated rows: {len(X):,}")

    # --------------------------------------------------------
    # SAME 80/20 STRATIFIED SPLIT
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Training rows: {len(X_train):,}")
    print(f"Testing rows:  {len(X_test):,}")

    # --------------------------------------------------------
    # BASELINE MODEL
    # Same baseline configuration as original experiment
    # --------------------------------------------------------

    print("\nTraining baseline Random Forest...")

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    baseline_f1 = f1_score(y_test, predictions)

    print(f"Baseline F1: {baseline_f1:.6f}")

    # --------------------------------------------------------
    # PERMUTATION IMPORTANCE
    # --------------------------------------------------------

    print("\nCalculating permutation importance...")
    print("This may take several minutes.")

    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="f1",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    importance_df = pd.DataFrame({
        "Feature": X_test.columns,
        "Importance_Mean": result.importances_mean,
        "Importance_Std": result.importances_std,
    })

    importance_df = importance_df.sort_values(
        by="Importance_Mean",
        ascending=False,
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # SAVE FULL TABLE
    # --------------------------------------------------------

    full_output_path = (
        RESULTS_TABLES
        / "rf_ddos_permutation_importance_all_features.csv"
    )

    importance_df.to_csv(
        full_output_path,
        index=False,
    )

    # --------------------------------------------------------
    # TOP 15 FEATURES
    # --------------------------------------------------------

    top15 = importance_df.head(15).copy()

    top15_output_path = (
        RESULTS_TABLES
        / "rf_ddos_permutation_importance_top15.csv"
    )

    top15.to_csv(
        top15_output_path,
        index=False,
    )

    print("\n" + "=" * 72)
    print("TOP 15 PERMUTATION IMPORTANCE FEATURES")
    print("=" * 72)

    print(
        top15.to_string(
            index=False,
            float_format=lambda x: f"{x:.8f}"
        )
    )

    # --------------------------------------------------------
    # PLOT
    # --------------------------------------------------------

    plot_df = top15.sort_values(
        "Importance_Mean",
        ascending=True,
    )

    plt.figure(figsize=(9, 7))

    plt.barh(
        plot_df["Feature"],
        plot_df["Importance_Mean"],
        xerr=plot_df["Importance_Std"],
    )

    plt.xlabel("Mean decrease in F1 after permutation")
    plt.ylabel("Feature")
    plt.title(
        "Permutation Importance - Friday CICIDS2017 DDoS Random Forest"
    )

    plt.tight_layout()

    figure_path = (
        RESULTS_FIGURES
        / "rf_ddos_permutation_importance_top15.png"
    )

    plt.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # --------------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("PERMUTATION IMPORTANCE COMPLETE")
    print("=" * 72)

    print(f"\nBaseline F1: {baseline_f1:.6f}")

    print("\nSaved:")
    print(full_output_path)
    print(top15_output_path)
    print(figure_path)


if __name__ == "__main__":
    main()