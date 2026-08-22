from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parents[1]
CSV_PATH = BASE_DIR / "results" / "tables" / "rf_ddos_ablation_results.csv"
OUTPUT_PATH = BASE_DIR / "results" / "figures" / "rf_ablation_f1_comparison.png"


def main():
    df = pd.read_csv(CSV_PATH)

    labels = [
        "All features",
        "Without Destination Port",
        "Without port and\nTCP window features",
        "Without five selected\nfeatures",
    ]

    f1_scores = df["f1_score"].tolist()

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.bar(labels, f1_scores)

    ax.set_title("Random Forest Feature Ablation F1-Score Comparison", fontsize=18, fontweight="bold", pad=18)
    ax.set_xlabel("Experiment", fontsize=13)
    ax.set_ylabel("F1-score", fontsize=13)
    ax.set_ylim(0.9995, 1.0000)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    for bar, score in zip(bars, f1_scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            score + 0.000008,
            f"{score:.6f}",
            ha="center",
            va="bottom",
            fontsize=12,
        )

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
