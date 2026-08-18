from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "MachineLearningCSV"
    / "MachineLearningCVE"
    / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
)


def main():

    print("=" * 72)
    print("CICIDS2017 FRIDAY DDoS - TEMPORAL/GROUP SPLIT INSPECTION")
    print("=" * 72)

    df = pd.read_csv(DATA_PATH)

    df.columns = df.columns.str.strip()

    print(f"\nRows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    print("\n" + "=" * 72)
    print("ALL COLUMNS")
    print("=" * 72)

    for i, column in enumerate(df.columns, start=1):
        print(f"{i:02d}. {column}")

    # --------------------------------------------------------
    # SEARCH FOR POSSIBLE TEMPORAL / GROUP IDENTIFIERS
    # --------------------------------------------------------

    keywords = [
        "time",
        "timestamp",
        "date",
        "source",
        "src",
        "destination",
        "dst",
        "ip",
        "flow id",
        "protocol",
    ]

    possible_metadata = []

    for column in df.columns:
        lower = column.lower()

        if any(keyword in lower for keyword in keywords):
            possible_metadata.append(column)

    print("\n" + "=" * 72)
    print("POSSIBLE TEMPORAL / GROUPING COLUMNS")
    print("=" * 72)

    if possible_metadata:
        for column in possible_metadata:
            print(column)
    else:
        print("None found.")

    # --------------------------------------------------------
    # LABEL ORDER / TRANSITIONS
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("LABEL ORDER")
    print("=" * 72)

    print("\nFirst 30 labels:")
    print(df["Label"].head(30).to_string(index=True))

    print("\nLast 30 labels:")
    print(df["Label"].tail(30).to_string(index=True))

    labels = df["Label"].astype(str).str.strip()

    transitions = labels.ne(labels.shift())

    transition_rows = df.loc[
        transitions,
        ["Label"]
    ].copy()

    print("\nLabel transition locations:")
    print(transition_rows.to_string())

    print(
        f"\nNumber of label runs: "
        f"{int(transitions.sum()):,}"
    )

    # --------------------------------------------------------
    # COUNTS BY QUARTER OF FILE
    # This does NOT prove time ordering.
    # It only helps inspect the dataset structure.
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("LABEL DISTRIBUTION BY FILE QUARTER")
    print("=" * 72)

    temp = pd.DataFrame({
        "Label": labels,
        "Position": range(len(df)),
    })

    temp["Quarter"] = pd.qcut(
        temp["Position"],
        q=4,
        labels=[
            "Q1",
            "Q2",
            "Q3",
            "Q4",
        ],
    )

    print(
        pd.crosstab(
            temp["Quarter"],
            temp["Label"],
        )
    )


if __name__ == "__main__":
    main()