from pathlib import Path
import pandas as pd
import numpy as np

project_path = Path(r"C:\Users\tasos\Desktop\thesis-ids-comparison")

cicids_file = project_path / "data" / "processed" / "cicids2017_clean_binary.csv.gz"
unsw_file = project_path / "data" / "processed" / "unsw_nb15_clean_binary.csv.gz"

processed_path = project_path / "data" / "processed"
results_tables_path = project_path / "results" / "tables"

processed_path.mkdir(parents=True, exist_ok=True)
results_tables_path.mkdir(parents=True, exist_ok=True)

print("Creating stratified harmonised feature datasets...")

# Load CICIDS2017 full processed dataset
print("\nLoading CICIDS2017 full processed dataset...")
cicids = pd.read_csv(cicids_file)
cicids.columns = cicids.columns.str.strip()

print("CICIDS2017 full shape:", cicids.shape)
print("CICIDS2017 label distribution:")
print(cicids["Binary_Label"].value_counts())

# Create stratified sample from CICIDS2017
# 250,000 benign + 250,000 attack where possible
benign_sample = cicids[cicids["Binary_Label"] == 0].sample(
    n=250000,
    random_state=42
)

attack_sample = cicids[cicids["Binary_Label"] == 1].sample(
    n=250000,
    random_state=42
)

cicids_sample = pd.concat([benign_sample, attack_sample], ignore_index=True)
cicids_sample = cicids_sample.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nCICIDS2017 stratified sample shape:", cicids_sample.shape)
print("CICIDS2017 stratified sample label distribution:")
print(cicids_sample["Binary_Label"].value_counts())

# Free memory
del cicids

# Load UNSW-NB15 full dataset
print("\nLoading UNSW-NB15 full dataset...")
unsw = pd.read_csv(unsw_file)
unsw.columns = unsw.columns.str.strip()

print("UNSW-NB15 shape:", unsw.shape)
print("UNSW-NB15 label distribution:")
print(unsw["Binary_Label"].value_counts())

# Create harmonised CICIDS2017 dataset
cicids_h = pd.DataFrame()

cicids_h["duration"] = cicids_sample["Flow Duration"]
cicids_h["fwd_packets"] = cicids_sample["Total Fwd Packets"]
cicids_h["bwd_packets"] = cicids_sample["Total Backward Packets"]
cicids_h["fwd_bytes"] = cicids_sample["Total Length of Fwd Packets"]
cicids_h["bwd_bytes"] = cicids_sample["Total Length of Bwd Packets"]
cicids_h["packet_rate"] = cicids_sample["Flow Packets/s"]
cicids_h["fwd_pkt_mean"] = cicids_sample["Fwd Packet Length Mean"]
cicids_h["bwd_pkt_mean"] = cicids_sample["Bwd Packet Length Mean"]
cicids_h["fwd_iat_mean"] = cicids_sample["Fwd IAT Mean"]
cicids_h["bwd_iat_mean"] = cicids_sample["Bwd IAT Mean"]
cicids_h["Binary_Label"] = cicids_sample["Binary_Label"]

# Free memory
del cicids_sample

# Create harmonised UNSW-NB15 dataset
unsw_h = pd.DataFrame()

unsw_h["duration"] = unsw["dur"]
unsw_h["fwd_packets"] = unsw["spkts"]
unsw_h["bwd_packets"] = unsw["dpkts"]
unsw_h["fwd_bytes"] = unsw["sbytes"]
unsw_h["bwd_bytes"] = unsw["dbytes"]
unsw_h["packet_rate"] = unsw["rate"]
unsw_h["fwd_pkt_mean"] = unsw["smean"]
unsw_h["bwd_pkt_mean"] = unsw["dmean"]
unsw_h["fwd_iat_mean"] = unsw["sinpkt"]
unsw_h["bwd_iat_mean"] = unsw["dinpkt"]
unsw_h["Binary_Label"] = unsw["Binary_Label"]

# Free memory
del unsw

# Replace infinite values and remove missing values
for name, df in [("CICIDS2017", cicids_h), ("UNSW-NB15", unsw_h)]:
    print(f"\n{name} harmonised before cleaning:", df.shape)
    print("Missing values:", df.isna().sum().sum())
    print("Infinite values:", np.isinf(df.select_dtypes(include=[np.number])).sum().sum())

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    print(f"{name} harmonised after cleaning:", df.shape)
    print("Missing values:", df.isna().sum().sum())
    print("Infinite values:", np.isinf(df.select_dtypes(include=[np.number])).sum().sum())
    print("Binary label distribution:")
    print(df["Binary_Label"].value_counts())

# Save harmonised datasets
cicids_out = processed_path / "cicids2017_harmonised_stratified_sample.csv.gz"
unsw_out = processed_path / "unsw_nb15_harmonised.csv.gz"

cicids_h.to_csv(cicids_out, index=False, compression="gzip")
unsw_h.to_csv(unsw_out, index=False, compression="gzip")

print("\nSaved stratified harmonised CICIDS2017 file:")
print(cicids_out)

print("\nSaved harmonised UNSW-NB15 file:")
print(unsw_out)

# Save mapping table
mapping_df = pd.DataFrame({
    "Harmonised_Feature": [
        "duration",
        "fwd_packets",
        "bwd_packets",
        "fwd_bytes",
        "bwd_bytes",
        "packet_rate",
        "fwd_pkt_mean",
        "bwd_pkt_mean",
        "fwd_iat_mean",
        "bwd_iat_mean"
    ],
    "CICIDS2017_Column": [
        "Flow Duration",
        "Total Fwd Packets",
        "Total Backward Packets",
        "Total Length of Fwd Packets",
        "Total Length of Bwd Packets",
        "Flow Packets/s",
        "Fwd Packet Length Mean",
        "Bwd Packet Length Mean",
        "Fwd IAT Mean",
        "Bwd IAT Mean"
    ],
    "UNSW_NB15_Column": [
        "dur",
        "spkts",
        "dpkts",
        "sbytes",
        "dbytes",
        "rate",
        "smean",
        "dmean",
        "sinpkt",
        "dinpkt"
    ]
})

mapping_file = results_tables_path / "harmonised_feature_mapping.csv"
mapping_df.to_csv(mapping_file, index=False)

print("\nFeature mapping saved:")
print(mapping_file)

print("\nStratified harmonised dataset creation completed successfully.")