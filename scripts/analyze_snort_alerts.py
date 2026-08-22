from pathlib import Path
import argparse
import re
from datetime import datetime

import pandas as pd


HEADER_RE = re.compile(
    r"^(?P<stamp>\d{2}/\d{2}-\d{2}:\d{2}:\d{2}\.\d+)\s+"
    r"\[\*\*\]\s+\[(?P<gid>\d+):(?P<sid>\d+):(?P<rev>\d+)\]\s+"
    r"(?P<msg>.*?)\s+\[\*\*\]"
)

FLOW_RE = re.compile(
    r"(?P<src>\d{1,3}(?:\.\d{1,3}){3})(?::\d+)?\s+->\s+"
    r"(?P<dst>\d{1,3}(?:\.\d{1,3}){3})(?::\d+)?"
)


def parse_snort_timestamp(stamp, year):
    return datetime.strptime(f"{year}/{stamp}", "%Y/%m/%d-%H:%M:%S.%f")


def parse_reference_time(text):
    if not text:
        return None
    for fmt in (
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
    ):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    raise ValueError(
        "Could not parse --reference-time. Example: "
        "'2017-07-07 19:56:00.000000'"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Extract operational metrics from a Snort fast-alert log."
    )
    parser.add_argument("alert_file", type=Path)
    parser.add_argument(
        "--sid",
        type=int,
        action="append",
        default=[],
        help="Custom SID to analyse. Repeat for multiple SIDs.",
    )
    parser.add_argument(
        "--message-contains",
        type=str,
        default=None,
        help="Alternative filter when the SID is unknown; case-insensitive substring match.",
    )
    parser.add_argument("--year", type=int, default=2017)
    parser.add_argument(
        "--reference-time",
        type=str,
        default=None,
        help=(
            "Optional reference timestamp for latency, ideally the first matching "
            "attack packet or, if unavailable, the selected DDoS window start."
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/snort"),
    )
    args = parser.parse_args()

    if not args.alert_file.exists():
        raise FileNotFoundError(args.alert_file)

    if not args.sid and not args.message_contains:
        raise ValueError("Provide at least one --sid or --message-contains filter.")

    sid_filter = set(args.sid)
    message_filter = args.message_contains.lower() if args.message_contains else None

    events = []

    with args.alert_file.open("r", encoding="utf-8", errors="replace") as f:
        for line_number, line in enumerate(f, start=1):
            m = HEADER_RE.search(line)
            if not m:
                continue

            sid = int(m.group("sid"))
            message = m.group("msg").strip()

            sid_match = sid in sid_filter if sid_filter else False
            message_match = (
                message_filter in message.lower()
                if message_filter is not None
                else False
            )

            if not (sid_match or message_match):
                continue

            flow = FLOW_RE.search(line)

            events.append({
                "line_number": line_number,
                "timestamp": parse_snort_timestamp(m.group("stamp"), args.year),
                "gid": int(m.group("gid")),
                "sid": sid,
                "rev": int(m.group("rev")),
                "message": message,
                "source_ip": flow.group("src") if flow else None,
                "destination_ip": flow.group("dst") if flow else None,
            })

    if not events:
        raise RuntimeError(
            "No matching custom-rule alerts were parsed. "
            "Check that this is a Snort fast-alert file and that the SID is correct."
        )

    df = pd.DataFrame(events).sort_values("timestamp").reset_index(drop=True)
    df["alert_second"] = pd.to_datetime(df["timestamp"]).dt.floor("s")

    per_second = (
        df.groupby(["sid", "alert_second"])
        .size()
        .reset_index(name="alerts")
    )

    reference = parse_reference_time(args.reference_time)

    metrics_rows = []

    for sid, sid_df in df.groupby("sid"):
        sec_df = per_second[per_second["sid"] == sid]

        first_ts = sid_df["timestamp"].min()
        last_ts = sid_df["timestamp"].max()
        active_span = (last_ts - first_ts).total_seconds()

        latency = None
        if reference is not None:
            latency = (first_ts - reference).total_seconds()

        metrics_rows.append({
            "sid": int(sid),
            "message": sid_df["message"].iloc[0],
            "custom_rule_alerts": int(len(sid_df)),
            "first_alert_timestamp": first_ts.isoformat(sep=" "),
            "last_alert_timestamp": last_ts.isoformat(sep=" "),
            "latency_from_reference_seconds": latency,
            "alert_activity_span_seconds": float(active_span),
            "distinct_alert_seconds": int(sid_df["alert_second"].nunique()),
            "distinct_source_ips": int(sid_df["source_ip"].dropna().nunique()),
            "distinct_destination_ips": int(sid_df["destination_ip"].dropna().nunique()),
            "mean_alerts_per_active_second": float(sec_df["alerts"].mean()),
            "median_alerts_per_active_second": float(sec_df["alerts"].median()),
            "maximum_alerts_in_one_second": int(sec_df["alerts"].max()),
        })

    args.results_dir.mkdir(parents=True, exist_ok=True)

    stem = args.alert_file.stem
    metrics_path = args.results_dir / f"{stem}_extra_metrics.csv"
    per_second_path = args.results_dir / f"{stem}_alerts_per_second.csv"
    parsed_path = args.results_dir / f"{stem}_parsed_custom_alerts.csv"

    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
    per_second.to_csv(per_second_path, index=False)
    df.to_csv(parsed_path, index=False)

    print("=" * 78)
    print("SNORT EXTRA OPERATIONAL METRICS")
    print("=" * 78)
    print(pd.DataFrame(metrics_rows).to_string(index=False))
    print()
    print(f"Metrics:          {metrics_path}")
    print(f"Per-second data:  {per_second_path}")
    print(f"Parsed alerts:    {parsed_path}")
    print()
    print(
        "For the thesis, describe latency relative to the exact reference used. "
        "If the reference is only the PCAP-window start, do not call it true "
        "attack-onset latency."
    )


if __name__ == "__main__":
    main()
