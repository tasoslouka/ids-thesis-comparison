#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 4 ]; then
  echo "Usage:"
  echo "  $0 <snort.conf> <pcap> <pcap_duration_seconds> <packet_count> [runs]"
  echo
  echo "Example:"
  echo "  $0 /etc/snort/snort.conf ddos_window.pcap 1259.30 1501082 3"
  exit 1
fi

CONF="$1"
PCAP="$2"
PCAP_DURATION="$3"
PACKETS="$4"
RUNS="${5:-3}"

RESULTS_DIR="results/snort/processing_benchmark"
mkdir -p "$RESULTS_DIR"

CSV="$RESULTS_DIR/snort_processing_benchmark.csv"
echo "run,elapsed_seconds,user_seconds,system_seconds,max_rss_kb,packets_per_second,realtime_factor" > "$CSV"

for i in $(seq 1 "$RUNS"); do
  RUN_DIR="$RESULTS_DIR/run_$i"
  rm -rf "$RUN_DIR"
  mkdir -p "$RUN_DIR"

  TIME_FILE="$RUN_DIR/time.csv"

  echo "Running benchmark $i/$RUNS ..."

  /usr/bin/time \
    -f "%e,%U,%S,%M" \
    -o "$TIME_FILE" \
    snort -q -c "$CONF" -r "$PCAP" -l "$RUN_DIR" -A fast \
    > "$RUN_DIR/stdout.txt" \
    2> "$RUN_DIR/stderr.txt"

  IFS=',' read -r ELAPSED USER_SEC SYS_SEC MAX_RSS < "$TIME_FILE"

  PPS=$(awk -v p="$PACKETS" -v e="$ELAPSED" 'BEGIN { if (e>0) printf "%.3f", p/e; else print "NA" }')
  RTF=$(awk -v d="$PCAP_DURATION" -v e="$ELAPSED" 'BEGIN { if (e>0) printf "%.3f", d/e; else print "NA" }')

  echo "$i,$ELAPSED,$USER_SEC,$SYS_SEC,$MAX_RSS,$PPS,$RTF" >> "$CSV"
done

echo
echo "Benchmark complete:"
echo "  $CSV"
echo
echo "Interpretation:"
echo "  realtime_factor > 1 means offline Snort processed the PCAP faster than real time."
echo "  Run all repetitions with the exact same Snort configuration and rule set used in the thesis."
