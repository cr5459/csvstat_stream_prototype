#!/usr/bin/env bash
set -euo pipefail

# Small (~10–20MB) and Large (~1GB) datasets
python make_csv.py /tmp/small.csv 100000 6 2 0.05 1337
python make_csv.py /tmp/large.csv 10000000 6 2 0.05 1337

# Baseline vs prototype, run 3x each; capture /usr/bin/time -v
# Small: BUFFER expected
for i in 1 2 3; do /usr/bin/time -v csvstat /tmp/small.csv > /dev/null 2>> /tmp/base_small.txt || true; done
for i in 1 2 3; do /usr/bin/time -v python csvstat_stream.py --auto /tmp/small.csv > /dev/null 2>> /tmp/stream_small.txt || true; done

# Large: STREAM expected
for i in 1 2 3; do /usr/bin/time -v csvstat /tmp/large.csv > /dev/null 2>> /tmp/base_large.txt || true; done
for i in 1 2 3; do /usr/bin/time -v python csvstat_stream.py --auto /tmp/large.csv > /dev/null 2>> /tmp/stream_large.txt || true; done

# Summarize to markdown table
python summarize_times.py /tmp/base_small.txt /tmp/stream_small.txt /tmp/base_large.txt /tmp/stream_large.txt
