# csvstat_stream_prototype

Streaming / auto-mode prototype inspired by csvkit's `csvstat` for large CSV profiling with constant memory.

## Features
- Auto mode chooses BUFFER (delegates to `csvstat`) or STREAM (one-pass) using file size, RAM budget, projected in-memory cost, and projected buffer time.
- STREAM mode (pure Python) provides: row count, per-column type inference, null counts, min/max/mean/stdev/sum (numeric), longest value length (text), approximate unique counts (KMV sketch with early exact mode), decimal precision detection.
- Alternate engine: Polars (lazy scan, streaming collect) for potentially faster computation on large files.
- Progress reporting with rows/s, percent complete, ETA (optional pre-pass for total rows).
- Graceful fallback if `csvstat` or optional engine not installed.
- Supports stdin (STREAM python engine only).

## Installation
Requires Python 3.9+.
Optional: install csvkit, polars, xxhash, psutil
```
pip install csvkit polars xxhash psutil
```

## CLI
```
./csvstat_stream.py FILE [options]
  --auto                     Auto choose mode
  --force-stream / --force-buffer
  --mem-budget FLOAT         Fraction RAM budget for BUFFER (default 0.25)
  --threshold-mb INT         Always BUFFER if file <= this size (default 100)
  --hard-cap-mb INT          Always STREAM if file > this size (default 300)
  --target-buffer-seconds F  If projected BUFFER time exceeds, choose STREAM (default 60)
  --buffer-timeout-sec INT   Kill csvstat if exceeds time and fallback (default 300)
  --progress                 Show streaming progress (extra pre-pass)
  --progress-every-rows INT  Progress update interval (default 500k)
  --engine python|polars
  --seed INT                 Deterministic sampling
  --compare-csvstat          Run 3 timing/memory trials vs csvstat (buffer) and report averages
  --skip-unique              Skip unique estimation for speed
```
stdin is supported with `-` (forces STREAM python engine).

## Example
```
./csvstat_stream.py big.csv --auto --progress --engine polars
```

## Accuracy Contract (STREAM)
Core numeric stats are exact. Unique counts are approximate when cardinality exceeds 2k * sketch_k (KMV), rounded for readability. Omitted heavier stats (median, top frequencies) for constant memory and speed. Text length max is exact.

## Engines
- python: one-pass csv.reader loop with light Python data structures.
- polars: lazy scan, expression aggregation (may use more memory but very fast).
Falls back to python if Polars is unavailable.

Note: Previous experimental PyArrow engine support has been removed to keep the code lean.

## Synthetic Data Generator
`make_csv.py` makes synthetic CSVs of arbitrary size. These CSVs have a mix of nulls, numbers, and strings to mimic real world data.
```
python make_csv.py output.csv ROWS NUM_COLS STR_COLS NULL_RATE SEED
# Example (~100MB target try):
python make_csv.py large_100mb.csv 1200000 6 2 0.05 1337
```

## Manual Benchmarking
To do a quick comparison now:
```
gtime -v csvstat large.csv > /dev/null
gtime -v ./csvstat_stream.py --auto large.csv > /dev/null
./csvstat_stream.py --compare-csvstat large.csv
```
Replace `large.csv` with your dataset. `--compare-csvstat` still performs 3 streaming vs csvstat trials and reports averages (DNFs if csvstat exceeds timeout).

## Development Notes
- Unique estimator: KMV with k=256 (tunable). Switches from exact set to KMV after 512 unique insertions for speed.
- Progress reporting keeps a 5s moving window for rows/s smoothing.
- Auto mode memory model uses sampled avg row size * row count * overhead factor (3x) vs RAM budget.
- Buffer time projection samples initial chunk to estimate full parse time; if exceeding threshold, chooses STREAM early.
- Compare mode runs 3 trials each and reports averages; csvstat runs exceeding 300s are DNF.

## License
Prototype code for evaluation.
