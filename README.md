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
`make_csv.py` streams synthetic CSVs of arbitrary size.
```
python make_csv.py output.csv ROWS NUM_COLS STR_COLS NULL_RATE SEED
# Example (~100MB target try):
python make_csv.py large_100mb.csv 1200000 6 2 0.05 1337
```

## Benchmark Harness
`bench.sh` runs repeated timings across modes and engines.
```
./bench.sh large.csv --engines python,polars --modes auto,stream --reps 5
```
Outputs a TSV in `bench_runs/` with timing and derived rows/s.

Summarize:
```
./summarize_times.py bench_runs/bench_*.tsv
```

Quick ad‑hoc comparison (3 trials, averages + DNFs):
```
./csvstat_stream.py large.csv --compare-csvstat
```

## Benchmark results (median of 3 runs)
(Fill in with your measurements; SMALL = e.g. < threshold (BUFFER expected), LARGE = triggers STREAM.)

| Case         | Wall time (s) | Peak Memory (MB) |
|--------------|---------------|------------------|
| Small BASE   |               |                  |
| Small STREAM |               |                  |
| Large BASE   |               |                  |
| Large STREAM |               |                  |

Notes:
- BASE = csvkit csvstat (BUFFER) run
- STREAM = this prototype
- Use `--compare-csvstat` for automated averages; for medians run multiple times manually and compute median.

## Parity Test
`tests_small_parity.py` creates a small CSV and checks basic numeric parity vs csvstat (if available).
```
python3 tests_small_parity.py
```

## Development Notes
- Unique estimator: KMV with k=256 (tunable). Switches from exact set to KMV after 512 unique insertions for speed.
- Progress reporting keeps a 5s moving window for rows/s smoothing.
- Auto mode memory model uses sampled avg row size * row count * overhead factor (3x) vs RAM budget.
- Buffer time projection samples initial chunk to estimate full parse time; if exceeding threshold, chooses STREAM early.
- Compare mode now runs 3 trials each and reports averages; csvstat runs exceeding 300s are DNF.

## Roadmap Ideas
- Optional median / quantiles via t-digest or reservoir sampling.
- Frequency sketch (Heavy Hitters) for top values.
- Threaded IO prefetch for Python engine.
- Adaptive KMV size based on memory.

## License
Prototype code for evaluation.
