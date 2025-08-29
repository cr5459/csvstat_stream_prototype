# csvstat_stream_prototype

Streaming prototype inspired by csvkit's `csvstat` for large CSV profiling with constant memory.

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
python make_csv.py sample.csv 1500000 6 2 0.05 1337
```

## Manual Benchmarking
To do a quick comparison now:
```
gtime -v csvstat sample.csv >
gtime -v ./csvstat_stream.py --force-stream sample.csv
```
Replace `sample.csv` with your dataset.

## 🔄 Comparison: Buffering vs. Streaming vs. Polars vs. Pandas

Benchmarks on synthetic CSVs (6 numeric + 2 string columns, 5% nulls, seed=1337).  
Hardware: MacBook Pro M2 Pro, 16 GB RAM. Numbers are median of 3 runs.  
Memory = peak resident set size (RSS). Times from GNU `time -v`.

File Sizes: Small = 7.4 MB (100,000 rows), Large = 111 MB (1,500,000 rows)

| File Size | Tool / Mode         | Runtime (wall) | Peak Memory | Notes |
|-----------|---------------------|----------------|-------------|-------|
| Small     | csvstat (BUFFER)    | 4.71 s          | 210 MB      |  |
| Small     | csvstat (STREAM)    | 0.59 s          | 21 MB       |  |
| Small     | Polars ()   | 0.01 s          | 125 MB       | |
| Large   | csvstat (BUFFER)    | 96 s   | 2.38 GB     | |
| Large    | csvstat (STREAM)    | 7.61 s       | 21 MB   |  |
| Large    | Polars (read_csv)   | 0.29 s           | 769 MB  |  |

### 🧾 Takeaways
- **Buffering mode** in csvstat often stalls on 300 MB+ files because it loads everything into memory.  
- **Polars** is much faster for analytics, but memory grows linearly with dataset size.  

## Development Notes
- Counts uniques exactly when small, then uses a quick sampling shortcut (KMV) after 512 items.  
- Shows speed as a smooth average over the last 5 seconds.  
- Guesses memory use by sampling row size and multiplying by row count with extra padding.  
- Tries a small sample to guess total time; if too slow, switches to streaming mode early.  
- Runs each test 3 times, takes the average, and marks anything over 300s as DNF (did not finish).  

## License
Prototype code for evaluation.
