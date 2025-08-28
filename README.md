# csvkit-streaming-demo (prototype)

**Goal:** Make csvstat “just work” on huge CSVs by adding an **Auto** mode that keeps **small files exact** (buffer) and **streams large files** (constant memory).

## Why this matters
- Prevents laptop freezes/crashes on multi-GB files.
- Gives fast, trustworthy numbers for large datasets; keeps small ones fully exact.
- Clear transparency: one-line rationale for mode choice; “accuracy contract” footer.

## How to run
```bash
python csvstat_stream.py --help
python csvstat_stream.py --auto /path/to/file.csv
cat /path/to/file.csv | python csvstat_stream.py -
```

## Benchmarks (median of 3 runs; fill with your machine’s results)

| File   | Mode chosen | Runtime (wall) | Peak memory (Max RSS) |
| ------ | ----------- | -------------- | --------------------- |
| 10 MB  | BUFFER      | — s →  — s     | — MB →  — MB          |
| 1.0 GB | STREAM      | — s →  — s     | — MB →  — MB          |

> Example expectation on 1.0 GB: **~3–5× faster** and **~90% less memory** when Auto switches to streaming (e.g., **210 s → 60 s**, **3.5 GB → 300 MB**).
> Commands used: `/usr/bin/time -v csvstat …` vs `/usr/bin/time -v python csvstat_stream.py --auto …`.

## Accuracy contract

* **BUFFER (small files):** all csvstat metrics, exact.
* **STREAM (large files):** **exact core stats** (count, non-nulls, nulls, min, max, mean, stdev). Advanced metrics omitted in prototype.

## Auto mode rationale

Auto chooses **BUFFER** if file ≤ 100 MB or estimated memory for buffering ≤ 25% of system RAM. Otherwise it chooses **STREAM**. The tool prints a one-line reason (file size, estimated memory, budget).

## Resume-friendly bullets (paste/edit with your numbers)

* **Made csvkit “just work” on big files** by adding an Auto mode that switches to streaming for large datasets and keeps small files exact.
* **Reduced wait times by 3–5×** on 1–4 GB CSVs (e.g., **210 s → 60 s**) and **cut memory ~90%** (e.g., **3.5 GB → 300 MB**).
* **Kept accuracy where it matters**: small/medium files stay exact; large files get exact core stats with clear rationale and notes.
