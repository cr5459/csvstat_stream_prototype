#!/usr/bin/env python3
import csv, sys, os, math, argparse, random, subprocess
from typing import List, Tuple, Optional

try:
    import psutil  # optional
except Exception:
    psutil = None

MB = 1024 * 1024
GB = 1024 * MB

def human(n: int) -> str:
    if n >= GB: return f"{n/GB:.1f} GB"
    if n >= MB: return f"{n/MB:.1f} MB"
    return f"{n} B"

def total_ram_bytes() -> int:
    if psutil:
        try:
            return int(psutil.virtual_memory().total)
        except Exception:
            pass
    return 16 * GB  # conservative default

class RunningStats:
    __slots__ = ("n","nn","nulls","mean","M2","min","max")
    def __init__(self):
        self.n = 0
        self.nn = 0
        self.nulls = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.min = math.inf
        self.max = -math.inf
    def add_cell(self, cell: str):
        self.n += 1
        if cell == "":
            self.nulls += 1
            return
        try:
            x = float(cell)
        except ValueError:
            # treat text as non-numeric; still counted above
            return
        self.nn += 1
        if x < self.min: self.min = x
        if x > self.max: self.max = x
        d = x - self.mean
        self.mean += d / self.nn
        self.M2 += d * (x - self.mean)
    def result(self) -> dict:
        stdev = math.sqrt(self.M2 / (self.nn - 1)) if self.nn > 1 else None
        return {
            "count": self.n,
            "non_null": self.nn,
            "nulls": self.nulls,
            "min": None if self.nn == 0 else self.min,
            "max": None if self.nn == 0 else self.max,
            "mean": None if self.nn == 0 else self.mean,
            "stdev": stdev,
        }

def sample_avg_row_bytes(path: str, n: int = 50_000, seed: int = 1337) -> Tuple[float, int]:
    """Binary sample of first n lines to estimate avg row bytes. Deterministic via seed."""
    random.seed(seed)
    total = 0
    rows = 0
    with open(path, "rb") as f:
        for line in f:
            total += len(line)
            rows += 1
            if rows >= n:
                break
    if rows == 0:
        return (0.0, 0)
    return (total / rows, rows)

def choose_mode(path: str, auto: bool, force_stream: bool, force_buffer: bool,
                mem_pct: float, threshold_bytes: int, seed: int) -> Tuple[str, str]:
    if force_stream: return ("STREAM", "forced")
    if force_buffer: return ("BUFFER", "forced")
    if not auto: return ("BUFFER", "default")

    size = os.stat(path).st_size
    if size <= threshold_bytes:
        return ("BUFFER", f"≤ threshold ({human(threshold_bytes)})")

    avg_row, rows_sampled = sample_avg_row_bytes(path, n=50_000, seed=seed)
    if avg_row <= 0:
        return ("BUFFER", "empty file")

    est_rows = size / avg_row
    overhead = 3.0  # conservative Python object overhead factor
    est_mem = int(est_rows * avg_row * overhead)
    budget = int(total_ram_bytes() * mem_pct)

    if est_mem <= budget:
        return ("BUFFER", f"est_mem {human(est_mem)} ≤ budget {human(budget)}")
    return ("STREAM", f"est_mem {human(est_mem)} > budget {human(budget)}")

def stream_stats_filelike(fh) -> Tuple[List[str], List[dict]]:
    rdr = csv.reader(fh)
    headers = next(rdr)
    stats = [RunningStats() for _ in headers]
    for row in rdr:
        # Handle ragged rows gracefully
        if len(row) < len(headers):
            row += [""] * (len(headers) - len(row))
        for i, cell in enumerate(row[:len(headers)]):
            stats[i].add_cell(cell)
    return headers, [s.result() for s in stats]

def stream_stats_path(path: str) -> Tuple[List[str], List[dict]]:
    with open(path, newline="", encoding="utf-8") as f:
        return stream_stats_filelike(f)

def print_stream_results(headers: List[str], results: List[dict]):
    for h, r in zip(headers, results):
        print(
            f"{h}\tcount={r['count']}\t"
            f"non_null={r['non_null']}\tnulls={r['nulls']}\t"
            f"min={r['min']}\tmax={r['max']}\tmean={r['mean']}\tstdev={r['stdev']}"
        )

def main():
    ap = argparse.ArgumentParser(description="Streaming csvstat prototype with Auto mode")
    ap.add_argument("file", help="CSV path or '-' for stdin")
    ap.add_argument("--auto", action="store_true", help="Auto choose BUFFER for small files, STREAM for large files")
    ap.add_argument("--force-stream", action="store_true", help="Force STREAM mode")
    ap.add_argument("--force-buffer", action="store_true", help="Force BUFFER mode")
    ap.add_argument("--mem-budget", type=float, default=0.25, help="Fraction of RAM for BUFFER (default 0.25)")
    ap.add_argument("--threshold-mb", type=int, default=100, help="BUFFER if file ≤ this size (MB)")
    ap.add_argument("--seed", type=int, default=1337, help="Deterministic seed for sampling/benchmarks")
    args = ap.parse_args()

    # stdin special-case: no auto (no filesize); stream directly
    if args.file == "-":
        print(f"Mode: STREAM — stdin | reason=no size available | seed={args.seed}")
        headers, results = stream_stats_filelike(sys.stdin)
        print_stream_results(headers, results)
        print("\nNote: STREAM mode — core stats exact (count, nulls, min, max, mean, stdev); advanced metrics omitted in prototype.", file=sys.stderr)
        sys.exit(0)

    size = os.stat(args.file).st_size
    mode, why = choose_mode(
        args.file, args.auto, args.force_stream, args.force_buffer,
        args.mem_budget, args.threshold_mb * MB, args.seed
    )
    print(f"Mode: {mode} — file={human(size)} | reason={why} | seed={args.seed}")

    if mode == "BUFFER":
        # Try delegating to real csvstat for realism
        try:
            rc = subprocess.call(["csvstat", args.file])
            sys.exit(rc)
        except Exception:
            print("(csvstat not found — falling back to STREAM path)\n", file=sys.stderr)

    # STREAM path
    headers, results = stream_stats_path(args.file)
    print_stream_results(headers, results)
    print("\nNote: STREAM mode — core stats exact (count, nulls, min, max, mean, stdev); advanced metrics omitted in prototype.", file=sys.stderr)

if __name__ == "__main__":
    main()
