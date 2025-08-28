#!/usr/bin/env python3
import csv, sys, os, math, argparse, random, subprocess, hashlib, bisect
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

# ---- Unique (KMV) estimator ----
class KMVEstimator:
    __slots__ = ("k", "_exact", "_hashes")
    def __init__(self, k: int = 1024):
        self.k = k
        self._exact: Optional[set] = set()
        self._hashes: List[int] = []
    def _hash64(self, val: str) -> int:
        return int.from_bytes(hashlib.blake2b(val.encode('utf-8','ignore'), digest_size=8).digest(), 'big')
    def add(self, val: str):
        if self._exact is not None:
            self._exact.add(val)
            # Switch to sketch mode when exact set grows large (2k heuristic)
            if len(self._exact) >= self.k * 2:
                # Build KMV structure
                hashes = [self._hash64(v) for v in self._exact]
                hashes.sort()
                self._hashes = hashes[:self.k]
                self._exact = None
            return
        # KMV mode
        h = self._hash64(val)
        if len(self._hashes) < self.k:
            bisect.insort(self._hashes, h)
        elif h < self._hashes[-1]:
            # replace largest
            self._hashes.pop()
            bisect.insort(self._hashes, h)
    def estimate(self) -> Tuple[int, bool]:
        if self._exact is not None:
            return (len(self._exact), True)
        if not self._hashes:
            return (0, True)
        if len(self._hashes) < self.k:
            return (len(self._hashes), True)
        r_k = self._hashes[-1] / (2**64)
        if r_k <= 0:
            return (len(self._hashes), False)
        est = int(round((self.k - 1) / r_k))
        return (est, False)

# ---- Running numeric/text stats ----
class RunningStats:
    __slots__ = ("n","nn","nulls","mean","M2","min","max","sum","max_decimals","longest_len","is_numeric","res_sample","res_k","unique","value_counts","heavy","_heavy_k")
    def __init__(self, res_k: int = 2048, heavy_k: int = 10):
        self.n = 0
        self.nn = 0
        self.nulls = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.min = math.inf
        self.max = -math.inf
        self.sum = 0.0
        self.max_decimals = 0
        self.longest_len = 0
        self.is_numeric = True
        self.res_sample: List[float] = []
        self.res_k = res_k
        self.unique = KMVEstimator(k=1024)
        self.value_counts: Optional[dict] = {}  # exact until >5k uniques
        self.heavy = {}  # Misra-Gries approximate heavy hitters
        self._heavy_k = heavy_k
    def _heavy_add(self, val: str):
        if val in self.heavy:
            self.heavy[val] += 1
            return
        if len(self.heavy) < self._heavy_k:
            self.heavy[val] = 1
        else:
            drop = []
            for k in self.heavy:
                self.heavy[k] -= 1
                if self.heavy[k] == 0:
                    drop.append(k)
            for k in drop:
                del self.heavy[k]
    def _reservoir_add(self, x: float):
        if len(self.res_sample) < self.res_k:
            self.res_sample.append(x)
        else:
            # Reservoir sampling replacement
            if random.random() < self.res_k / self.nn:
                idx = random.randint(0, self.res_k - 1)
                self.res_sample[idx] = x
    def _count_value(self, val: str):
        vc = self.value_counts
        if vc is not None:
            vc[val] = vc.get(val, 0) + 1
            if len(vc) > 5000:  # switch off exact counting
                self.value_counts = None  # fallback to heavy hitters only
        # Always feed heavy hitters (gives approximate top values even after cutoff)
        self._heavy_add(val)
    def add_cell(self, cell: str):
        self.n += 1
        if cell == "":
            self.nulls += 1
            self._count_value("None")
            self.unique.add("None")
            return
        # Try numeric first
        try:
            x = float(cell)
            # Numeric branch
            self.nn += 1
            if x < self.min: self.min = x
            if x > self.max: self.max = x
            d = x - self.mean
            self.mean += d / self.nn
            self.M2 += d * (x - self.mean)
            self.sum += x
            # Track decimal places (raw cell string)
            if "." in cell:
                decs = len(cell.rsplit(".",1)[1].rstrip("0"))
                if decs > self.max_decimals:
                    self.max_decimals = decs
            self._count_value(cell)
            self.unique.add(cell)
            self._reservoir_add(x)
        except ValueError:
            # Non-numeric; mark column as mixed/text
            self.is_numeric = False
            l = len(cell)
            if l > self.longest_len:
                self.longest_len = l
            self._count_value(cell)
            self.unique.add(cell)
    def _median(self) -> Optional[float]:
        if self.nn == 0 or not self.res_sample:
            return None
        # approximate median from reservoir sample
        s = sorted(self.res_sample)
        m = len(s)
        return s[m//2] if m % 2 == 1 else 0.5 * (s[m//2 -1] + s[m//2])
    def result(self) -> dict:
        stdev = math.sqrt(self.M2 / (self.nn - 1)) if self.nn > 1 else None
        uniq_est, uniq_exact = self.unique.estimate()
        median_val = self._median()
        median_exact = (self.nn > 0 and self.nn <= self.res_k and self.is_numeric)
        return {
            "count": self.n,
            "non_null_numeric": self.nn,
            "nulls": self.nulls,
            "min": None if self.nn == 0 else self.min,
            "max": None if self.nn == 0 else self.max,
            "mean": None if self.nn == 0 else self.mean,
            "stdev": stdev,
            "sum": None if self.nn == 0 else self.sum,
            "max_decimals": self.max_decimals if self.nn > 0 else None,
            "longest_len": self.longest_len if not self.is_numeric else None,
            "is_numeric": self.is_numeric and self.nn > 0,
            "unique": uniq_est,
            "unique_exact": uniq_exact,
            "median": median_val,
            "median_exact": median_exact,
            "value_counts": self.value_counts,
            "heavy": dict(self.heavy),
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
    def pr(label: str, value: str):
        print(f"    {label.ljust(24)}{value}")
    def fmt_num(x: Optional[float]):
        if x is None or (isinstance(x, float) and (math.isinf(x) or math.isnan(x))):
            return "None"
        # Show integers without decimal places, others with 3 decimals; always thousands separators
        if isinstance(x, (int,)) or (isinstance(x, float) and float(x).is_integer()):
            return f"{int(round(x)):,}"
        return f"{x:,.3f}"
    def fmt_unique(u: Optional[int]):
        if u is None:
            return "None"
        return f"{int(round(u/1000.0))*1000:,}"  # round to nearest 1000
    total_rows = results[0]['count'] if results else 0
    for idx, (h, r) in enumerate(zip(headers, results), start=1):
        is_num = r["is_numeric"]
        print(f"{idx}. \"{h}\"")
        has_nulls = r['nulls'] > 0
        if is_num:
            pr("Type of data:", "Number")
            pr("Contains null values:", f"{'True' if has_nulls else 'False'} (excluded from numeric stats)")
            pr("Non-null values:", f"{r['non_null_numeric']:,}")
            pr("Unique values:", fmt_unique(r['unique']))
            if r['min'] is not None:
                pr("Smallest value:", fmt_num(r['min']))
                pr("Largest value:", fmt_num(r['max']))
            if r['sum'] is not None:
                pr("Sum:", fmt_num(r['sum']))
            mean = r['mean']; stdev = r['stdev']
            pr("Mean:", fmt_num(mean))
            pr("StDev:", fmt_num(stdev))
            if r['max_decimals'] is not None:
                pr("Most decimal places:", str(r['max_decimals']))
        else:
            pr("Type of data:", "Text")
            pr("Contains null values:", 'True' if has_nulls else 'False')
            non_null_text = r['count'] - r['nulls']
            pr("Non-null values:", f"{non_null_text:,}")
            pr("Unique values:", fmt_unique(r['unique']))
            pr("Longest value length:", str(r['longest_len'] if r['longest_len'] is not None else 0))
        print()
    print(f"Row count: {total_rows:,}")

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
        print("\nNote: STREAM mode — core stats exact (row count, non-null numeric count, min, max, mean, stdev, sum). Unique counts are approximate. Median and most common values omitted in streaming mode.", file=sys.stderr)
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
    print("\nNote: STREAM mode — core stats exact (row count, non-null numeric count, min, max, mean, stdev, sum). Unique counts may be approximate. Median and most common values omitted in streaming mode.", file=sys.stderr)

if __name__ == "__main__":
    main()
