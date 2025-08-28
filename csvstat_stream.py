#!/usr/bin/env python3
import csv, sys, os, math, argparse, random, subprocess, hashlib, bisect, time, re
from typing import List, Tuple, Optional
from collections import deque  # progress window

# optional psutil (restored)
try:
    import psutil  # type: ignore
except Exception:
    psutil = None

# --- performance additions ---
try:
    import xxhash  # optional fast hash
    def _fast_hash64(val: str) -> int:
        return xxhash.xxh64(val).intdigest()  # 64-bit
except Exception:  # fallback
    def _fast_hash64(val: str) -> int:
        return int.from_bytes(hashlib.blake2b(val.encode('utf-8','ignore'), digest_size=8).digest(), 'big')

_NUM_RE = re.compile(r'^[+-]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][+-]?\d+)?$')

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
    def __init__(self, k: int = 256):  # reduced k for speed
        self.k = k
        self._exact: Optional[set] = set()
        self._hashes: List[int] = []
    def _hash64(self, val: str) -> int:
        return _fast_hash64(val)
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
    __slots__ = ("n","nn","nulls","mean","M2","min","max","sum","max_decimals","longest_len","is_numeric","unique")
    def __init__(self):
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
        self.unique = KMVEstimator(k=256)
    def add_cell(self, cell: str):
        self.n += 1
        if cell == "":
            self.nulls += 1
            self.unique.add("None")
            return
        # Fast numeric path (avoid exceptions):
        if self.is_numeric and _NUM_RE.match(cell):
            try:
                x = float(cell)
            except ValueError:  # extremely rare given regex
                self.is_numeric = False
            else:
                self.nn += 1
                if x < self.min: self.min = x
                if x > self.max: self.max = x
                d = x - self.mean
                self.mean += d / self.nn
                self.M2 += d * (x - self.mean)
                self.sum += x
                if "." in cell:
                    decs = len(cell.rsplit(".",1)[1].rstrip("0"))
                    if decs > self.max_decimals:
                        self.max_decimals = decs
                self.unique.add(cell)
                return
        # Text / mixed fallback
        self.is_numeric = False
        l = len(cell)
        if l > self.longest_len:
            self.longest_len = l
        self.unique.add(cell)
    def result(self) -> dict:
        stdev = math.sqrt(self.M2 / (self.nn - 1)) if self.nn > 1 else None
        uniq_est, uniq_exact = self.unique.estimate()
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

# --- Adaptive time projection for BUFFER ---
def project_buffer_time(path: str, max_seconds: float = 1.5, max_rows: int = 200_000) -> Optional[float]:
    size = os.stat(path).st_size
    start = time.time()
    rows = 0
    bytes_read = 0
    try:
        with open(path, 'rb') as f:
            for line in f:
                rows += 1
                bytes_read += len(line)
                if rows >= max_rows or (time.time() - start) >= max_seconds:
                    break
    except Exception:
        return None
    elapsed = time.time() - start
    if elapsed <= 0 or rows == 0 or bytes_read == 0:
        return None
    bytes_per_sec = bytes_read / elapsed
    projected = size / bytes_per_sec
    return projected

def choose_mode(path: str, auto: bool, force_stream: bool, force_buffer: bool,
                mem_pct: float, threshold_bytes: int, seed: int,
                hard_cap_bytes: int, target_buffer_seconds: float) -> Tuple[str, str]:
    if force_stream: return ("STREAM", "forced")
    if force_buffer: return ("BUFFER", "forced")
    if not auto: return ("BUFFER", "default")

    size = os.stat(path).st_size
    # Hard cap first
    if size > hard_cap_bytes:
        return ("STREAM", f"> hard cap ({human(hard_cap_bytes)})")

    if size <= threshold_bytes:
        return ("BUFFER", f"≤ threshold ({human(threshold_bytes)})")

    avg_row, rows_sampled = sample_avg_row_bytes(path, n=50_000, seed=seed)
    if avg_row <= 0:
        return ("BUFFER", "empty file")

    est_rows = size / avg_row
    overhead = 3.0  # conservative Python object overhead factor
    est_mem = int(est_rows * avg_row * overhead)
    budget = int(total_ram_bytes() * mem_pct)

    mem_reason = None
    if est_mem <= budget:
        mem_reason = f"est_mem {human(est_mem)} ≤ budget {human(budget)}"
        # Still candidate for projection-based early STREAM decision
        projected = project_buffer_time(path)
        if projected is not None and projected > target_buffer_seconds:
            return ("STREAM", f"projected buffer time ~{int(projected)}s > target {int(target_buffer_seconds)}s")
        return ("BUFFER", mem_reason)
    return ("STREAM", f"est_mem {human(est_mem)} > budget {human(budget)}")

def stream_stats_filelike(fh, *, progress_every_rows: int = 0, show_progress: bool = False, total_size: Optional[int] = None, estimated_total_rows: Optional[float] = None):
    rdr = csv.reader(fh)
    headers = next(rdr)
    stats = [RunningStats() for _ in headers]
    rows = 0
    start = time.time()
    window = deque()  # (time, rows)
    last_report = 0
    def report():
        nonlocal last_report
        if not show_progress or progress_every_rows <= 0:
            return
        now = time.time()
        # maintain 5s window
        while window and now - window[0][0] > 5:
            window.popleft()
        if len(window) >= 2:
            dt = window[-1][0] - window[0][0]
            dr = window[-1][1] - window[0][1]
            rps = dr / dt if dt > 0 else 0
        else:
            rps = rows / (now - start) if (now - start) > 0 else 0
        pct_str = '  ?.%'
        eta_str = ''
        if estimated_total_rows and estimated_total_rows > 0 and rows > 0:
            pct = min(100.0, (rows / estimated_total_rows) * 100)
            pct_str = f"{pct:5.1f}%"
            if rps > 0 and rows < estimated_total_rows:
                remaining_rows = max(0.0, estimated_total_rows - rows)
                eta = remaining_rows / rps
                eta_str = f" | ETA ~{int(eta)}s"
        print(f"[progress] {pct_str} | {rps:,.0f} rows/s{eta_str}", file=sys.stderr)
        last_report = rows
    for row in rdr:
        # Handle ragged rows gracefully
        if len(row) < len(headers):
            row += [""] * (len(headers) - len(row))
        for i, cell in enumerate(row[:len(headers)]):
            stats[i].add_cell(cell)
        rows += 1
        if show_progress and progress_every_rows > 0 and rows % progress_every_rows == 0:
            window.append((time.time(), rows))
            report()
    # final progress report
    if show_progress and rows != last_report:
        window.append((time.time(), rows))
        report()
    return headers, [s.result() for s in stats]

def stream_stats_path(path: str, *, progress_every_rows: int = 0, show_progress: bool = False, estimated_total_rows: Optional[float] = None) -> Tuple[List[str], List[dict]]:
    total_size = os.stat(path).st_size if show_progress else None
    with open(path, newline="", encoding="utf-8") as f:
        return stream_stats_filelike(
            f,
            progress_every_rows=progress_every_rows,
            show_progress=show_progress,
            total_size=total_size,
            estimated_total_rows=estimated_total_rows
        )

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
    ap.add_argument("--hard-cap-mb", type=int, default=300, help="> this size forces STREAM regardless of other heuristics (MB)")
    ap.add_argument("--target-buffer-seconds", type=float, default=60.0, help="If projected BUFFER time exceeds this, use STREAM")
    ap.add_argument("--buffer-timeout-sec", type=int, default=300, help="Max seconds to allow BUFFER (csvstat) before falling back to STREAM (0 = no timeout)")
    ap.add_argument("--progress-every-rows", type=int, default=500_000, help="Emit streaming progress every N rows (STREAM mode only; 0=disable)")
    ap.add_argument("--no-progress", action="store_true", help="Disable progress/ETA output in STREAM mode")
    ap.add_argument("--expected-rows", type=float, default=None, help="Known total data rows (excludes header). If provided, used for exact progress % & ETA.")
    ap.add_argument("--seed", type=int, default=1337, help="Deterministic seed for sampling/benchmarks")
    args = ap.parse_args()

    # stdin special-case: no auto (no filesize); stream directly
    if args.file == "-":
        print(f"Mode: STREAM — stdin | reason=no size available | seed={args.seed}")
        headers, results = stream_stats_filelike(sys.stdin, progress_every_rows=args.progress_every_rows, show_progress=not args.no_progress, total_size=None)
        print_stream_results(headers, results)
        print("\nNote: STREAM mode — core stats exact (row count, non-null numeric count, min, max, mean, stdev, sum). Unique counts are approximate (rounded). Median, frequency, and decimal-place heavy analysis removed for speed.", file=sys.stderr)
        sys.exit(0)

    size = os.stat(args.file).st_size
    mode, why = choose_mode(
        args.file, args.auto, args.force_stream, args.force_buffer,
        args.mem_budget, args.threshold_mb * MB, args.seed,
        args.hard_cap_mb * MB, args.target_buffer_seconds
    )
    print(f"Mode: {mode} — file={human(size)} | reason={why} | seed={args.seed}")

    if mode == "BUFFER":
        try:
            start = time.time()
            # Use run with timeout if specified
            if args.buffer_timeout_sec > 0:
                try:
                    r = subprocess.run(["csvstat", args.file], timeout=args.buffer_timeout_sec)
                    sys.exit(r.returncode)
                except subprocess.TimeoutExpired:
                    elapsed = time.time() - start
                    print(f"(BUFFER csvstat exceeded {args.buffer_timeout_sec}s (elapsed ~{int(elapsed)}s); falling back to STREAM)\n", file=sys.stderr)
                except FileNotFoundError:
                    print("(csvstat not found — falling back to STREAM path)\n", file=sys.stderr)
            else:
                r = subprocess.run(["csvstat", args.file])
                sys.exit(r.returncode)
        except Exception as e:
            print(f"(BUFFER path error: {e}; falling back to STREAM)\n", file=sys.stderr)

    # STREAM path
    est_total_rows = None
    if not args.no_progress and args.progress_every_rows > 0:
        if args.expected_rows and args.expected_rows > 0:
            est_total_rows = float(args.expected_rows)
        else:
            # reuse sampling to estimate total rows for percent + ETA
            avg_row, _ = sample_avg_row_bytes(args.file, n=50_000, seed=args.seed)
            if avg_row > 0:
                # subtract header size impact by ignoring first line in avg (already done in sample as first line included) – acceptable approximation
                est_total_rows = size / avg_row
    headers, results = stream_stats_path(
        args.file,
        progress_every_rows=args.progress_every_rows,
        show_progress=not args.no_progress,
        estimated_total_rows=est_total_rows
    )
    print_stream_results(headers, results)
    print("\nNote: STREAM mode — core stats exact (row count, non-null numeric count, min, max, mean, stdev, sum). Unique counts may be approximate (rounded). Median, frequency, and decimal-place heavy analysis removed for speed.", file=sys.stderr)

if __name__ == "__main__":
    main()
