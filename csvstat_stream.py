#!/usr/bin/env python3
import csv, sys, os, math, argparse, random, subprocess, hashlib, bisect, time, re, shutil, threading
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
    __slots__ = ("n","nn","nulls","mean","M2","min","max","sum","max_decimals","longest_len","is_numeric","unique","track_unique")
    def __init__(self, track_unique: bool = True):
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
        self.track_unique = track_unique
        self.unique = KMVEstimator(k=256) if track_unique else None
    def add_cell(self, cell: str):
        self.n += 1
        if cell == "":
            self.nulls += 1
            if self.track_unique and self.unique is not None:
                self.unique.add("None")
            return
        if self.is_numeric:
            try:
                x = float(cell)
            except ValueError:
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
                if self.track_unique and self.unique is not None:
                    self.unique.add(cell)
                return
        self.is_numeric = False
        l = len(cell)
        if l > self.longest_len:
            self.longest_len = l
        if self.track_unique and self.unique is not None:
            self.unique.add(cell)
    def result(self) -> dict:
        stdev = math.sqrt(self.M2 / (self.nn - 1)) if self.nn > 1 else None
        if not self.track_unique or self.unique is None:
            uniq_est, uniq_exact = (None, True)
        else:
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

def count_total_data_rows(path: str, chunk_size: int = 8 * 1024 * 1024) -> int:
    """Exact newline count (minus header) for ETA when --eta is enabled."""
    total_newlines = 0
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            total_newlines += chunk.count(b'\n')
    return max(0, total_newlines - 1)

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

def stream_stats_filelike(fh, *, progress_every_rows: int = 0, show_progress: bool = False, total_size: Optional[int] = None, estimated_total_rows: Optional[float] = None, track_unique: bool = True):
    rdr = csv.reader(fh)
    headers = next(rdr)
    stats = [RunningStats(track_unique=track_unique) for _ in headers]
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

def stream_stats_path(path: str, *, progress_every_rows: int = 0, show_progress: bool = False, estimated_total_rows: Optional[float] = None, track_unique: bool = True) -> Tuple[List[str], List[dict]]:
    total_size = os.stat(path).st_size if show_progress else None
    with open(path, newline="", encoding="utf-8", buffering=1024*1024) as f:  # larger buffer
        return stream_stats_filelike(
            f,
            progress_every_rows=progress_every_rows,
            show_progress=show_progress,
            total_size=total_size,
            estimated_total_rows=estimated_total_rows,
            track_unique=track_unique
        )

# ---- Polars / PyArrow engines ----
def _is_polars_numeric(dtype) -> bool:  # resilient numeric dtype checker
    try:
        import polars as pl  # type: ignore
        from polars.datatypes import INTEGER_DTYPES, FLOAT_DTYPES  # type: ignore
        return (dtype in INTEGER_DTYPES) or (dtype in FLOAT_DTYPES)
    except Exception:
        s = str(dtype).lower()
        return any(x in s for x in ("int", "float", "decimal"))

def polars_stats_path(path: str, skip_unique: bool = False):  # returns headers, results
    try:
        import polars as pl  # type: ignore
    except Exception as e:
        raise RuntimeError(f"polars not available: {e}")
    lf = pl.scan_csv(path)
    try:
        schema = lf.collect_schema()
    except Exception:
        schema = lf.schema
    # Revert to per-column non-null counts (c.count()), not total row length
    exprs = []
    for col, dtype in schema.items():
        c = pl.col(col)
        exprs.append(c.count().alias(f"{col}__count"))          # non-null count only
        exprs.append(c.null_count().alias(f"{col}__nulls"))     # null count
        if not skip_unique:
            exprs.append(c.n_unique().alias(f"{col}__unique"))  # distinct non-null values
        if _is_polars_numeric(dtype):
            exprs.extend([
                c.min().alias(f"{col}__min"),
                c.max().alias(f"{col}__max"),
                c.mean().alias(f"{col}__mean"),
                c.std().alias(f"{col}__stdev"),
                c.sum().alias(f"{col}__sum"),
            ])
        else:
            exprs.append(c.cast(pl.Utf8).str.len_chars().max().alias(f"{col}__longest"))
    try:
        out = lf.select(exprs).collect(engine="streaming")
    except TypeError:
        out = lf.select(exprs).collect(streaming=True)
    headers = list(schema.keys())
    res_list = []
    cols_set = set(out.columns)
    for col, dtype in schema.items():
        # count is non-null count (legacy / potentially inaccurate vs total rows with nulls)
        count = int(out[f"{col}__count"][0]) if f"{col}__count" in cols_set else 0
        nulls = int(out[f"{col}__nulls"][0]) if f"{col}__nulls" in cols_set else 0
        unique = int(out[f"{col}__unique"][0]) if (not skip_unique and f"{col}__unique" in cols_set) else None
        if _is_polars_numeric(dtype) and f"{col}__min" in cols_set:
            res_list.append({
                "count": count,                      # non-null count only
                "non_null_numeric": count,           # since count excludes nulls
                "nulls": nulls,
                "min": out[f"{col}__min"][0],
                "max": out[f"{col}__max"][0],
                "mean": out[f"{col}__mean"][0],
                "stdev": out[f"{col}__stdev"][0],
                "sum": out[f"{col}__sum"][0],
                "max_decimals": None,
                "longest_len": None,
                "is_numeric": True,
                "unique": unique,
                "unique_exact": True,
            })
        else:
            longest_col = f"{col}__longest"
            longest_val = int(out[longest_col][0]) if longest_col in cols_set and out[longest_col][0] is not None else 0
            res_list.append({
                "count": count,
                "non_null_numeric": 0,
                "nulls": nulls,
                "min": None,
                "max": None,
                "mean": None,
                "stdev": None,
                "sum": None,
                "max_decimals": None,
                "longest_len": longest_val,
                "is_numeric": False,
                "unique": unique,
                "unique_exact": True,
            })
    return headers, res_list

def print_stream_results(headers: List[str], results: List[dict], unique_skipped: bool = False):
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
            return "(skipped)" if unique_skipped else "None"
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
            pr("Mean:", fmt_num(r['mean']))
            pr("StDev:", fmt_num(r['stdev']))
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

# Memory/time helper functions (define unconditionally once)
# (Placed early so linter sees them before use.)

def _mb(n_bytes: Optional[int]) -> str:
    if not n_bytes:
        return "N/A"
    return f"{n_bytes/1024/1024:.1f} MB"

def measure_streaming_with_memory(path: str, skip_unique: bool = False) -> Tuple[float, Optional[int], List[str], List[dict]]:
    peak_rss = 0
    stop_flag = False
    if psutil:
        proc = psutil.Process(os.getpid())
        def sampler():
            nonlocal peak_rss, stop_flag
            while not stop_flag:
                try:
                    rss = proc.memory_info().rss
                    if rss > peak_rss:
                        peak_rss = rss
                except Exception:
                    pass
                time.sleep(0.05)
        th = threading.Thread(target=sampler, daemon=True)
        th.start()
    t0 = time.time()
    headers, results = stream_stats_path(path, progress_every_rows=0, show_progress=False, estimated_total_rows=None, track_unique=not skip_unique)
    elapsed = time.time() - t0
    stop_flag = True
    if psutil:
        th.join(timeout=0.2)
        try:
            rss = psutil.Process(os.getpid()).memory_info().rss
            if rss > peak_rss:
                peak_rss = rss
        except Exception:
            pass
    return elapsed, (peak_rss if peak_rss else None), headers, results

def measure_csvstat_time_memory(path: str, timeout: float = 300.0) -> Tuple[Optional[float], Optional[int], bool]:
    """Run external csvstat capturing elapsed time and peak RSS.
    Returns (elapsed_seconds_or_None, peak_rss_bytes_or_None, dnf_flag).
    dnf_flag is True if the run exceeded timeout and was terminated.
    """
    time_tool = '/usr/bin/time' if os.path.exists('/usr/bin/time') else None
    # Try BSD -l then GNU -v variants
    for flags, pattern, scale in ((['-l'], 'maximum resident set size', 1), (['-v'], 'Maximum resident set size', 1024)):
        if not time_tool:
            break
        cmd = [time_tool] + flags + ['csvstat', path]
        try:
            start = time.time()
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            stderr_data = []
            while True:
                ret = proc.poll()
                if ret is not None:
                    break
                if (time.time() - start) > timeout:
                    # timeout -> kill
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    try:
                        _, _ = proc.communicate(timeout=1)
                    except Exception:
                        pass
                    return (None, None, True)
                time.sleep(0.05)
            _, err = proc.communicate()
            if ret == 0 and err:
                for line in err.splitlines():
                    if pattern in line:
                        nums = re.findall(r'(\d+)', line)
                        if nums:
                            rss = int(nums[0]) * scale
                            elapsed = time.time() - start
                            return (elapsed, rss, False)
        except Exception:
            continue
    # Fallback psutil-based measurement
    if psutil:
        try:
            start = time.time()
            p = subprocess.Popen(['csvstat', path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            ps_p = psutil.Process(p.pid)
            peak = 0
            while True:
                ret = p.poll()
                try:
                    rss = ps_p.memory_info().rss
                    if rss > peak:
                        peak = rss
                except Exception:
                    pass
                if ret is not None:
                    break
                if (time.time() - start) > timeout:
                    try:
                        p.kill()
                    except Exception:
                        pass
                    return (None, None, True)
                time.sleep(0.05)
            elapsed = time.time() - start
            return (elapsed, peak if peak else None, False)
        except Exception:
            pass
    # Last resort: just time it (still respect timeout)
    start = time.time()
    try:
        p2 = subprocess.Popen(['csvstat', path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        while True:
            ret = p2.poll()
            if ret is not None:
                break
            if (time.time() - start) > timeout:
                try:
                    p2.kill()
                except Exception:
                    pass
                return (None, None, True)
            time.sleep(0.05)
    except Exception:
        pass
    return (time.time() - start, None, False)

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
    ap.add_argument("--progress", action="store_true", help="Show streaming progress with % complete, rows/s, ETA (extra pre-pass for total rows)")
    ap.add_argument("--progress-every-rows", type=int, default=500_000, help="Emit progress every N rows when --progress is set (0=disable)")
    ap.add_argument("--seed", type=int, default=1337, help="Deterministic seed for sampling/benchmarks")
    ap.add_argument("--engine", choices=["python","polars"], default="python", help="Computation engine for STREAM mode (default python)")
    ap.add_argument("--skip-unique", action="store_true", help="Skip unique counting (faster, sets Unique values to '(skipped)'")
    args = ap.parse_args()

    # stdin special-case: no auto (no filesize); stream directly
    if args.file == "-":
        print(f"Mode: STREAM — stdin | reason=no size available | seed={args.seed}")
        if args.engine != "python":
            print("(Non-python engines not supported for stdin; using python engine)", file=sys.stderr)
        headers, results = stream_stats_filelike(sys.stdin, progress_every_rows=args.progress_every_rows, show_progress=args.progress, total_size=None, track_unique=not args.skip_unique)
        print_stream_results(headers, results, unique_skipped=args.skip_unique)
        print("\nNote: STREAM mode — core stats exact. Unique counts are approximate when enabled; median & frequencies omitted.", file=sys.stderr)
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
    if args.progress and args.progress_every_rows > 0:
        # exact count pass for percent + ETA
        est_total_rows = count_total_data_rows(args.file)
    if args.engine == "python":
        headers, results = stream_stats_path(
            args.file,
            progress_every_rows=args.progress_every_rows,
            show_progress=args.progress,
            estimated_total_rows=est_total_rows,
            track_unique=not args.skip_unique
        )
    elif args.engine == "polars":
        try:
            headers, results = polars_stats_path(args.file, skip_unique=args.skip_unique)
        except Exception as e:
            print(f"(polars engine failed: {e}; falling back to python)", file=sys.stderr)
            headers, results = stream_stats_path(
                args.file,
                progress_every_rows=args.progress_every_rows,
                show_progress=args.progress,
                estimated_total_rows=est_total_rows,
                track_unique=not args.skip_unique
            )
    else:  # fallback safety
        headers, results = stream_stats_path(
            args.file,
            progress_every_rows=args.progress_every_rows,
            show_progress=args.progress,
            estimated_total_rows=est_total_rows,
            track_unique=not args.skip_unique
        )
    print_stream_results(headers, results, unique_skipped=args.skip_unique)
    uniq_note = " Unique counts skipped." if args.skip_unique else " Unique counts may be approximate (rounded)."
    print(f"\nNote: STREAM mode — core stats exact (row count, numeric min/max/mean/stdev/sum).{uniq_note} Median & frequency heavy analysis removed for speed.", file=sys.stderr)

if __name__ == "__main__":
    main()
