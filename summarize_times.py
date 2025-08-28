#!/usr/bin/env python3
import sys, statistics, re

# Parse /usr/bin/time -v output files and print a markdown table with medians.

def parse_metrics(path):
    wall = []
    rss = []
    with open(path) as f:
        for ln in f:
            if "Elapsed (wall clock) time" in ln:
                # format like H:MM:SS or M:SS
                t = ln.split(": ",1)[1].strip()
                parts = t.split(":")
                sec = 0.0
                if len(parts) == 3:
                    h,m,s = parts
                    sec = int(h)*3600 + int(m)*60 + float(s)
                elif len(parts) == 2:
                    m,s = parts
                    sec = int(m)*60 + float(s)
                else:
                    sec = float(parts[0])
                wall.append(sec)
            if "Maximum resident set size" in ln:
                kbytes = int(re.findall(r"(\d+)", ln)[0])
                rss.append(kbytes / 1024.0 / 1024.0)  # GB -> actually MB: kB/1024/1024
    return (statistics.median(wall) if wall else None, statistics.median(rss) if rss else None)

def main():
    if len(sys.argv) != 5:
        print("Usage: summarize_times.py base_small.txt stream_small.txt base_large.txt stream_large.txt")
        sys.exit(2)
    labels = ["Small BASE", "Small STREAM", "Large BASE", "Large STREAM"]
    rows = []
    for label, path in zip(labels, sys.argv[1:]):
        w, r = parse_metrics(path)
        rows.append((label, w, r))
    print("\n## Benchmark results (median of 3 runs)\n")
    print("| Case         | Wall time (s) | Peak Memory (MB) |")
    print("|--------------|---------------|------------------|")
    for label, w, r in rows:
        print(f"| {label:<12} | {w:.1f}         | {r:.0f}              |")
    print("\n> Replace these numbers in README once you run on your machine.\n")

if __name__ == "__main__":
    main()
