#!/usr/bin/env python3
import csv, random, sys
from pathlib import Path

# Usage: python make_csv.py out.csv rows num_cols str_cols null_rate seed
# Example: python make_csv.py /tmp/mixed_1gb.csv 10_000_000 6 2 0.05 1337

def main():
    if len(sys.argv) != 7:
        print("Usage: python make_csv.py out.csv rows num_cols str_cols null_rate seed", file=sys.stderr)
        sys.exit(2)
    out, rows, nnum, nstr, null_rate, seed = (
        sys.argv[1], int(sys.argv[2].replace('_','')), int(sys.argv[3]),
        int(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6])
    )
    random.seed(seed)
    p = Path(out)
    with p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"n{i}" for i in range(nnum)] + [f"s{j}" for j in range(nstr)])
        rr = random.random
        ri = random.randint
        ru = random.uniform
        choice = random.choice
        letters = "abcdefxyz"
        for _ in range(rows):
            row = []
            for _ in range(nnum):
                if rr() < null_rate:
                    row.append("")
                else:
                    if rr() < 0.5:
                        row.append(str(ri(-10**6,10**6)))
                    else:
                        row.append(f"{ru(-1e6,1e6):.6f}")
            for _ in range(nstr):
                if rr() < null_rate:
                    row.append("")
                else:
                    k = ri(3,10)
                    row.append("".join(choice(letters) for _ in range(k)))
            w.writerow(row)

if __name__ == "__main__":
    main()
