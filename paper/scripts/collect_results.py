"""Per-seed accuracies from train.py logs.

Parses every ``Run:N | ... | Accuracy X |`` line in each log (not only the
last one, unlike extract_ablation_results.py) and prints the per-seed
accuracies, mean, sample std (ddof=1), 95% t confidence interval and n.
train.py prints the population std; this script reports the sample std.

Usage:
    python scripts/collect_results.py logs/svhn_eval/A_*.txt
    python scripts/collect_results.py --format=csv logs/svhn_eval/C_*.txt
"""
import argparse
import os
import re

import numpy as np
from scipy import stats

# Search rather than match: '\r' progress output can share the line.
_RUN_RE = re.compile(r"Run:\s*(\d+)\s*\|\s*Best Loss.*?\|\s*Accuracy\s+([0-9.]+)\s*\|")


def parse_log(path):
    """Return {run_number: accuracy}. A later line for the same run (a resumed
    log) replaces the earlier one."""
    runs = {}
    with open(path, errors='replace') as f:
        for line in f:
            match = _RUN_RE.search(line)
            if match:
                runs[int(match.group(1))] = float(match.group(2))
    return runs


def summarize(accs):
    """Mean, sample std, 95% CI half-width and n of a list of accuracies."""
    a = np.asarray(accs, dtype=float)
    n = len(a)
    mean = a.mean() if n else float('nan')
    std = a.std(ddof=1) if n > 1 else float('nan')
    half = stats.t.ppf(0.975, n - 1) * std / np.sqrt(n) if n > 1 else float('nan')
    return mean, std, half, n


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('logs', nargs='+', help='train.py log files')
    parser.add_argument('--format', choices=['text', 'csv'], default='text')
    args = parser.parse_args()

    if args.format == 'csv':
        print('log,n,mean,std,ci95,per_seed')
    for path in args.logs:
        runs = parse_log(path)
        accs = [runs[k] for k in sorted(runs)]
        mean, std, half, n = summarize(accs)
        name = os.path.basename(path)
        seeds = ' '.join(f'{a:.2f}' for a in accs)
        if args.format == 'csv':
            print(f'{name},{n},{mean:.2f},{std:.2f},{half:.2f},{seeds}')
        else:
            print(f'{name:45s} n={n:2d}  {mean:6.2f} ± {std:5.2f}  (95% CI ± {half:5.2f})  [{seeds}]')


if __name__ == '__main__':
    main()
