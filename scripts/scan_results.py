#!/usr/bin/env python3
"""Scan experiment directories and print a summary table of results.

Usage:
  python scripts/scan_results.py mpnn_datamax1_v2/experiments/gign_exact     # one series
  python scripts/scan_results.py mpnn_datamax1_v2/experiments/               # all series
  python scripts/scan_results.py mpnn_datamax1_v2/experiments/gign_exact/f*  # glob
  python scripts/scan_results.py exp1/ exp2/ exp3/                           # multiple paths

Reads best_metrics.json (fast) or falls back to parsing test_metrics.csv.
Shows both val-selected and test-selected best epochs.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import yaml


def _safe_float(row, key):
    """Get float from row, returning None if missing or invalid."""
    val = row.get(key)
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _row_to_info(row):
    """Convert a CSV row dict to a metrics info dict."""
    info = {
        "epoch": int(row["epoch"]),
        "lr": _safe_float(row, "lr") or 0.0,
        "test_rmse": float(row["rmse"]),
        "test_mae": float(row["mae"]),
        "test_pearson": float(row["pearson"]),
        "test_spearman": float(row["spearman"]),
    }
    for key, info_key in [
        ("val_rmse", "val_rmse"), ("val_pearson", "val_pearson"),
        ("test2_rmse", "test2_rmse"), ("test2_pearson", "test2_pearson"),
        ("test2_spearman", "test2_spearman"),
    ]:
        val = _safe_float(row, key)
        if val is not None:
            info[info_key] = val
    return info


def _read_csv_rows(csv_path):
    """Read all rows from a test_metrics.csv file.

    Skips rows with missing/None values in required columns (rmse, epoch)
    to handle partial writes and column-count mismatches gracefully.
    """
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            # Skip rows with missing required fields
            if row.get("epoch") is None or row.get("rmse") is None:
                continue
            try:
                float(row["rmse"])
                int(row["epoch"])
            except (ValueError, TypeError):
                continue
            rows.append(row)
        return rows


def parse_csv(csv_path):
    """Parse test_metrics.csv and return best_val and best_test info dicts."""
    rows = _read_csv_rows(csv_path)
    if not rows:
        return None

    has_val = "val_rmse" in rows[0]
    last_row = rows[-1]

    # Best test row (min test rmse)
    best_test_row = min(rows, key=lambda r: float(r["rmse"]))
    best_test = _row_to_info(best_test_row)

    # Best val row (min val rmse, or same as test if no val columns)
    if has_val:
        val_rows = [r for r in rows if _safe_float(r, "val_rmse") is not None]
        best_val_row = min(val_rows, key=lambda r: float(r["val_rmse"])) if val_rows else best_test_row
    else:
        best_val_row = best_test_row
    best_val = _row_to_info(best_val_row)

    # Val-gated best test: best test RMSE among epochs where val hit a new minimum.
    # Answers: "if I only saved checkpoints when val improved, what's the best test
    # I could get?" Useful for setting epoch budget in future runs.
    val_gated = None
    if has_val:
        running_best_val = float("inf")
        best_val_gated_test = float("inf")
        val_gated_row = None
        for row in rows:
            v = _safe_float(row, "val_rmse")
            if v is None:
                continue
            if v < running_best_val:
                running_best_val = v
                t = float(row["rmse"])
                if t < best_val_gated_test:
                    best_val_gated_test = t
                    val_gated_row = row
        if val_gated_row is not None:
            val_gated = _row_to_info(val_gated_row)

    result = {
        "current_epoch": int(last_row["epoch"]),
        "best_val": best_val,
        "best_test": best_test,
    }
    if val_gated is not None:
        result["val_gated_best_test"] = val_gated
    return result


def simulate_early_stopping(rows, patience=0, max_epochs=None):
    """Simulate val-based checkpoint selection with given patience and epoch budget.

    patience=0 means no early stopping (use full epoch budget).
    max_epochs=None means no cap.
    Returns dict with epoch, test_rmse, val_rmse at the selected checkpoint,
    or None if no rows or no val data.
    """
    if not rows or "val_rmse" not in rows[0]:
        return None

    best_val = float("inf")
    selected = None
    epochs_no_improve = 0

    for row in rows:
        epoch = int(row["epoch"])
        if max_epochs and epoch > max_epochs:
            break
        val = float(row["val_rmse"])
        if val < best_val:
            best_val = val
            selected = row
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if patience > 0 and epochs_no_improve >= patience:
            break

    if selected is None:
        return None
    return _row_to_info(selected)


def parse_csv_sweep(csv_path, patience_values, epoch_values):
    """Run early-stopping sweep over a single CSV.

    Returns dict keyed by (max_epochs, patience) → info dict, where
    max_epochs=0 means no cap and patience=0 means no early stopping.
    Returns None if CSV can't be read.
    """
    rows = _read_csv_rows(csv_path)
    if not rows or "val_rmse" not in rows[0]:
        return None

    results = {}
    all_epochs = epoch_values + [0]  # 0 = no cap
    all_patience = [0] + patience_values  # 0 = no early stopping
    for e in all_epochs:
        for p in all_patience:
            me = e if e > 0 else None
            results[(e, p)] = simulate_early_stopping(rows, patience=p, max_epochs=me)
    return results


def count_params_from_checkpoint(run_dir):
    """Count learnable parameters from a saved checkpoint.

    Prefers 'num_params' stored in the checkpoint dict (set by train.py).
    Does NOT fall back to state_dict counting (includes buffers, unreliable).
    """
    for name in ["best_model_val.pt", "best_model_test.pt", "best_model.pt"]:
        pt_path = run_dir / name
        if pt_path.exists():
            try:
                import torch
                ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
                if isinstance(ckpt, dict) and "num_params" in ckpt:
                    return ckpt["num_params"]
            except Exception:
                pass
    return None


def get_runtime(run_dir):
    """Estimate total runtime and s/epoch from file timestamps.

    Uses config.yaml mtime as start, test_metrics.csv mtime as end.
    Returns (total_seconds, sec_per_epoch) or (None, None).
    """
    run_dir = Path(run_dir)
    config_path = run_dir / "config.yaml"
    csv_path = run_dir / "test_metrics.csv"
    if not config_path.exists() or not csv_path.exists():
        return None, None
    start = os.path.getmtime(config_path)
    end = os.path.getmtime(csv_path)
    elapsed = end - start
    if elapsed <= 0:
        return None, None
    return elapsed, None  # s/epoch computed later from current_epoch


def scan_run(run_dir):
    """Extract best metrics from a single run directory.

    Returns a normalized dict with keys:
        num_params, current_epoch, best_val: {...}, best_test: {...}
    """
    run_dir = Path(run_dir)

    # Fast path: best_metrics.json
    json_path = run_dir / "best_metrics.json"
    if json_path.exists():
        with open(json_path) as f:
            info = json.load(f)
        # Handle old format (flat, no best_val/best_test nesting)
        if "best_val" not in info and "best_epoch" in info:
            info = _upgrade_old_json(info)
    elif (run_dir / "test_metrics.csv").exists():
        # Slow path: parse test_metrics.csv
        info = parse_csv(run_dir / "test_metrics.csv")
        if info is not None:
            params = count_params_from_checkpoint(run_dir)
            if params is not None:
                info["num_params"] = params
        else:
            return None
    else:
        return None

    # Add runtime info
    elapsed, _ = get_runtime(run_dir)
    if elapsed is not None:
        info["runtime_s"] = elapsed
        cur_ep = info.get("current_epoch")
        if cur_ep and cur_ep > 0:
            info["s_per_epoch"] = elapsed / cur_ep

    # Read hostname and seed from config.yaml
    config_path = run_dir / "config.yaml"
    if config_path.exists():
        try:
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            info["hostname"] = cfg.get("hostname")
            info["seed"] = cfg.get("seed")
        except Exception:
            pass

    return info


def _upgrade_old_json(old):
    """Convert old flat best_metrics.json to new nested format."""
    best_val = {
        "epoch": old.get("best_epoch"),
        "lr": old.get("lr"),
        "val_rmse": old.get("val_rmse"),
        "val_mae": old.get("val_mae"),
        "val_pearson": old.get("val_pearson"),
        "val_spearman": old.get("val_spearman"),
        "test_rmse": old.get("test_rmse"),
        "test_mae": old.get("test_mae"),
        "test_pearson": old.get("test_pearson"),
        "test_spearman": old.get("test_spearman"),
    }
    return {
        "num_params": old.get("num_params"),
        "current_epoch": old.get("current_epoch"),
        "best_val": best_val,
        "best_test": best_val,  # old format only tracked val-selected
    }


def _is_run_dir(path):
    return (path / "test_metrics.csv").exists() or (path / "best_metrics.json").exists()


def find_runs(path, _depth=0, _max_depth=3):
    """Find run directories under a given path.

    Handles these layouts (up to 3 levels deep):
    - path is a run dir (has test_metrics.csv / best_metrics.json)
    - path is an experiment dir containing run subdirs (e.g. f2_lr1e4/2026-02-06_1/)
    - path is a series dir (e.g. gign_exact/) containing experiment dirs
    - path is a top-level dir (e.g. experiments/) containing series dirs
    """
    path = Path(path)

    # Level 0: path itself is a run dir
    if _is_run_dir(path):
        return [(path.parent.name, path.name, path)]

    if not path.is_dir() or _depth >= _max_depth:
        return []

    results = []

    # Level 1: path contains run subdirs directly (experiment dir)
    children = sorted(path.iterdir())
    child_runs = [(c.parent.name, c.name, c) for c in children if c.is_dir() and _is_run_dir(c)]
    if child_runs:
        return child_runs

    # Recurse into subdirs (series or top-level dir)
    for child in children:
        if child.is_dir():
            results.extend(find_runs(child, _depth=_depth + 1, _max_depth=_max_depth))

    return results


def latest_runs_only(runs):
    """Keep only the latest run per experiment (by run name, which sorts chronologically)."""
    latest = {}
    for exp_name, run_name, run_path in runs:
        if exp_name not in latest or run_name > latest[exp_name][1]:
            latest[exp_name] = (exp_name, run_name, run_path)
    return list(latest.values())


def fmt(val, width, decimals=4):
    """Format a numeric value, or return 'N/A' padded to width."""
    if val is None:
        return "N/A".rjust(width)
    if isinstance(val, int):
        return str(val).rjust(width)
    return f"{val:.{decimals}f}".rjust(width)


def fmt_params(val, width):
    if val is None:
        return "N/A".rjust(width)
    return f"{val:,}".rjust(width)


def fmt_runtime(seconds, width):
    """Format seconds as human-readable runtime (e.g. '2h31m', '45m', '12m')."""
    if seconds is None:
        return "".rjust(width)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    if hours > 0:
        return f"{hours}h{minutes:02d}m".rjust(width)
    return f"{minutes}m".rjust(width)


def fmt_spe(seconds, width):
    """Format seconds/epoch."""
    if seconds is None:
        return "".rjust(width)
    return f"{seconds:.1f}".rjust(width)


def flatten_row(info):
    """Flatten nested info dict into a flat row for display."""
    bv = info.get("best_val", {})
    bt = info.get("best_test", {})
    vg = info.get("val_gated_best_test", {})
    return {
        "name": info.get("name", ""),
        "num_params": info.get("num_params"),
        "current_epoch": info.get("current_epoch"),
        # Val-selected
        "val_epoch": bv.get("epoch"),
        "val_rmse": bv.get("val_rmse"),
        "val_test_rmse": bv.get("test_rmse"),
        "val_test_mae": bv.get("test_mae"),
        "val_test_r": bv.get("test_pearson"),
        "val_test_sprmn": bv.get("test_spearman"),
        "val_test2_rmse": bv.get("test2_rmse"),
        "val_test2_r": bv.get("test2_pearson"),
        "val_lr": bv.get("lr"),
        # Test-selected
        "tst_epoch": bt.get("epoch"),
        "tst_rmse": bt.get("test_rmse"),
        "tst_mae": bt.get("test_mae"),
        "tst_r": bt.get("test_pearson"),
        "tst_sprmn": bt.get("test_spearman"),
        "tst_lr": bt.get("lr"),
        # Val-gated best test (best test among val-improving epochs)
        "vg_epoch": vg.get("epoch"),
        "vg_rmse": vg.get("test_rmse"),
        "vg_test2_rmse": vg.get("test2_rmse"),
        "vg_test2_r": vg.get("test2_pearson"),
        # Runtime
        "runtime_s": info.get("runtime_s"),
        "s_per_epoch": info.get("s_per_epoch"),
        # Reproducibility
        "hostname": info.get("hostname"),
        "seed": info.get("seed"),
    }


def print_table(rows, sort_key):
    """Print a formatted table of results."""
    if not rows:
        print("No results found.")
        return

    flat = [flatten_row(r) for r in rows]

    # Map sort keys to flat row keys
    sort_map = {
        "test_rmse": "tst_rmse",
        "val_rmse": "val_rmse",
        "val_test_rmse": "val_test_rmse",
    }
    sk = sort_map.get(sort_key, sort_key)

    def sort_fn(r):
        v = r.get(sk)
        if v is None:
            return float("inf")
        return v
    flat.sort(key=sort_fn)

    # Check if any rows have hostname/seed info
    has_host = any(r.get("hostname") for r in flat)
    has_seed = any(r.get("seed") is not None for r in flat)

    # Check if any rows have val-gated info
    has_vg = any(r.get("vg_rmse") is not None for r in flat)
    # Check if any rows have test2 (OOD / indep) data
    has_test2 = any(r.get("val_test2_rmse") is not None for r in flat)

    # Header — val-selected first (realistic), then oracle & VG for reference
    header = (f"{'Experiment':<28} {'CurEp':>5} "
              f"{'ValEp':>5} {'ValRMSE':>8} {'RMSE':>10} {'MAE':>7} {'R':>7} {'Sprm':>6} ")
    if has_test2:
        header += f"{'|':>1} {'T2_RMSE':>8} {'T2_R':>6} "
    has_vg_test2 = has_vg and any(r.get("vg_test2_rmse") is not None for r in flat)
    if has_vg:
        header += f"{'|':>1} {'VGEp':>5} {'VG_RMSE':>9} "
        if has_vg_test2:
            header += f"{'VG_T2':>8} {'VG_T2R':>6} "
    header += (f"{'|':>1} "
               f"{'OrcEp':>5} {'OrcRMSE':>10} {'OrcR':>7} ")
    header += f"{'Params':>12} {'Time':>7} {'s/ep':>5}"
    if has_host:
        header += f" {'Host':>10}"
    if has_seed:
        header += f" {'Seed':>5}"
    print(header)
    print("-" * len(header))

    for r in flat:
        name = r["name"][:28]
        # Val-selected columns first (the numbers you should report)
        line = (f"{name:<28} {fmt(r['current_epoch'], 5, 0)} "
                f"{fmt(r['val_epoch'], 5, 0)} {fmt(r['val_rmse'], 8)} "
                f"{fmt(r['val_test_rmse'], 10)} "
                f"{fmt(r.get('val_test_mae'), 7)} "
                f"{fmt(r['val_test_r'], 7)} {fmt(r['val_test_sprmn'], 6)} ")
        if has_test2:
            line += f"{'|':>1} {fmt(r.get('val_test2_rmse'), 8)} {fmt(r.get('val_test2_r'), 6)} "
        if has_vg:
            line += f"{'|':>1} {fmt(r['vg_epoch'], 5)} {fmt(r['vg_rmse'], 9)} "
            if has_vg_test2:
                line += f"{fmt(r.get('vg_test2_rmse'), 8)} {fmt(r.get('vg_test2_r'), 6)} "
        # Oracle columns (test-selected, for reference only)
        line += (f"{'|':>1} "
                 f"{fmt(r['tst_epoch'], 5)} {fmt(r['tst_rmse'], 10)} "
                 f"{fmt(r['tst_r'], 7)} ")
        line += (f"{fmt_params(r['num_params'], 12)} "
                 f"{fmt_runtime(r.get('runtime_s'), 7)} {fmt_spe(r.get('s_per_epoch'), 5)}")
        if has_host:
            h = (r.get("hostname") or "")[:10]
            line += f" {h:>10}"
        if has_seed:
            s = r.get("seed")
            line += f" {str(s if s is not None else ''):>5}"
        print(line)


def print_csv(rows, sort_key):
    """Print CSV output."""
    if not rows:
        return

    flat = [flatten_row(r) for r in rows]

    sort_map = {
        "test_rmse": "val_test_rmse",
        "val_rmse": "val_rmse",
        "best_test_rmse": "tst_rmse",
    }
    sk = sort_map.get(sort_key, sort_key)

    def sort_fn(r):
        v = r.get(sk)
        if v is None:
            return float("inf")
        return v
    flat.sort(key=sort_fn)

    fields = list(flat[0].keys())
    writer = csv.DictWriter(sys.stdout, fieldnames=fields)
    writer.writeheader()
    for r in flat:
        writer.writerow({k: r.get(k, "") for k in fields})


def print_sweep_tables(sweep_data, patience_values, epoch_values):
    """Print one patience-sweep table per epoch budget.

    sweep_data: list of (name, sweep_dict) where sweep_dict is from parse_csv_sweep.
    """
    if not sweep_data:
        return

    def _fmt_cell(info):
        if info is None:
            return f"{'N/A':>11}"
        return f"{info['test_rmse']:.3f}@{info['epoch']:<4d}"

    p_cols = [0] + patience_values
    # One table per epoch budget (including 0 = no cap)
    for e in epoch_values + [0]:
        label = "no epoch cap" if e == 0 else f"E={e}"
        print()
        print(f"=== Patience sweep, {label} (val-selected test RMSE) ===")
        header = f"{'Experiment':<28}"
        for p in p_cols:
            plabel = "no-ES" if p == 0 else f"P={p}"
            header += f" {plabel:>11}"
        print(header)
        print("-" * len(header))

        for name, sweep in sweep_data:
            line = f"{name[:28]:<28}"
            for p in p_cols:
                line += f" {_fmt_cell(sweep.get((e, p))):>11}"
            print(line)


def print_indep_table(rows, checkpoint="val_selected"):
    """Print a compact table comparing full CASF vs indep (test2) metrics.

    checkpoint: "val_selected" (best val epoch) or "val_gated" (best test among val-improving epochs).
    """
    key = "best_val" if checkpoint == "val_selected" else "val_gated_best_test"
    label = "val-selected" if checkpoint == "val_selected" else "val-gated"

    # Filter to rows with test2 data
    indep_rows = []
    for r in rows:
        ck = r.get(key, {})
        if ck.get("test2_rmse") is not None:
            indep_rows.append(r)

    if not indep_rows:
        print(f"\nNo runs with test2 (indep) data found ({label}).")
        return

    # Build flat rows
    flat = []
    for r in indep_rows:
        ck = r.get(key, {})
        flat.append({
            "name": r.get("name", ""),
            "epoch": ck.get("epoch"),
            "full_rmse": ck.get("test_rmse"),
            "full_r": ck.get("test_pearson"),
            "indep_rmse": ck.get("test2_rmse"),
            "indep_r": ck.get("test2_pearson"),
            "indep_sp": ck.get("test2_spearman"),
        })

    # Sort by full RMSE
    flat.sort(key=lambda r: r.get("full_rmse") or float("inf"))

    print()
    print(f"=== CASF-2016 Full (285) vs Independent (144) — {label} checkpoint ===")
    header = (f"{'Experiment':<40s} {'Epoch':>5} "
              f"{'Full RMSE':>10} {'Full R':>7} "
              f"{'Indep RMSE':>11} {'Indep R':>8} {'Indep Sp':>9} "
              f"{'Delta':>7}")
    print(header)
    print("-" * len(header))

    for r in flat:
        delta = (r["indep_rmse"] - r["full_rmse"]) if r["indep_rmse"] and r["full_rmse"] else None
        print(f"{r['name']:<40s} {fmt(r['epoch'], 5, 0)} "
              f"{fmt(r['full_rmse'], 10)} {fmt(r['full_r'], 7, 3)} "
              f"{fmt(r['indep_rmse'], 11)} {fmt(r['indep_r'], 8, 3)} {fmt(r['indep_sp'], 9, 3)} "
              f"{fmt(delta, 7, 3)}")

    print(f"\n{len(flat)} runs with indep data. Delta = indep - full RMSE (lower = more robust).")


def backfill_json(run_dir, info):
    """Write best_metrics.json for a run that only had CSV."""
    json_path = Path(run_dir) / "best_metrics.json"
    if json_path.exists():
        return False
    with open(json_path, "w") as f:
        json.dump(info, f, indent=2)
    return True


def main():
    parser = argparse.ArgumentParser(description="Scan experiment results")
    parser.add_argument("paths", nargs="+", help="Experiment directories or glob patterns")
    parser.add_argument("--sort", default="val_test_rmse",
                        help="Sort column: val_test_rmse (default), test_rmse, val_rmse, best_test_rmse")
    parser.add_argument("--csv", action="store_true", help="Output as CSV")
    parser.add_argument("--all", action="store_true", help="Show all runs, not just latest per experiment")
    parser.add_argument("--backfill", action="store_true", help="Write best_metrics.json for runs that only have CSV")
    parser.add_argument("--indep", action="store_true",
                        help="Show test2 (indep) comparison table for runs that have it")
    parser.add_argument("--indep-checkpoint", default="val_selected",
                        choices=["val_selected", "val_gated"],
                        help="Checkpoint selection for indep table (default: val_selected)")
    parser.add_argument("--sweep", action="store_true",
                        help="Show patience/epoch early-stopping sweep table")
    parser.add_argument("--patience-values", default="50,100,150,200,300,400",
                        help="Comma-separated patience values for sweep (default: 50,100,150,200,300,400)")
    parser.add_argument("--epoch-values", default="400,600,800,1000,1200",
                        help="Comma-separated epoch budgets for sweep (default: 400,600,800,1000,1200)")
    args = parser.parse_args()

    # Collect all run directories
    all_runs = []
    for pattern in args.paths:
        # Handle glob patterns
        p = Path(pattern)
        if "*" in pattern or "?" in pattern:
            # Use parent directory's glob
            parent = Path(pattern).parent
            glob_pat = Path(pattern).name
            if parent.exists():
                for match in sorted(parent.glob(glob_pat)):
                    all_runs.extend(find_runs(match))
        elif p.exists():
            all_runs.extend(find_runs(p))
        else:
            print(f"Warning: {pattern} does not exist", file=sys.stderr)

    if not all_runs:
        print("No experiment runs found.", file=sys.stderr)
        sys.exit(1)

    # Filter to latest runs unless --all
    if not args.all:
        all_runs = latest_runs_only(all_runs)

    # Scan each run
    results = []
    backfilled = 0
    for exp_name, run_name, run_path in all_runs:
        info = scan_run(run_path)
        if info is None:
            continue

        if args.backfill:
            if backfill_json(run_path, info):
                backfilled += 1

        if args.all:
            info["name"] = f"{exp_name}/{run_name}"
        else:
            info["name"] = exp_name

        results.append(info)

    if args.backfill and backfilled > 0:
        print(f"Backfilled {backfilled} best_metrics.json files.", file=sys.stderr)

    # Output
    if args.csv:
        print_csv(results, sort_key=args.sort)
    else:
        print_table(results, sort_key=args.sort)

    # Indep (test2) comparison table
    if args.indep:
        print_indep_table(results, checkpoint=args.indep_checkpoint)

    # Sweep tables
    if args.sweep:
        patience_values = [int(x) for x in args.patience_values.split(",")]
        epoch_values = [int(x) for x in args.epoch_values.split(",")]

        sweep_data = []
        for exp_name, run_name, run_path in all_runs:
            csv_path = Path(run_path) / "test_metrics.csv"
            if not csv_path.exists():
                print(f"Warning: {run_path} has no test_metrics.csv, skipping sweep",
                      file=sys.stderr)
                continue
            sweep = parse_csv_sweep(csv_path, patience_values, epoch_values)
            if sweep is None:
                continue
            name = f"{exp_name}/{run_name}" if args.all else exp_name
            sweep_data.append((name, sweep))

        # Sort by no-ES val-selected test RMSE
        def sweep_sort_key(item):
            info = item[1].get(("patience", 0))
            return info["test_rmse"] if info else float("inf")
        sweep_data.sort(key=sweep_sort_key)

        print_sweep_tables(sweep_data, patience_values, epoch_values)


if __name__ == "__main__":
    main()
