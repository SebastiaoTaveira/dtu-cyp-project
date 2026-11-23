#!/usr/bin/env python
"""Collect this user's runs into results/.

  results/experiments_<user>.csv   every metric, for the record
  results/summary_<user>.csv       the columns you actually compare

Per-user filenames so teammates never overwrite each other.
Select configurations on val_*. test_* is for the final table only.

    python log_results.py            update your files
    python log_results.py --table    also print a ranking of everyone's runs
    python log_results.py --push     also commit and push
"""
import csv, glob, json, os, subprocess, sys

USER = os.environ.get("USER", "unknown")
FULL = f"results/experiments_{USER}.csv"
SUMMARY = f"results/summary_{USER}.csv"
KEEP = ["timestamp", "task", "model", "who", "overrides",
        "val_macro_ST-RAE", "val_macro_MCC",
        "test_macro_ST-RAE", "test_macro_MCC", "notes"]


def flat(prefix, obj, out):
    """Turn nested metric dicts into flat prefixed columns."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            flat(f"{prefix}_{k}", v, out)
    else:
        out[prefix] = obj


def rnd(v, n=4):
    try:
        return round(float(v), n)
    except (TypeError, ValueError):
        return v


def collect():
    notes = {}
    if os.path.exists(FULL):
        with open(FULL) as f:
            for r in csv.DictReader(f):
                if r.get("notes"):
                    notes[r["run_dir"]] = r["notes"]

    rows = []
    for run_dir in sorted({os.path.dirname(p) for p in
                           glob.glob("runs/**/*_metrics.json", recursive=True)}):
        parts = run_dir.split(os.sep)
        row = {"timestamp": parts[-1],
               "task": parts[1] if len(parts) > 1 else "",
               "model": parts[2] if len(parts) > 2 else "",
               "run_dir": run_dir,
               "who": USER,
               "notes": notes.get(run_dir, "")}

        for fname, prefix in (("validation_metrics.json", "val"),
                              ("test_metrics.json", "test")):
            p = os.path.join(run_dir, fname)
            if os.path.exists(p):
                try:
                    with open(p) as f:
                        flat(prefix, json.load(f), row)
                except Exception as e:
                    row[prefix + "_error"] = str(e)

        ov = os.path.join(run_dir, ".hydra", "overrides.yaml")
        if os.path.exists(ov):
            with open(ov) as f:
                row["overrides"] = " ".join(l.strip()[2:] for l in f
                                            if l.strip().startswith("- "))
        rows.append(row)
    return rows


def write(rows):
    os.makedirs("results", exist_ok=True)
    front = ["timestamp", "task", "model", "who", "notes", "overrides"]
    fields = front + sorted({k for r in rows for k in r} - set(front))
    with open(FULL, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)

    with open(SUMMARY, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=KEEP, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r.get("task", ""), r.get("timestamp", ""))):
            w.writerow({k: rnd(r.get(k, "")) for k in KEEP})

    print(f"{FULL}: {len(rows)} runs")
    print(f"{SUMMARY}: {len(KEEP)} columns")


def table():
    """Rank every summary file in results/ — yours and your teammates'."""
    rows = []
    for path in sorted(glob.glob("results/summary_*.csv")):
        with open(path) as f:
            rows += list(csv.DictReader(f))

    for task, metric, better in (("regression", "val_macro_ST-RAE", "lower"),
                                 ("classification", "val_macro_MCC", "higher")):
        sel = [r for r in rows if r.get("task") == task and r.get(metric)]
        sel.sort(key=lambda r: float(r[metric]), reverse=(better == "higher"))
        print(f"\n{task}  ({metric}, {better} is better)")
        print(f"  {'model':<20} {'score':>8}  {'who':<9} overrides")
        for r in sel:
            print(f"  {r['model'][:20]:<20} {float(r[metric]):8.4f}  "
                  f"{r.get('who', '')[:9]:<9} {r.get('overrides', '')[:70]}")
        if not sel:
            print("  (nothing yet)")


if __name__ == "__main__":
    rows = collect()
    write(rows)
    if "--table" in sys.argv:
        table()
    if "--push" in sys.argv:
        for cmd in (["git", "add", FULL, SUMMARY],
                    ["git", "commit", "-m", f"Results from {USER}: {len(rows)} runs"],
                    ["git", "push"]):
            subprocess.run(cmd, check=False)
