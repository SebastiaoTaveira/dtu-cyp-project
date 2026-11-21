#!/usr/bin/env python

"""Collect every finished run into results/experiments.csv.

Safe to run as often as you like: it rebuilds the file from runs/,
so there are never duplicates, and it preserves any notes you type.

    python log_results.py           # update the file
    python log_results.py --push    # update, commit and push to GitHub
"""
import csv, glob, json, os, subprocess, sys

NOTES = {}
EXISTING = "results/experiments.csv"
if os.path.exists(EXISTING):
    with open(EXISTING) as f:
        for r in csv.DictReader(f):
            if r.get("notes"):
                NOTES[r["run_dir"]] = r["notes"]

rows = []
for mpath in sorted(glob.glob("runs/**/test_metrics.json", recursive=True)):
    run_dir = os.path.dirname(mpath)
    parts = run_dir.split(os.sep)
    row = {
        "timestamp": parts[-1],
        "task": parts[1] if len(parts) > 1 else "",
        "model": parts[2] if len(parts) > 2 else "",
        "run_dir": run_dir,
        "notes": NOTES.get(run_dir, ""),
    }
    try:
        with open(mpath) as f:
            for k, v in json.load(f).items():
                row["test_" + k] = v
    except Exception as e:
        row["error"] = str(e)

    for cand in ("config.yaml", ".hydra/config.yaml",
                 "overrides.yaml", ".hydra/overrides.yaml"):
        p = os.path.join(run_dir, cand)
        if os.path.exists(p):
            row["config"] = open(p).read().replace("\n", " | ")[:600]
            break
    rows.append(row)

os.makedirs("results", exist_ok=True)
front = ["timestamp", "task", "model", "notes"]
fields = front + sorted({k for r in rows for k in r} - set(front))
with open(EXISTING, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
print(f"Wrote {EXISTING}: {len(rows)} runs, {len(fields)} columns")

if "--push" in sys.argv:
    subprocess.run(["git", "add", EXISTING], check=False)
    subprocess.run(["git", "commit", "-m", f"Results: {len(rows)} runs"], check=False)
    subprocess.run(["git", "push"], check=False)