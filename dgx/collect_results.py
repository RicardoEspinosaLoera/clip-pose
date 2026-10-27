"""
collect_results.py — one table over every run in --runs: synthetic test metrics (group "all" of
<run>/test/summary.json) and real-frame metrics (<run>/real/summary.json), plus per-subgroup CSVs.

Usage: python collect_results.py --runs pose_train/runs --out pose_train/runs/results.md
"""

import os
import csv
import json
import glob
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--real-sub", default="real", help="<run>/<sub>/summary.json with the real-frame metrics (real, real_v2, ...)")
    a = ap.parse_args()
    rows, sub_rows = [], []
    for run in sorted(glob.glob(os.path.join(a.runs, "*"))):
        name = os.path.basename(run)
        cfg_p, test_p, real_p = (os.path.join(run, f) for f in ("config.json", "test/summary.json", f"{a.real_sub}/summary.json"))
        if not os.path.exists(cfg_p):
            continue
        cfg = json.load(open(cfg_p))
        cfg = cfg.get("cfg", cfg)
        row = dict(run=name, model=cfg.get("model"), n_train=cfg.get("n_train"), iters=cfg.get("iters"), seed=cfg.get("seed"),
                   suffix=cfg.get("image_suffix", ""), data=os.path.basename(str(cfg.get("data", ""))))
        if os.path.exists(test_p):
            for m in json.load(open(test_p)):
                if m["name"] == "all":
                    row.update(pnp_ok=m["pnp_ok"], kp_px=m["kp_px_med"], rot_deg=m["rot_deg_med"], rot_p90=m["rot_deg_p90"],
                               trans=m["trans_med"], depth=m["depth_med"], add_s=m["add_s_frac_med"], add_s_lt_01=m["add_s_lt_0.1"],
                               rot_lt_5=m["rot_lt_5"], reproj_px=m["reproj_px_med"])
                sub_rows.append(dict(run=name, split="test", **m))
        if os.path.exists(real_p):
            for m in json.load(open(real_p)):
                if m["name"] == "all":
                    row.update(real_n=m["n"], real_err_L=m["mean_frac_med"], real_pck01=m["pck_0_1"], real_pck02=m["pck_0_2"],
                               real_axis_deg=m["axis_deg_med"], real_axis_lt_10=m["axis_lt_10"],
                               real_undir=m.get("axis_undir_med"), real_flip=m.get("flip_rate"))
                sub_rows.append(dict(run=name, split="real", **m))
        rows.append(row)
    if not rows:
        print("no runs in", a.runs)
        return
    cols = ["run", "model", "n_train", "iters", "seed", "suffix", "data", "pnp_ok", "kp_px", "rot_deg", "rot_p90", "trans", "depth",
            "add_s", "add_s_lt_01", "rot_lt_5", "reproj_px", "real_n", "real_err_L", "real_pck01", "real_pck02", "real_axis_deg", "real_axis_lt_10",
            "real_undir", "real_flip"]

    def fmt(v):
        if v is None or v == "":
            return "-"
        return f"{v:.3f}" if isinstance(v, float) else str(v)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
        for r in rows:
            f.write("| " + " | ".join(fmt(r.get(c)) for c in cols) + " |\n")
    with open(os.path.splitext(a.out)[0] + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows({c: r.get(c, "") for c in cols} for r in rows)
    keys = sorted({k for r in sub_rows for k in r})
    with open(os.path.splitext(a.out)[0] + "_subgroups.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(sub_rows)
    print(open(a.out, encoding="utf-8").read())


if __name__ == "__main__":
    main()
