"""Vergleicht eine Merge-Experimentdatei gegen die Median-Baseline.

Aufruf (aus implementation/, PYTHONPATH=.):
    python -m eval.exp_compare results/experiments/exp_strict_glm_<ts>.jsonl
"""
import json
import sys

import numpy as np
import pandas as pd

from eval.analyze import load_raw, point_estimates


def pe_map(pe, ml, pv):
    s = pe[(pe["model_label"] == ml) & (pe["prompt_version"] == pv)]
    return {r["answer_id"]: r["pe"] for _, r in s.iterrows()}


def main():
    exp_path = sys.argv[1]
    base = load_raw("results/raw")
    pe = point_estimates(base)
    gt = {r["answer_id"]: r["gt_points"]
          for _, r in pe[(pe["model_label"] == "ensemble") &
                         (pe["prompt_version"] == "v6_median")].iterrows()}
    med = pe_map(pe, "ensemble", "v6_median")
    glm = pe_map(pe, "main", "v5_rubrik")
    mis = pe_map(pe, "sensitivity", "v5_rubrik")
    oss = pe_map(pe, "tertiary", "v5_rubrik")

    # Experiment laden -> pe je Antwort (Median ueber Reps)
    rows = [json.loads(l) for l in open(exp_path, encoding="utf-8") if l.strip()]
    df = pd.DataFrame(rows)
    exp = {}
    for aid, g in df.groupby("answer_id"):
        preds = g.dropna(subset=["pred_points"])["pred_points"].astype(float).tolist()
        exp[aid] = float(np.median(preds)) if preds else np.nan
    tag = rows[0].get("tag", "exp")

    aids = sorted(exp)
    print(f"\n== Experiment '{tag}' ({exp_path}) -- {len(aids)} Antworten ==")
    print(f"  {'answer_id':22} {'GT':>3} {'GLM':>4} {'Mis':>4} {'OSS':>4} "
          f"{'Med':>4} {'Exp':>4}  status")
    fixed = broken = 0
    for aid in aids:
        g = gt[aid]
        me = med.get(aid, np.nan)
        ex = exp[aid]
        med_ok = round(me) == g if not np.isnan(me) else False
        exp_ok = round(ex) == g if not np.isnan(ex) else False
        status = ""
        if med_ok and not exp_ok:
            status = "BROKEN"; broken += 1
        elif not med_ok and exp_ok:
            status = "FIXED"; fixed += 1
        elif not med_ok and not exp_ok:
            status = "both-wrong"
        print(f"  {aid:22} {g:3.0f} {glm.get(aid,np.nan):4.0f} {mis.get(aid,np.nan):4.0f} "
              f"{oss.get(aid,np.nan):4.0f} {me:4.0f} {ex:4.0f}  {status}")

    print(f"\n  FIXED (Median falsch -> Exp richtig): {fixed}")
    print(f"  BROKEN (Median richtig -> Exp falsch): {broken}")
    print(f"  Netto: {fixed - broken:+d}")

    # Falls Volllauf (alle 162): Gesamt-Exaktquote
    if len(aids) >= 160:
        exp_exact = np.mean([round(exp[a]) == gt[a] for a in aids if not np.isnan(exp[a])])
        med_exact = np.mean([round(med[a]) == gt[a] for a in aids])
        exp_mae = np.mean([abs(exp[a] - gt[a]) for a in aids if not np.isnan(exp[a])])
        med_mae = np.mean([abs(med[a] - gt[a]) for a in aids])
        print(f"\n  GESAMT exakt -- Exp: {100*exp_exact:.1f}%  Median: {100*med_exact:.1f}%")
        print(f"  GESAMT MAE   -- Exp: {exp_mae:.3f}   Median: {med_mae:.3f}")

        # Hybrid: bei Modell-Uneinigkeit den Median nehmen (dort perfekt),
        # nur bei unanimer Einigkeit das (strenge) Exp-Merge anwenden.
        def agree(a):
            vals = {round(glm[a]), round(mis[a]), round(oss[a])}
            return len(vals) == 1
        hyb = {a: (exp[a] if agree(a) and not np.isnan(exp[a]) else med[a]) for a in aids}
        hyb_exact = np.mean([round(hyb[a]) == gt[a] for a in aids])
        hyb_mae = np.mean([abs(hyb[a] - gt[a]) for a in aids])
        hyb_fixed = sum(round(med[a]) != gt[a] and round(hyb[a]) == gt[a] for a in aids)
        hyb_broken = sum(round(med[a]) == gt[a] and round(hyb[a]) != gt[a] for a in aids)
        print(f"\n  HYBRID (Median bei Uneinigkeit, Exp bei Einigkeit):")
        print(f"    exakt: {100*hyb_exact:.1f}%  MAE: {hyb_mae:.3f}  "
              f"(fixed {hyb_fixed}, broken {hyb_broken}, netto {hyb_fixed-hyb_broken:+d})")


if __name__ == "__main__":
    main()
