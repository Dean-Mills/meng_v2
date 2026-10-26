"""
Re-run and verify every PEng number reported 2026-10-19..23 (Dean: "CRITICAL - make 100 % sure").

  python -m analysis.verify_all --run --compare [--jobs 4] [--only gap,ap,tha,analysis,noise]

1. --run: re-computes every reported evaluation from the saved checkpoints and detector caches into
   outputs/verify_<YYYYMMDD>/ (originals are never touched; THA evals run on symlinked checkpoint dirs).
   Everything runs on the CPU with --seed 0: the GPU forward pass is not bit-deterministic (CUDA float
   atomics, ~1e-7, enough to flip k-means / THA in near-tie scenes), the CPU pass is, and COP-Kmeans
   restarts are seeded. The "noise" job re-runs two evaluations twice on the GPU to measure the
   run-to-run evaluation noise of the recorded (GPU) numbers.
2. --compare: compares every metric with the recorded output files (or, for the four analyses whose
   scratch outputs were lost, with the values written in docs/results.md) and writes report.md.

Training is NOT re-run (single-seed models; seed repeats are a separate task), nor are the detectors
(their caches reproduced the MEng detector swap exactly when built).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

O = Path("outputs")
PY = sys.executable
CK = {
    "3362f752": "outputs/pg_gat_finetune_sweep/3362f752", "08d0ffde": "outputs/pg_gat_arch_sweep/08d0ffde",
    "1cbce195": "outputs/train_pg_gat_hier/1cbce195", "a0c59235": "outputs/train_pg_gat_hier_aug/a0c59235",
    "030c416a": "outputs/train_pg_gat_flat_aug/030c416a", "c7b02e4a": "outputs/finetune_pg_gat_hier_coco/c7b02e4a",
    "8094ccec": "outputs/finetune_pg_gat_flat_aug_coco/8094ccec", "b59d602d": "outputs/finetune_pg_gat_hier_aug_coco/b59d602d",
    "372658c6": "outputs/finetune_pg_gat_hier_img_coco/372658c6",
    "d4422c37": "outputs/finetune_pg_gat_hier_img_crowd_coco/d4422c37",
}
KH = {
    "meng": "outputs/k_head_meng_headline_coco/a636b1c4/best.pt",
    "aug_3362f752": "outputs/k_head_aug_3362f752/758f77b8/best.pt",
    "aug_b59d602d": "outputs/k_head_aug_b59d602d/d8d101e7/best.pt",
    "aug_372658c6": "outputs/k_head_aug_372658c6/4579ce0c/best.pt",
    "aug_d4422c37": None,   # filled from outputs/k_head_aug_d4422c37/latest
}
ARR = {"hrnet": "outputs/e2e_per_image/arrays", "norefine": "outputs/e2e_per_image_norefine/arrays",
       "yolo": "outputs/e2e_cache_yolo/arrays", "rtmo": "outputs/e2e_cache_rtmo/arrays"}
APC = {"hrnet": "outputs/e2e_ap/hrnet_cache", "yolo": "outputs/e2e_ap_yolo/cache", "rtmo": "outputs/e2e_ap_rtmo/cache",
       "norefine": "outputs/e2e_ap_norefine/hrnet_cache"}


def resolve_kh():
    link = O / "k_head_aug_d4422c37" / "latest"
    if link.exists():
        KH["aug_d4422c37"] = str(O / "k_head_aug_d4422c37" / os.readlink(link) / "best.pt")


def gap_jobs():
    J = []  # (name, ckpt, k_head, arrays, letterbox, recorded peng_gap dir)
    add = lambda n, c, k, a, lb, r: J.append((n, c, k, a, lb, r))
    add("hrnet_3362f752_mengkh", "3362f752", "meng", "hrnet", False, "3362f752")
    add("hrnet_3362f752_letterbox", "3362f752", None, "hrnet", True, "3362f752_letterbox")
    add("hrnet_3362f752_khaug", "3362f752", "aug_3362f752", "hrnet", False, "3362f752_khaug")
    add("norefine_3362f752_mengkh", "3362f752", "meng", "norefine", False, "3362f752_norefine")
    for g in ("08d0ffde", "1cbce195", "c7b02e4a", "8094ccec", "b59d602d", "372658c6", "d4422c37"):
        add(f"hrnet_{g}", g, None, "hrnet", False, g)
    add("hrnet_b59d602d_letterbox", "b59d602d", None, "hrnet", True, "b59d602d_letterbox")
    add("hrnet_b59d602d_khaug", "b59d602d", "aug_b59d602d", "hrnet", False, "b59d602d_khaug")
    add("hrnet_372658c6_khaug", "372658c6", "aug_372658c6", "hrnet", False, "372658c6_khaug")
    add("norefine_372658c6_khaug", "372658c6", "aug_372658c6", "norefine", False, "norefine_372658c6_khaug")
    add("hrnet_d4422c37_khaug", "d4422c37", "aug_d4422c37", "hrnet", False, "d4422c37_khaug")
    for d in ("yolo", "rtmo"):
        add(f"{d}_3362f752_mengkh", "3362f752", "meng", d, False, f"{d}_3362f752")
        add(f"{d}_b59d602d_khaug", "b59d602d", "aug_b59d602d", d, False, f"{d}_b59d602d_khaug")
        add(f"{d}_372658c6", "372658c6", None, d, False, f"{d}_372658c6")
        add(f"{d}_372658c6_khaug", "372658c6", "aug_372658c6", d, False, f"{d}_372658c6_khaug")
        add(f"{d}_d4422c37_khaug", "d4422c37", "aug_d4422c37", d, False, f"{d}_d4422c37_khaug")
    return J


def ap_jobs():
    models = [("headline", "3362f752", "meng"), ("hier_aug", "b59d602d", "aug_b59d602d"),
              ("peng", "372658c6", "aug_372658c6"), ("crowd", "d4422c37", "aug_d4422c37")]
    rec = {("hrnet", 0.1): ["e2e_ap_peng", "e2e_ap"], ("hrnet", 0.0): ["e2e_ap_peng_t0", "e2e_ap_t0"],
           ("yolo", 0.1): ["e2e_ap_yolo"], ("yolo", 0.0): ["e2e_ap_yolo_t0"],
           ("rtmo", 0.1): ["e2e_ap_rtmo"], ("rtmo", 0.0): ["e2e_ap_rtmo_t0"]}
    J = []
    for (det, t), recs in rec.items():
        if t == 0.0:
            recs = recs + [f"e2e_ap_crowd_d4422c37_{det}_t0"]
        J.append((f"{det}_t{t:g}", APC[det], t, models, recs))
    J.append(("norefine_t0.1", APC["norefine"], 0.1, models[:1], ["e2e_ap_norefine"]))
    return J


def tha_jobs():
    return [(g, sets) for g, sets in (
        ("3362f752", ("virtual", "coco")), ("08d0ffde", ("coco",)), ("1cbce195", ("virtual", "coco")),
        ("a0c59235", ("virtual", "coco")), ("030c416a", ("virtual", "coco")), ("c7b02e4a", ("virtual", "coco")),
        ("8094ccec", ("virtual", "coco")), ("b59d602d", ("virtual", "coco")), ("372658c6", ("virtual", "coco")),
        ("d4422c37", ("virtual", "coco")))]


def sh(cmd, log, env):
    with open(log, "w") as fh:
        r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
    return r.returncode


def run(vdir: Path, only: set, jobs: int):
    resolve_kh()
    cpu = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "4", "PYTHONUNBUFFERED": "1"}
    gpu = {**os.environ, "PYTHONUNBUFFERED": "1"}
    tasks = []
    if "gap" in only:
        for n, c, k, a, lb, _ in gap_jobs():
            cmd = [PY, "eval_peng_gap.py", "--checkpoint", f"{CK[c]}/best.pt", "--arrays", ARR[a],
                   "--out_dir", str(vdir / "gap" / n), "--device", "cpu", "--seed", "0"]
            if k:
                cmd += ["--k_head", KH[k]]
            if lb:
                cmd += ["--letterbox"]
            tasks.append((f"gap/{n}", cmd, cpu))
    if "ap" in only:
        for n, cache, t, models, _ in ap_jobs():
            cmd = [PY, "eval_e2e_ap_peng.py", "--cache_dir", cache, "--conf_threshold", str(t),
                   "--out_dir", str(vdir / "ap" / n), "--device", "cpu", "--seed", "0"]
            for name, c, k in models:
                if KH[k] is not None:
                    cmd += ["--model", f"{name}={CK[c]}/best.pt,{KH[k]}"]
            tasks.append((f"ap/{n}", cmd, cpu))
    if "tha" in only:
        for g, sets in tha_jobs():
            d = vdir / "tha" / g
            d.mkdir(parents=True, exist_ok=True)
            link = d / "best.pt"
            if not link.exists():
                link.symlink_to(Path(CK[g]).resolve() / "best.pt")
            for s in sets:
                extra = (["--virtual_dir", "data/virtual", "--split", "test"] if s == "virtual" else
                         ["--coco_img_dir", "data/coco2017/val2017",
                          "--coco_ann_file", "data/coco2017/annotations/person_keypoints_val2017.json"])
                cmd = [PY, "-c", "import numpy,torch,sys,runpy; numpy.random.seed(0); torch.manual_seed(0); "
                       "sys.argv=['eval_hungarian_grouping.py']+sys.argv[1:]; runpy.run_path('eval_hungarian_grouping.py', run_name='__main__')",
                       "--checkpoint", str(link), "--device", "cpu", "--wandb_run_id", "none", *extra]
                tasks.append((f"tha/{g}_{s}", cmd, cpu))
    if "analysis" in only:
        A = vdir / "analysis"
        A.mkdir(parents=True, exist_ok=True)
        tasks += [
            ("analysis/gap_sources", [PY, "analyze_gap_sources.py", "--checkpoint", f"{CK['3362f752']}/best.pt",
                                      "--out", str(A / "gap_sources_3362f752.json"), "--device", "cpu"], cpu),
            ("analysis/k_bones", [PY, "eval_k_bones.py", "--bones_checkpoint", f"{CK['c7b02e4a']}/best.pt",
                                  "--headline", f"{CK['3362f752']}/best.pt", "--k_head", KH["meng"], "--tau", "0.5", "0.8", "0.95",
                                  "--out", str(A / "k_bones_c7b02e4a.json"), "--device", "cpu"], cpu),
            ("analysis/candidate_recall", [PY, "-m", "analysis.candidate_bone_recall", "--out", str(A / "candidate_recall.json")], cpu),
            ("analysis/piece_purity", [PY, "-m", "analysis.piece_purity", "--checkpoint", f"{CK['c7b02e4a']}/best.pt",
                                       "--out", str(A / "piece_purity.json"), "--device", "cpu"], cpu),
            ("analysis/bone_validity", [PY, "-m", "analysis.bone_validity", "--checkpoint", f"{CK['c7b02e4a']}/best.pt",
                                        "--out", str(A / "bone_validity.json"), "--device", "cpu"], cpu),
            ("analysis/shift", [PY, "-m", "analysis.shift_diagnostic", "--model", f"headline={CK['3362f752']}/best.pt",
                                "--model", f"hier={CK['c7b02e4a']}/best.pt", "--out", str(A / "shift.json"), "--device", "cpu"], cpu),
            ("analysis/crowd", [PY, "-m", "analysis.crowd_breakdown", "--coco_train",
                                "--run", f"hrnet={vdir}/gap/hrnet_372658c6_khaug", "--run", f"yolo={vdir}/gap/yolo_372658c6_khaug",
                                "--run", f"rtmo={vdir}/gap/rtmo_372658c6_khaug", "--out", str(A / "crowd.json")], cpu),
        ]
    noise = []
    if "noise" in only:
        for rep in (1, 2):
            for n, c, k in (("hrnet_3362f752_mengkh", "3362f752", "meng"), ("hrnet_372658c6_khaug", "372658c6", "aug_372658c6")):
                noise.append((f"noise/{n}_gpu{rep}", [PY, "eval_peng_gap.py", "--checkpoint", f"{CK[c]}/best.pt", "--k_head", KH[k],
                              "--out_dir", str(vdir / "noise" / f"{n}_gpu{rep}"), "--device", "cuda", "--seed", "0"], gpu))

    (vdir / "logs").mkdir(parents=True, exist_ok=True)
    crowd_dep = [t for t in tasks if t[0] == "analysis/crowd"]
    tasks = [t for t in tasks if t[0] != "analysis/crowd"]

    def go(t):
        name, cmd, env = t
        t0 = time.time()
        rc = sh(cmd, vdir / "logs" / (name.replace("/", "__") + ".log"), env)
        print(f"{time.strftime('%H:%M:%S')} {'OK ' if rc == 0 else 'ERR'} {name} ({time.time() - t0:.0f} s)", flush=True)
        return name, rc

    with ThreadPoolExecutor(max_workers=jobs) as ex:
        results = list(ex.map(go, tasks))
    results += [go(t) for t in crowd_dep]           # needs the verified per-image files
    results += [go(t) for t in noise]               # GPU, one at a time
    json.dump(dict(results), open(vdir / "run_status.json", "w"), indent=2)


# ── comparison ────────────────────────────────────────────────────────────────

DOC = {  # values written in docs/results.md for analyses whose scratch outputs were lost
    "candidate_recall": {"coco16": {"m1": {"recall": 0.9617}, "m2": {"recall": 0.9977}, "m3": {"recall": 0.9998, "bones_mean": 57},
                                    "slots_with_true_neighbour": 84324},
                         "coco19": {"m1": {"recall": 0.9648}, "m3": {"recall": 0.9998, "bones_mean": 68},
                                    "slots_with_true_neighbour": 99270},
                         "triplet_types": {"coco16": 19, "coco19": 31}, "n_scenes": 2166},
    "piece_purity": {"0.5": {"joints_in_impure_frac": 0.0901, "singleton_frac": 0.258, "joints_per_piece": 4.18},
                     "0.8": {"joints_in_impure_frac": 0.0510, "singleton_frac": 0.290, "joints_per_piece": 3.94},
                     "0.95": {"joints_in_impure_frac": 0.0279, "singleton_frac": 0.330, "joints_per_piece": 3.70},
                     "0.99": {"joints_in_impure_frac": 0.0162, "singleton_frac": 0.389, "joints_per_piece": 3.38}},
    "bone_validity": {"e2e_detections": {"bones": 124108, "auc": 0.9988, "acc@0": 0.9884, "top1_when_true_present": 0.9930,
                                         "orphan_frac": 0.1343, "orphan_top_scored_valid": 0.0954},
                      "coco_gt_keypoints": {"bones": 175283, "auc": 0.9992, "acc@0": 0.9867, "top1_when_true_present": 0.9874,
                                            "orphan_frac": 0.0894, "orphan_top_scored_valid": 0.0544}},
    "shift": {"n_graphs": 1248,
              "GT as is": {"headline": {"knn": 0.9472, "tha": 0.9591}, "hier": {"knn": 0.9729, "tha": 0.9773}},
              "v=1": {"headline": {"knn": 0.9468, "tha": 0.9575}, "hier": {"knn": 0.9664, "tha": 0.9718}},
              "drop 20%": {"headline": {"knn": 0.9312, "tha": 0.9438}, "hier": {"knn": 0.9296, "tha": 0.9385}},
              "jitter 3px": {"headline": {"knn": 0.9463, "tha": 0.9572}, "hier": {"knn": 0.9707, "tha": 0.9754}},
              "v=1 + drop + jitter": {"headline": {"knn": 0.9283, "tha": 0.9401}, "hier": {"knn": 0.9218, "tha": 0.9334}}},
    "crowd": {"synthetic": {"train": {"2": 100, "3": 100, "4": 100, "5": 100}, "val": {"2": 25, "3": 25, "4": 25, "5": 25},
                            "test": {"2": 25, "3": 25, "4": 25, "5": 25}},
              "coco_train": {"n": 56599, "k1": 0.478, "k2_5": 0.401, "k6plus": 0.121, "k9plus": 0.049},
              "yolo": {"6-8": {"gap_true_k": 0.0164, "gap_auto": 0.0226}, "9-99": {"gap_true_k": 0.0145}},
              "rtmo": {"6-8": {"gap_true_k": 0.0182}}, "hrnet": {"9-99": {"gap_true_k": -0.0344}}},
}


def leaves(d, prefix=()):
    """Numeric leaves as (key tuple, value); key tuples, not '/'-joined strings, because metric
    names such as 'headline/pred_knn' or 'all/ae/pga' contain '/'."""
    if isinstance(d, dict):
        for k, v in d.items():
            yield from leaves(v, prefix + (str(k),))
    elif isinstance(d, (int, float)) and not isinstance(d, bool):
        yield prefix, float(d)


def get(d, path):
    for k in (path if isinstance(path, tuple) else tuple(path.split("/"))):
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def compare(vdir: Path):
    resolve_kh()
    rows, missing = [], []
    tol_doc = 6e-5   # documented values are rounded to 4 decimals

    def cmp(kind, name, metric, rec, ver, tol):
        if rec is None or ver is None:
            missing.append((kind, name, metric, rec, ver)); return
        rows.append((kind, name, metric, rec, ver, ver - rec, abs(ver - rec) > tol))

    for n, c, k, a, lb, r in gap_jobs():
        rp, vp = O / "peng_gap" / r / "summary.json", vdir / "gap" / n / "summary.json"
        if not rp.exists() or not vp.exists():
            missing.append(("gap", n, "summary.json", rp.exists(), vp.exists())); continue
        R, V = json.load(open(rp)), json.load(open(vp))
        cmp("gap", n, "n", R.get("n"), V.get("n"), 0)
        cmp("gap", n, "ae", R.get("ae"), V.get("ae"), 1e-9)
        for sel in ("oracle", "eff", "pred", "lb", "clamp", "kdet"):
            if sel in R:
                for m in ("k_mean", "knn", "copkm", "tha"):
                    cmp("gap", n, f"{sel}/{m}", R[sel].get(m), get(V, (sel, m)), 1e-9)
    for n, cache, t, models, recs in ap_jobs():
        vp = vdir / "ap" / n / "summary.json"
        if not vp.exists():
            missing.append(("ap", n, "summary.json", None, False)); continue
        V = json.load(open(vp))
        for rname in recs:
            rp = O / rname / "summary.json"
            if not rp.exists():
                missing.append(("ap", n, rname, False, True)); continue
            R = json.load(open(rp))
            if "coco" in R and "all" in R["coco"]:           # eval_e2e_ap_peng.py format
                for sub, rowsd in R["coco"].items():
                    for row, stats in rowsd.items():
                        for st in ("AP", "AP50", "AP75", "AR"):
                            cmp("ap", f"{n} [{rname}]", f"{sub}|{row}|{st}", stats.get(st), get(V, ("coco", sub, row, st)), 1e-9)
            else:                                            # eval_e2e_ap.py (Sept) format
                mp = {"native_full": "native_full", "native_pooled": "native_pooled", "pggat_pred": "headline/pred_knn"}
                for old, new in mp.items():
                    for st in ("AP", "AP50", "AP75", "AR"):
                        cmp("ap", f"{n} [{rname}]", f"all|{new}|{st}", get(R, ("coco", old, st)),
                            get(V, ("coco", "all", new, st)), 1e-9)
    for g, sets in tha_jobs():
        for s in sets:
            rp, vp = Path(CK[g]) / f"eval_tha_{s}.json", vdir / "tha" / g / f"eval_tha_{s}.json"
            if not rp.exists() or not vp.exists():
                missing.append(("tha", f"{g}_{s}", "json", rp.exists(), vp.exists())); continue
            R, V = json.load(open(rp)), json.load(open(vp))
            cmp("tha", f"{g}_{s}", "n_images", R.get("n_images"), V.get("n_images"), 0)
            for m in ("knn", "cop_kmeans", "tha"):
                cmp("tha", f"{g}_{s}", m, get(R, ("pga", m, "mean")), get(V, ("pga", m, "mean")), 1e-9)
    A = vdir / "analysis"
    for name, rec in (("gap_sources_3362f752", O / "peng_gap" / "gap_sources_3362f752.json"),
                      ("k_bones_c7b02e4a", O / "peng_gap" / "k_bones_c7b02e4a.json")):
        vp = A / f"{name}.json"
        if not rec.exists() or not vp.exists():
            missing.append(("analysis", name, "json", rec.exists(), vp.exists())); continue
        R, V = json.load(open(rec)), json.load(open(vp))
        for path, val in leaves(R):
            cmp("analysis", name, "|".join(path), val, get(V, path), 1e-9)
    for name, fname in (("candidate_recall", "candidate_recall.json"), ("piece_purity", "piece_purity.json"),
                        ("bone_validity", "bone_validity.json"), ("shift", "shift.json"), ("crowd", "crowd.json")):
        vp = A / fname
        if not vp.exists():
            missing.append(("analysis-doc", name, "json", True, False)); continue
        V = json.load(open(vp))
        for path, val in leaves(DOC[name]):
            ver = get(V, path)
            last = path[-1]
            tol = (0.5 if last == "bones_mean" else 0.005 if last == "joints_per_piece"
                   else 0 if float(val).is_integer() and val > 1
                   else 6e-4 if last in ("singleton_frac", "k1", "k2_5", "k6plus", "k9plus") else tol_doc)
            cmp("analysis-doc", name, "|".join(path), val, ver, tol)
    # GPU evaluation noise: two GPU repeats vs each other and vs the CPU canonical run
    noise = []
    for n in ("hrnet_3362f752_mengkh", "hrnet_372658c6_khaug"):
        runs = [vdir / "noise" / f"{n}_gpu{r}" / "summary.json" for r in (1, 2)] + [vdir / "gap" / n / "summary.json"]
        if all(p.exists() for p in runs):
            S = [json.load(open(p)) for p in runs]
            for sel in ("oracle", "eff", "pred", "lb", "clamp"):
                for m in ("knn", "copkm", "tha"):
                    vals = [s[sel][m] for s in S]
                    noise.append((n, f"{sel}/{m}", *vals, max(vals) - min(vals)))

    flagged = [r for r in rows if r[6]]
    L = [f"# PEng verification report ({vdir})", "",
         f"Compared {len(rows)} metrics; **{len(flagged)} differ** beyond tolerance (recorded vs re-run on CPU, seed 0); "
         f"{len(missing)} could not be compared.", "",
         "Tolerances: exact (1e-9) for recorded JSON; 6e-5 for 4-decimal documented values; COP-KM rows recorded before "
         "seeding and all recorded GPU runs carry evaluation noise - see the GPU noise table.", ""]
    L += ["## Differences beyond tolerance", "", "| kind | run | metric | recorded | verified | diff |", "|---|---|---|---:|---:|---:|"]
    L += [f"| {k} | {n} | {m} | {r:.6f} | {v:.6f} | {d:+.6f} |" for k, n, m, r, v, d, _ in flagged]
    by_kind = {}
    for k, n, m, r, v, d, f in rows:
        by_kind.setdefault(k, []).append(abs(d))
    L += ["", "## Summary by kind", "", "| kind | metrics | flagged | max abs diff |", "|---|---:|---:|---:|"]
    L += [f"| {k} | {len(v)} | {sum(1 for r in rows if r[0] == k and r[6])} | {max(v):.6f} |" for k, v in by_kind.items()]
    L += ["", "## GPU evaluation noise (two GPU runs + CPU canonical)", "", "| run | metric | GPU 1 | GPU 2 | CPU | spread |",
          "|---|---|---:|---:|---:|---:|"]
    L += [f"| {n} | {m} | {a:.4f} | {b:.4f} | {c:.4f} | {s:.4f} |" for n, m, a, b, c, s in noise]
    L += ["", "## Not compared", ""] + [f"- {x}" for x in missing]
    (vdir / "report.md").write_text("\n".join(L) + "\n")
    json.dump({"rows": rows, "missing": missing, "noise": noise}, open(vdir / "comparison.json", "w"), indent=2, default=str)
    print("\n".join(L[:4]))
    print(f"report: {vdir / 'report.md'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--vdir", type=Path, default=O / f"verify_{date.today():%Y%m%d}")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--only", default="gap,ap,tha,analysis,noise")
    a = ap.parse_args()
    a.vdir.mkdir(parents=True, exist_ok=True)
    if a.run:
        run(a.vdir, set(a.only.split(",")), a.jobs)
    if a.compare:
        compare(a.vdir)


if __name__ == "__main__":
    main()
