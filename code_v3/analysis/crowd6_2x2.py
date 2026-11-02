"""2x2 read-out for crowd 6x vs 3x (pre-fixed criterion, results.md § Crowd emphasis; 2026-10-29). Run from code_v3: python analysis/crowd6_2x2.py"""
import json, numpy as np
O = "outputs"
G1, G2 = "ce027cda", "0e08656b"
PFX = {"hrnet": "", "yolo": "yolo_", "rtmo": "rtmo_"}
PAIRS = ["s1_c1", "s1_c2", "s2_c1", "s2_c2"]

def star_ap(x):
    return "*" if (x["diff_ci"][0] > 0 or x["diff_ci"][1] < 0) and (x["diff_ci_pct"][0] > 0 or x["diff_ci_pct"][1] < 0) else ""

def verdict(ds, d1, d2):
    same = all(d > 0 for d in ds) or all(d < 0 for d in ds)
    m = float(np.mean(ds)); thr = max(abs(d1), abs(d2))
    return m, thr, ("PASS" if same and abs(m) > thr else "fail") + ("+" if m > 0 else "-")

print("=== AP (t0, all val2017) ===")
for row in ("lb", "clamp"):
    for sub in ("all", "ge6", "1-5"):
        print(f"\n[AP {row} {sub}]  S1-C1  S1-C2  S2-C1  S2-C2 | mean | drawC drawS | verdict")
        for det in PFX:
            A = f"{O}/ap_ci/{det}_t0"
            get = lambda f: next(x for x in json.load(open(f)) if x["subset"] == sub)
            ds = [get(f"{A}/crowd6_{p}_{row}.json") for p in PAIRS]
            dS = get(f"{A}/crowd6_crowd6_draw_{row}.json")["diff"]
            dC = get(f"{A}/rep_{row}.json")["diff"] if row == "lb" or sub == "all" else float("nan")
            try: dC = get(f"{A}/rep_{row}.json")["diff"]
            except Exception: pass
            m, thr, v = verdict([d["diff"] for d in ds], dC, dS)
            cells = "  ".join(f"{d['diff']:+.4f}{star_ap(d):1s}" for d in ds)
            print(f"  {det:<6}{cells} | {m:+.4f} | {dC:+.4f} {dS:+.4f} | {v}")
        if sub == "all":
            for det in PFX:
                A = f"{O}/ap_ci/{det}_t0"
                g = [next(x for x in json.load(open(f"{A}/crowd6_gap_{t}_{row}.json")) if x["subset"] == "all") for t in ("s1", "s2")]
                c1 = next(x for x in json.load(open(f"{A}/crowd6_s1_c1_{row}.json")) if x["subset"] == "all")
                c2 = next(x for x in json.load(open(f"{A}/crowd6_s1_c2_{row}.json")) if x["subset"] == "all")
                print(f"    {det:<6} AP S1 {c1['ap_a']:.4f} S2 {g[1]['ap_b']:.4f} C1 {c1['ap_b']:.4f} C2 {c2['ap_b']:.4f}"
                      f" native {g[0]['ap_a']:.4f}  gap S1 {g[0]['diff']:+.4f}{star_ap(g[0])} S2 {g[1]['diff']:+.4f}{star_ap(g[1])}")

print("\n=== PGA (fixed scene sets, CPU) ===")
for k6, rows in (("", ["clamp:tha", "lb:tha", "eff:tha", "clamp:copkm"]), ("_k6", ["clamp:tha", "lb:tha", "eff:tha"])):
    for r in rows:
        print(f"\n[PGA {r}{' K_gt>=6' if k6 else ''}]  S1-C1  S1-C2  S2-C1  S2-C2 | mean | drawC drawS | verdict")
        for det in PFX:
            P = f"{O}/peng_gap/{PFX[det]}{G2}_khaug_cpu"
            get = lambda f: next((x for x in json.load(open(f)) if x["row"] == r), None)
            ds = [get(f"{P}/ci_{p}{k6}.json") for p in PAIRS]
            dS = get(f"{P}/ci_crowd6_draw{k6}.json")["mean"]
            c = get(f"{O}/peng_gap/{PFX[det]}b2cdc656_khaug_cpu/ci_vs_d4422c37{k6}.json")
            dC = c["mean"] if c else float("nan")
            m, thr, v = verdict([d["mean"] for d in ds], dC, dS)
            cells = "  ".join(f"{d['mean']:+.4f}{'*' if d['ci_lo'] > 0 or d['ci_hi'] < 0 else ' '}" for d in ds)
            print(f"  {det:<6}{cells} | {m:+.4f} | {dC:+.4f} {dS:+.4f} | {v}   S1 {ds[0]['value_a']:.4f} S2 {ds[2]['value_a']:.4f}")
