"""GT-keypoint -> detection shift diagnostic (2026-10-19; restored from the session scratch script).

Multi-person COCO val GT graphs (K = true people), perturbed at test time only: v forced to 1,
20 % joint dropout, 3 px jitter, all three. kNN and THA PGA for each model. Fixed seed 0.

    python -m analysis.shift_diagnostic --model headline=outputs/pg_gat_finetune_sweep/3362f752/best.pt \
        --model hier=outputs/finetune_pg_gat_hier_coco/c7b02e4a/best.pt --out <json>
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Data
from coco_adapter import CocoAdapter
from dataloader import create_dataloader
from dataset import PoseDataset
from eval_hungarian_grouping import predict_tha
from eval_peng_gap import load_models
from evaluator import compute_pga, predict_knn

CONDS = [("GT as is", False, 0.0, 0), ("v=1", True, 0.0, 0), ("drop 20%", False, 0.2, 0),
         ("jitter 3px", False, 0.0, 3), ("v=1 + drop + jitter", True, 0.2, 3)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, help="NAME=CHECKPOINT")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev = a.device
    models = {}
    for spec in a.model:
        name, ck = spec.split("=", 1)
        models[name] = load_models(Path(ck), None, dev)
    pre = next(iter(models.values()))[2]
    ad = CocoAdapter(img_dir=Path("data/coco2017/val2017"),
                     ann_file=Path("data/coco2017/annotations/person_keypoints_val2017.json"), use_depth=False)
    graphs = []
    for batch in create_dataloader(PoseDataset(ad), batch_size=4, shuffle=False, num_workers=4):
        graphs += [g.to(dev) for g in pre.process_batch(batch)]
    graphs = [g for g in graphs if len(torch.unique(g.person_labels)) >= 2]

    def perturb(g, v1, drop, jit, rng):
        x = g.x.clone(); n = x.size(0)
        if v1:
            x[:, 2] = 1.0
        if jit:
            x[:, :2] += torch.tensor(rng.normal(0, jit / pre.image_size, (n, 2)), dtype=x.dtype, device=dev)
        keep = torch.ones(n, dtype=torch.bool, device=dev)
        if drop:
            keep = torch.tensor(rng.random(n) >= drop, device=dev)
        if keep.sum() < 2:
            return None
        x, jt, pl = x[keep], g.joint_types[keep], g.person_labels[keep]
        return Data(x=x, edge_index=pre._knn_edges(x[:, :2], k=pre.k_neighbors), joint_types=jt, person_labels=pl)

    out = {"n_graphs": len(graphs)}
    for cname, v1, drop, jit in CONDS:
        rng = np.random.default_rng(0)
        pg = [perturb(g, v1, drop, jit, rng) for g in graphs]
        out[cname] = {}
        for mname, (gat, _, _) in models.items():
            kn, th = [], []
            with torch.no_grad():
                for g in pg:
                    if g is None:
                        continue
                    k = len(torch.unique(g.person_labels)); e = gat(g)
                    kn.append(compute_pga(predict_knn(e, k), g.person_labels))
                    th.append(compute_pga(predict_tha(e, k, g.joint_types), g.person_labels))
            out[cname][mname] = {"knn": float(np.mean(kn)), "tha": float(np.mean(th))}
        print(cname, out[cname], flush=True)
    json.dump(out, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
