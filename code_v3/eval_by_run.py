"""Resolve a W&B artifact to a local checkpoint path and dispatch to an eval script.

Replaces the manual two-step workflow:

    GUID=$(...)              # find a run id by hand
    python eval_X.py --checkpoint outputs/<config>/<GUID>/best.pt ...

with a single call:

    python eval_by_run.py eval_X --run <artifact_name>:<alias_or_version> ...

The script downloads the artifact (W&B caches subsequent calls), finds
`best.pt` inside it, then invokes the eval script with `--checkpoint <that
path>` plus any other arguments you passed. The downloaded checkpoint
already carries `wandb_run_id` in its metadata, so the eval script's
auto-detect logic attaches results to the right training run.

Examples
--------
    # COP-Kmeans on the architecture-sweep champion (COCO val)
    python eval_by_run.py eval_cop_kmeans \
        --run pg_gat_arch_sweep:champion-arch \
        --coco_img_dir data/coco2017/val2017 \
        --coco_ann_file data/coco2017/annotations/person_keypoints_val2017.json

    # THA on the most recent run of a config
    python eval_by_run.py eval_hungarian_grouping \
        --run pg_gat_arch_sweep:latest \
        --virtual_dir data/virtual

    # End-to-end on a specific version (after manual lookup in the W&B UI)
    python eval_by_run.py eval_end_to_end \
        --run pg_gat_arch_sweep:v17 \
        --coco_img_dir data/coco2017/val2017 \
        --coco_ann_file data/coco2017/annotations/person_keypoints_val2017.json \
        --max_images 50

Available eval scripts (whitelisted to catch typos):
    evaluator, eval_cop_kmeans, eval_hungarian_grouping, eval_end_to_end
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_ENTITY  = "deanmills"
DEFAULT_PROJECT = "multiperson-pose-grouping"

# Eval scripts that have wandb hooks (see project_wandb_eval_hooks.md).
ALLOWED_SCRIPTS = {
    "evaluator",
    "eval_cop_kmeans",
    "eval_hungarian_grouping",
    "eval_end_to_end",
    "eval_k_estimation",
    "eval_trigat",
}


def resolve_checkpoint(run_spec: str, *, entity: str, project: str) -> Path:
    """Download a W&B model artifact, return the path to its best.pt file.

    `run_spec` is `<artifact_name>:<alias_or_version>` — e.g.
    `pg_gat_arch_sweep:champion-arch`, `pg_gat_arch_sweep:latest`,
    `pg_gat_arch_sweep:v17`. The artifact must have type `model`.
    """
    import wandb
    if ":" not in run_spec:
        raise SystemExit(
            f"--run must be '<artifact>:<alias_or_version>' (got {run_spec!r}). "
            "Examples: pg_gat_arch_sweep:champion-arch, pg_gat_arch_sweep:latest"
        )

    artifact_name, alias = run_spec.split(":", 1)
    full = f"{entity}/{project}/{artifact_name}:{alias}"

    api = wandb.Api()
    try:
        art = api.artifact(full, type="model")
    except Exception as e:
        raise SystemExit(f"Could not resolve {full!r}: {e}")

    download_dir = Path(art.download())
    pt_files = list(download_dir.glob("*.pt"))

    if not pt_files:
        raise SystemExit(f"No .pt files inside artifact {full!r} ({download_dir})")
    # Prefer best.pt if more than one .pt is present.
    for p in pt_files:
        if p.name == "best.pt":
            return p
    return pt_files[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("script", choices=sorted(ALLOWED_SCRIPTS),
                        help="Eval script name (without .py)")
    parser.add_argument("--run", required=True,
                        help="Artifact reference '<artifact>:<alias_or_version>'")
    parser.add_argument("--entity",  default=DEFAULT_ENTITY)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    args, rest = parser.parse_known_args()

    ckpt = resolve_checkpoint(args.run, entity=args.entity, project=args.project)
    print(f"[eval_by_run] {args.run!r} → {ckpt}")

    cmd = [sys.executable, f"{args.script}.py", "--checkpoint", str(ckpt), *rest]
    print(f"[eval_by_run] running: {' '.join(cmd)}")
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
