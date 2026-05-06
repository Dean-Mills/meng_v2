"""Small helpers for attaching post-hoc evaluation results to a W&B run.

Use case: a training run has finished; its W&B run is closed but the run
record still exists. We want to evaluate that run's checkpoint on
synth/COCO/E2E and have the resulting PGA numbers show up in the same W&B
run as a summary field, so the dashboard's run table has a single source of
truth per training run.

Why summary fields instead of a new wandb.init resume?
  - resume="allow" risks step-counter collision with the original training
    run's per-epoch logging.
  - Summary fields are written via wandb.Api() — clean, additive, no
    re-initialisation. They appear in the run table and the run page's
    Overview tab.

Use case 2: locating a run from a checkpoint. trainer._save_checkpoint
stores the W&B run id in the checkpoint's `wandb_run_id` field when wandb
was enabled during training. Eval scripts read it back and attach metrics
without the user having to remember the id.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

DEFAULT_ENTITY  = "deanmills"
DEFAULT_PROJECT = "multiperson-pose-grouping"


def get_wandb_run_id_from_ckpt(ckpt_path: Path) -> Optional[str]:
    """Read the W&B run id stored at training time. Returns None if the
    checkpoint predates W&B integration or wandb was disabled."""
    import torch
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return ckpt.get("wandb_run_id")


def attach_eval_metrics(run_id: str, prefix: str, metrics: Dict[str, float],
                        *, project: str = DEFAULT_PROJECT,
                        entity:  str = DEFAULT_ENTITY) -> Optional[str]:
    """Write metrics into the named W&B run's summary fields.

    Each key in `metrics` is prefixed with `prefix/` and stored as a summary
    field on the run. Existing summary fields with the same key are
    overwritten (this is what we want — re-running an eval should update,
    not duplicate).

    Returns the run URL on success, or None if the run could not be found
    (e.g. the user passed an id that does not exist; we do not crash so a
    typo doesn't lose the eval results).
    """
    try:
        import wandb
        api = wandb.Api()
        run = api.run(f"{entity}/{project}/{run_id}")
    except Exception as e:
        print(f"[wandb] could not attach metrics to {run_id!r}: {e}")
        return None

    for key, value in metrics.items():
        run.summary[f"{prefix}/{key}"] = float(value)
    run.summary.update()
    return run.url
