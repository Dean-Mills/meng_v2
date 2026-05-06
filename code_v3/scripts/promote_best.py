"""Promote the highest-PGA artifact version to the `best` (or `champion`) alias.

W&B's `latest` alias auto-rotates to whatever was logged most recently. That is
fine *within* a single training run, but across multiple runs of the same
config, `latest` can demote a better checkpoint when a worse retrain ships.

This script walks every version of an artifact, picks the version whose
metadata has the highest `val_pga`, and applies a stable alias (default
`best`) to it. After the first promotion `--run best` will always resolve to
the globally-best checkpoint for that artifact.

Usage:
    # Promote the best version of `train_sa_gat_full` to alias `best`
    python scripts/promote_best.py --artifact train_sa_gat_full

    # Use a different alias (e.g. mark a specific run as the dissertation champion)
    python scripts/promote_best.py --artifact train_sa_gat_full --alias champion

    # Pick the winner of a sweep (filters to runs in the sweep, then promotes)
    python scripts/promote_best.py --sweep <sweep_id> --alias best
"""
from __future__ import annotations

import argparse
from typing import Optional

import wandb


DEFAULT_ENTITY  = "deanmills"
DEFAULT_PROJECT = "multiperson-pose-grouping"
DEFAULT_METRIC  = "val_pga"
DEFAULT_ALIAS   = "best"


def _list_artifact_versions(api: wandb.Api, entity: str, project: str,
                            artifact_name: str):
    """Yield all versions of an artifact, regardless of alias."""
    collection_path = f"{entity}/{project}/{artifact_name}"
    try:
        collection = api.artifact_collection(type_name="model",
                                             name=collection_path)
    except wandb.errors.CommError as e:
        raise SystemExit(f"Could not find artifact collection {collection_path!r}: {e}")
    yield from collection.artifacts()


def _versions_from_sweep(api: wandb.Api, entity: str, project: str,
                         sweep_id: str):
    """Yield the model artifacts logged by every run in a sweep."""
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    for run in sweep.runs:
        for art in run.logged_artifacts():
            if art.type == "model":
                yield art


def promote(entity: str, project: str, artifact: Optional[str],
            sweep: Optional[str], alias: str, metric: str,
            dry_run: bool = False) -> None:
    api = wandb.Api()

    if sweep:
        versions = list(_versions_from_sweep(api, entity, project, sweep))
        scope = f"sweep {sweep}"
    elif artifact:
        versions = list(_list_artifact_versions(api, entity, project, artifact))
        scope = f"artifact {artifact}"
    else:
        raise SystemExit("Must pass either --artifact or --sweep")

    if not versions:
        raise SystemExit(f"No model artifacts found for {scope}")

    def _score(v):
        # Metric lives in the artifact metadata as written by trainer.py.
        return v.metadata.get(metric, float("-inf"))

    print(f"Found {len(versions)} model artifact version(s) in {scope}.")
    print(f"Scoring by metadata field {metric!r}.\n")
    print(f"  {'version':<18} {'val_pga':>10}  source run")
    print(f"  {'-'*18} {'-'*10}  {'-'*32}")
    for v in sorted(versions, key=_score, reverse=True):
        run = v.logged_by()
        run_id = run.id if run else "?"
        print(f"  {v.name:<18} {_score(v):>10.4f}  {run_id}")
    print()

    winner = max(versions, key=_score)
    score  = _score(winner)
    if score == float("-inf"):
        raise SystemExit(f"No version had metadata[{metric!r}]; nothing to promote.")

    print(f"Winner: {winner.name}  (val_pga={score:.4f})")
    print(f"Action: add alias {alias!r} to {winner.name}")
    if dry_run:
        print("Dry-run set; no changes pushed.")
        return

    if alias not in winner.aliases:
        winner.aliases.append(alias)
        winner.save()
        print(f"Done. {winner.name} now carries aliases: {sorted(winner.aliases)}")
    else:
        print(f"{winner.name} already has alias {alias!r}; nothing to do.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--entity",   default=DEFAULT_ENTITY)
    parser.add_argument("--project",  default=DEFAULT_PROJECT)
    parser.add_argument("--artifact", default=None,
                        help="Artifact name (e.g. train_sa_gat_full)")
    parser.add_argument("--sweep",    default=None,
                        help="Sweep ID; promote the best run from this sweep")
    parser.add_argument("--alias",    default=DEFAULT_ALIAS,
                        help=f"Alias to apply (default: {DEFAULT_ALIAS!r})")
    parser.add_argument("--metric",   default=DEFAULT_METRIC,
                        help=f"Metadata field to score on (default: {DEFAULT_METRIC!r})")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Print the winner but do not apply the alias")
    args = parser.parse_args()
    promote(args.entity, args.project, args.artifact, args.sweep,
            args.alias, args.metric, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
