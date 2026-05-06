# W&B sweeps — operating manual

How to launch, monitor, stop, and post-process Weights & Biases sweeps for the
PG-GAT / SCOT / THA pipeline. Written for the dissertation hyperparameter
defence; each sweep maps to a research question (RQ).

---

## 0. Prerequisites (one-time)

```bash
pip install wandb
wandb login                         # paste API key from wandb.ai/authorize
```

The W&B project is `multiperson-pose-grouping` under entity `deanmills`. All
sweep YAMLs in `configs/sweeps/` already pin both at the top of the file. **Do
not run a sweep from a YAML that omits `project:` and `entity:`** — `wandb
sweep` will silently infer the project from the current working directory and
create the runs in the wrong project.

---

## 1. The sweep YAML — anatomy

Every sweep YAML (e.g. `configs/sweeps/pg_gat_arch.yaml`) has six sections:

```yaml
project: multiperson-pose-grouping     # MUST be set; otherwise wandb infers from cwd
entity:  deanmills

program: trainer.py                    # what the agent runs

method: bayes                          # bayes | grid | random
metric:
  name: val/pga                        # what to optimise
  goal: maximize

parameters:                            # search space — names use `__` as path separator
  sa_gat__num_layers: { values: [2, 3, 4, 5] }
  sa_gat__hidden_dim: { values: [64, 128, 256] }
  training__lr:       { distribution: log_uniform_values, min: 1.0e-4, max: 5.0e-3 }

early_terminate:                       # kill obviously-bad runs early
  type: hyperband
  min_iter: 5
  eta: 2

run_cap: 32                            # stop after this many runs

command:                               # how the agent invokes the program
  - ${env}
  - python
  - ${program}
  - --config=configs/train_sa_gat_sweep_base.yaml
  - --name=pg_gat_arch_sweep
  - --set=wandb.group=pg-gat-arch-sweep      # appears in the runs table
  - --tag=sweep
  - --tag=arch
  - --tag=meng-paper
  - ${args}                            # expands to all --param=value pairs
```

### `command:` block — the gotchas

- `${env}` expands to the current shell environment, `${program}` to the file
  in the `program:` field, `${args}` to all sweep parameters as
  `--key=value` pairs. **There is no `${individual_param}` macro** — using
  `${num_layers}` literally puts the string `${num_layers}` in the arg list,
  which is the bug we hit on 2026-05-05.
- Parameter names use `__` as a path separator (`sa_gat__num_layers`).
  trainer.py's main() converts `--sa_gat__num_layers=4` into the override
  `--set sa_gat.num_layers=4`. We use `__` rather than `.` because wandb's
  parameter-name handling is inconsistent for dotted names across
  search/method combinations.
- The `${args}` macro emits `--key=value`, not `--key value`. trainer.py
  requires the `=` form; space-separated values are ignored with a warning.
- Boolean values are coerced by trainer.py: `true`, `false` (case-insensitive)
  become Python booleans. Strings like `"none"`/`"null"` become None.
- List-valued overrides (`--set wandb.tags=[...]`) are not supported. Use the
  separate `--tag` flag (repeatable) instead — the sweep YAML lists tags
  inline as `- --tag=sweep` lines.

### Selection metric

Always select on `val/pga`. The trainer logs this as the synth val PGA. The
COCO and end-to-end PGAs are computed *after* the sweep finishes by re-running
`eval_cop_kmeans.py`, `eval_hungarian_grouping.py`, `eval_end_to_end.py` on
the winning checkpoint. Those eval scripts attach their results to the
training run's W&B summary fields (auto-detected via `wandb_run_id` saved
inside the checkpoint), so the dashboard ends up with everything in one place.

---

## 2. Launch a sweep

From `code_v3/`:

```bash
# 1. Register the sweep with W&B (returns a sweep_id)
wandb sweep configs/sweeps/pg_gat_arch.yaml
# Output:
#   wandb: View sweep at: https://wandb.ai/deanmills/multiperson-pose-grouping/sweeps/<id>
#   wandb: Run sweep agent with: wandb agent deanmills/multiperson-pose-grouping/<id>

# 2. Launch an agent (this is the loop that actually runs trainer.py)
wandb agent deanmills/multiperson-pose-grouping/<id>
```

The agent runs in the foreground. Each child training run streams to stdout.
Ctrl-C stops the agent (sweep state persists on the W&B server; resume with
the same `wandb agent` command).

### Parallel agents

To parallelise across GPUs / machines, open another shell, repeat
`wandb agent <id>`. Each agent claims its next set of hyperparameters from the
server. Bayesian search converges faster with parallel agents up to ~4–8.

### Stopping a sweep

Two distinct things:

- **Stop one agent** — Ctrl-C in its shell. Other agents continue.
- **Stop the sweep itself** — go to the sweep page in the W&B UI, click
  *"Stop sweep"*. New runs are no longer issued; running runs continue to
  completion.

### Run cap

`run_cap: N` in the YAML stops the sweep after `N` runs. Set this so you
don't accidentally burn an open-ended budget. Each of our four sweeps has a
specific cap (32 / 24 / 16 / 16) sized to the parameter space.

---

## 3. Promote the winner

After a sweep finishes, the highest-PGA run's checkpoint is one artifact
version among many. Promote it to alias `best` (or any descriptive alias):

```bash
# Inspect first
python scripts/promote_best.py --sweep <sweep_id> --dry-run
# Drop --dry-run to apply
python scripts/promote_best.py --sweep <sweep_id> --alias champion-arch
```

After this, `--run champion-arch` in any future eval script (or in
`eval_by_run.py`, when wired) resolves to the winning checkpoint.

If you want to promote across all runs of a config (not just one sweep), use
`--artifact <name>` instead of `--sweep <id>`:

```bash
python scripts/promote_best.py --artifact pg_gat_arch_sweep --alias best
```

---

## 4. Evaluate the winner on COCO / E2E

Sweeps select on synth `val/pga`. The dissertation reports COCO and E2E, so
after `promote_best.py` you run eval scripts on the winning checkpoint. The
recommended path is **`eval_by_run.py`** — it resolves an artifact alias to a
checkpoint path, downloads it locally (W&B caches), and dispatches to the
chosen eval script. No GUID hunting:

```bash
# COCO COP-Kmeans on the architecture-sweep champion
python eval_by_run.py eval_cop_kmeans \
    --run pg_gat_arch_sweep:champion-arch \
    --coco_img_dir data/coco2017/val2017 \
    --coco_ann_file data/coco2017/annotations/person_keypoints_val2017.json

# THA on the same checkpoint (synth)
python eval_by_run.py eval_hungarian_grouping \
    --run pg_gat_arch_sweep:champion-arch \
    --virtual_dir data/virtual

# End-to-end with HigherHRNet detection
python eval_by_run.py eval_end_to_end \
    --run pg_gat_arch_sweep:champion-arch \
    --coco_img_dir data/coco2017/val2017 \
    --coco_ann_file data/coco2017/annotations/person_keypoints_val2017.json
```

Each call prints the resolved checkpoint path, runs the eval, then prints
`W&B summary updated: <url>`. The URL points at the training run; opening it
shows the new `eval/cop_kmeans/coco/*`, `eval/tha/coco/*`, `eval/e2e/*`
summary fields alongside the original training metrics. One-stop view for
the dissertation results table.

`eval_by_run.py` accepts any of:

- `<artifact>:champion-arch` (or any alias set by `promote_best.py`)
- `<artifact>:latest`
- `<artifact>:v17` (specific version)

If you still want to point at an on-disk checkpoint directly, the underlying
eval scripts (`eval_cop_kmeans.py`, `eval_hungarian_grouping.py`,
`eval_end_to_end.py`, `evaluator.py`) all accept `--checkpoint <path>` as
before — `eval_by_run.py` is a convenience wrapper, not a replacement.

---

## 5. Workflow for the dissertation

The four planned sweeps and how they chain:

| Sweep | YAML | Defends RQ | Budget |
|---|---|---|---|
| 1. Architecture | `configs/sweeps/pg_gat_arch.yaml` | RQ2 (which mods help) | 32 runs |
| 2. Loss / regularisation | `configs/sweeps/pg_gat_loss.yaml` | RQ2 (loss coefficients) | 24 runs |
| 3. COCO fine-tune | `configs/sweeps/pg_gat_finetune.yaml` | RQ6 (fine-tune lift) | 16 runs |
| 4. Hyperbolic dim | (deferred — single-run ablation already exists) | RQ4 | n/a |

After sweep 1 finishes:
- Promote winner: `promote_best.py --sweep <id> --alias champion-arch`
- Edit `configs/sweeps/pg_gat_loss.yaml`'s `--set sa_gat.*` lines to use the
  winning architecture (the file currently has the manual default 2/4/64/128).

After sweep 2 finishes:
- Promote winner: `promote_best.py --sweep <id> --alias champion-loss`
- Update `pg_gat_finetune.yaml` to use the joint winner.

After sweep 3 finishes:
- Promote: `promote_best.py --sweep <id> --alias meng-headline`
- Run all four eval scripts on the meng-headline checkpoint.
- That run's summary section is the dissertation Ch4 results table.

---

## 6. Common gotchas

| Symptom | Likely cause | Fix |
|---|---|---|
| Sweep created in wrong project (`mills_ds-code_v3` instead of `multiperson-pose-grouping`) | YAML missing `project:` and `entity:` at top | Add both; delete the misplaced sweep + project from UI |
| Agent runs immediately fail with "Input should be a valid integer, unable to parse string as an integer [input_value='${num_layers}']" | `command:` uses `${individual_param}` syntax which is not a real wandb macro | Replace per-parameter `${var}` with the single `${args}` macro at the end of the command list. Parameter names use `__` as path separator. |
| Trainer error: "Bad --set value" | Override has no `=`, or value is a list literal | Lists go through `--tag`, not `--set` |
| Eval script does not attach to wandb | Checkpoint predates W&B integration (no `wandb_run_id` field), or trained with `cfg.wandb = None` | Pass `--wandb_run_id <id>` explicitly, or train a fresh checkpoint |
| `latest` artifact is *worse* than a previous version | Most recent run was worse than an earlier run (cross-run case) | Run `promote_best.py --artifact <name>` to set `best` on the actual winner |

---

## 7. Quick mini-sweep for verifying the launch path

`configs/sweeps/pg_gat_arch_mini.yaml` is a 3-run / 3-epoch smoke test.
Useful for sanity-checking the sweep mechanism before committing to the real
overnight job.

```bash
wandb sweep configs/sweeps/pg_gat_arch_mini.yaml
wandb agent deanmills/multiperson-pose-grouping/<id>
# ~1-2 minutes total. Verify in dashboard:
#   - 3 runs under group `pg-gat-arch-sweep-mini`
#   - SWEEP column populated
#   - Each run has tags ["sweep", "arch", "mini-test"]
#   - Config tab shows different num_layers / hidden_dim across the 3 runs
```

Re-run this whenever the sweep YAML structure changes. It catches the
project-name issue, the `${var}` substitution issue, and the tag-merge issue
in one minute of total compute.
