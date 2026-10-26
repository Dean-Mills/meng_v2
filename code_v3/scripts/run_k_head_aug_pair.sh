#!/usr/bin/env bash
# PEng: augmented K-heads, one at a time (headline control first, then PG-GAT-H + aug),
# each followed by the fixed-scene E2E gap table with K-head and clamp rows. From code_v3/:
#   EPOCHS=5 bash scripts/run_k_head_aug_pair.sh
# Launch outside VS Code's snap scope so a VS Code restart does not kill it:
#   systemd-run --user --unit=peng-khead --working-directory=$PWD \
#     --setenv=PATH=$VIRTUAL_ENV/bin:/usr/local/cuda/bin:/usr/bin:/bin --setenv=PYTHONUNBUFFERED=1 \
#     --setenv=EPOCHS=5 bash scripts/run_k_head_aug_pair.sh
cd "$(dirname "$0")/.."
EPOCHS=${EPOCHS:-10}
for pair in "3362f752:outputs/pg_gat_finetune_sweep/3362f752" \
            "b59d602d:outputs/finetune_pg_gat_hier_aug_coco/b59d602d"; do
    G=${pair%%:*}; B=${pair#*:}; LOG=outputs/k_head_aug_${G}_e${EPOCHS}.log
    python train_k_head_aug.py --checkpoint "$B/best.pt" --name "k_head_aug_$G" --epochs "$EPOCHS" > "$LOG" 2>&1 \
    && python eval_peng_gap.py --checkpoint "$B/best.pt" \
           --k_head "outputs/k_head_aug_$G/$(readlink "outputs/k_head_aug_$G/latest")/best.pt" \
           --out_dir "outputs/peng_gap/${G}_khaug" >> "$LOG" 2>&1 \
    && echo PIPELINE_DONE >> "$LOG"
done
