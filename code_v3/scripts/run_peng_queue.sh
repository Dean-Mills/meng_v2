#!/usr/bin/env bash
# PEng: run the pending jobs strictly one at a time (GPU shared with other work).
# Optionally waits for an already-running job (WAIT_PID) first, then:
#   1. augmented K-head on PG-GAT-H + aug embeddings (b59d602d) + E2E gap table
#   2. #3 image evidence fine-tune (finetune_pg_gat_hier_img_coco) + evals
# Launch as a systemd user service so a VS Code restart does not kill it, e.g.
#   systemd-run --user --collect --unit=peng-queue --working-directory=$PWD \
#     --setenv=PATH=$VIRTUAL_ENV/bin:/usr/local/cuda/bin:/usr/bin:/bin --setenv=PYTHONUNBUFFERED=1 \
#     --setenv=WAIT_PID=<pid> bash scripts/run_peng_queue.sh
cd "$(dirname "$0")/.."
if [ -n "${WAIT_PID:-}" ]; then
    echo "$(date +%T) waiting for PID $WAIT_PID"
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
fi

G=b59d602d; B=outputs/finetune_pg_gat_hier_aug_coco/b59d602d; LOG=outputs/k_head_aug_${G}_e10.log
echo "$(date +%T) K-head on $G -> $LOG"
python train_k_head_aug.py --checkpoint "$B/best.pt" --name "k_head_aug_$G" --epochs 10 > "$LOG" 2>&1 \
&& python eval_peng_gap.py --checkpoint "$B/best.pt" \
       --k_head "outputs/k_head_aug_$G/$(readlink "outputs/k_head_aug_$G/latest")/best.pt" \
       --out_dir "outputs/peng_gap/${G}_khaug" >> "$LOG" 2>&1 \
&& echo PIPELINE_DONE >> "$LOG"

echo "$(date +%T) #3 image evidence -> outputs/run_pg_gat_hier_img.log"
EVAL_SYN=0 SYN_DIR=outputs/train_pg_gat_hier_aug/a0c59235 SYN_CFG=train_pg_gat_hier_aug \
    FT_CFG=finetune_pg_gat_hier_img_coco bash scripts/run_pg_gat_hier.sh > outputs/run_pg_gat_hier_img.log 2>&1
echo "$(date +%T) queue done"
