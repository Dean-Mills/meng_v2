#!/usr/bin/env bash
# PEng: launch the three pending jobs at once, each as a systemd user service so that
# closing or restarting VS Code (a snap: everything started from it lives in its scope
# and is killed with it, setsid/nohup included) does not kill them. From code_v3/:
#   bash scripts/launch_peng_jobs.sh
# Watch:  systemctl --user list-units 'peng-*'      Stop one:  systemctl --user stop peng-img
#   peng-img          #3 image evidence fine-tune (finetune_pg_gat_hier_img_coco) + evals
#                     A/B baseline already scored: outputs/peng_gap/b59d602d_letterbox/
#   peng-khead-<ckpt> augmented K-head on frozen embeddings + E2E gap table with K-head rows
#                     3362f752 = headline (control), b59d602d = PG-GAT-H + aug
set -euo pipefail
cd "$(dirname "$0")/.."
VENV=/home/dean/projects/mills_ds/.venv
EPOCHS=${EPOCHS:-10}
COMMON=(--user --collect --working-directory="$PWD"
        --setenv=PATH="$VENV/bin:/usr/local/cuda/bin:/usr/bin:/bin" --setenv=VIRTUAL_ENV="$VENV"
        --setenv=PYTHONUNBUFFERED=1)

systemd-run "${COMMON[@]}" --unit=peng-img \
    --setenv=EVAL_SYN=0 --setenv=SYN_DIR=outputs/train_pg_gat_hier_aug/a0c59235 \
    --setenv=SYN_CFG=train_pg_gat_hier_aug --setenv=FT_CFG=finetune_pg_gat_hier_img_coco \
    bash -c 'bash scripts/run_pg_gat_hier.sh > outputs/run_pg_gat_hier_img.log 2>&1'

for pair in "3362f752:outputs/pg_gat_finetune_sweep/3362f752" \
            "b59d602d:outputs/finetune_pg_gat_hier_aug_coco/b59d602d"; do
    G=${pair%%:*}; B=${pair#*:}; LOG=outputs/k_head_aug_${G}_e${EPOCHS}.log
    systemd-run "${COMMON[@]}" --unit="peng-khead-$G" bash -c "
        python train_k_head_aug.py --checkpoint $B/best.pt --name k_head_aug_$G --epochs $EPOCHS > $LOG 2>&1 &&
        python eval_peng_gap.py --checkpoint $B/best.pt \
            --k_head outputs/k_head_aug_$G/\$(readlink outputs/k_head_aug_$G/latest)/best.pt \
            --out_dir outputs/peng_gap/${G}_khaug >> $LOG 2>&1 &&
        echo PIPELINE_DONE >> $LOG"
done
systemctl --user list-units 'peng-*' --no-pager
