#!/usr/bin/env bash
# PEng: hierarchical PG-GAT end to end (no W&B), from code_v3/:
#   bash scripts/run_pg_gat_hier.sh [extra trainer --set overrides for both stages...]
# 1. synthetic stage (configs/train_pg_gat_hier.yaml)
# 2. COCO fine-tune from the synthetic best.pt (configs/finetune_pg_gat_hier_coco.yaml)
# 3. evals: THA/COP-KM/kNN on synth test + COCO val GT keypoints (both stages),
#    and the fixed-scene E2E gap table (eval_peng_gap.py, no K-head: K_lb row).
# Configs / run names can be changed with SYN_CFG / FT_CFG / SYN_NAME / FT_NAME, e.g.
#   SYN_CFG=train_pg_gat_flat_aug FT_CFG=finetune_pg_gat_flat_aug_coco bash scripts/run_pg_gat_hier.sh
set -euo pipefail
cd "$(dirname "$0")/.."

SYN_CFG=${SYN_CFG:-train_pg_gat_hier}
FT_CFG=${FT_CFG:-finetune_pg_gat_hier_coco}
SYN_NAME=${SYN_NAME:-$SYN_CFG}
FT_NAME=${FT_NAME:-$FT_CFG}
SETS=()
for s in "$@"; do SETS+=(--set "$s"); done
COCO=(--coco_img_dir data/coco2017/val2017 --coco_ann_file data/coco2017/annotations/person_keypoints_val2017.json)

# SYN_DIR=outputs/<name>/<guid> reuses a finished synthetic stage (skips step 1).
if [ -n "${SYN_DIR:-}" ]; then
    SYN=$SYN_DIR
else
    python trainer.py --config "configs/$SYN_CFG.yaml" --name "$SYN_NAME" "${SETS[@]}"
    SYN=outputs/$SYN_NAME/$(readlink "outputs/$SYN_NAME/latest")
fi
echo "synthetic stage: $SYN"

python trainer.py --config "configs/$FT_CFG.yaml" --name "$FT_NAME" \
    --set training.pretrained="$SYN/best.pt" "${SETS[@]}"
FT=outputs/$FT_NAME/$(readlink "outputs/$FT_NAME/latest")
echo "fine-tune stage: $FT"

# EVAL_SYN=0 skips re-evaluating a synthetic stage that was already evaluated.
CKS=("$FT"); [ "${EVAL_SYN:-1}" = "1" ] && CKS=("$SYN" "$FT")
for CK in "${CKS[@]}"; do
    python eval_hungarian_grouping.py --checkpoint "$CK/best.pt" --virtual_dir data/virtual --split test
    python eval_hungarian_grouping.py --checkpoint "$CK/best.pt" "${COCO[@]}"
done
python eval_peng_gap.py --checkpoint "$FT/best.pt" --out_dir "outputs/peng_gap/$(basename "$FT")"
echo "done: $SYN  $FT"
