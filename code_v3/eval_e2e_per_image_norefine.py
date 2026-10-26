"""Per-image E2E dump with HigherHRNet's AE `refine` step switched off (PEng, idea 2).

The standard dump (eval_e2e_per_image.py) pools joints from HigherHRNet's
*grouped* output: tag matching plus `refine`, which adds a joint for every
person by searching the heatmap near that person's mean tag. The joint set
PG-GAT re-groups is therefore partly placed by AE grouping. With refine off,
the pooled set is the heatmap peaks above the detection threshold (each still
carrying the AE tag-matching label), i.e. the detection set without any
grouping-driven placement; the AE-vs-PG-GAT comparison stays on identical
joints.

Same CLI and output format as eval_e2e_per_image.py, so eval_peng_gap.py
re-scores it with --arrays <out_dir>/arrays. The vendor parser is wrapped at
load time, not edited.

    python eval_e2e_per_image_norefine.py \
        --checkpoint outputs/pg_gat_finetune_sweep/3362f752/best.pt \
        --k_head outputs/k_head_meng_headline_coco/a636b1c4/best.pt \
        --coco_img_dir data/coco2017/val2017 \
        --coco_ann_file data/coco2017/annotations/person_keypoints_val2017.json \
        --out_dir outputs/e2e_per_image_norefine --hrnet_device cuda --save_arrays
"""
import eval_e2e_per_image as base

_load_models = base.load_models


def load_models(*args, **kwargs):
    gat, pre, k_head, hrnet = _load_models(*args, **kwargs)
    parse = hrnet.output_parser.parse
    hrnet.output_parser.parse = lambda det, tag, adjust=True, refine=True: parse(det, tag, adjust, False)
    print("HigherHRNet parser: refine OFF")
    return gat, pre, k_head, hrnet


base.load_models = load_models

if __name__ == "__main__":
    base.main()
