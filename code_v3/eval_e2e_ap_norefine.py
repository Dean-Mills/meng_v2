"""AP@OKS run (eval_e2e_ap.py) with HigherHRNet's AE `refine` step switched off (PEng, idea 2).

See eval_e2e_per_image_norefine.py for why: with refine off, the pooled joints are
the heatmap peaks above threshold, with no grouping-driven joint placement. AP is
the fair cross-condition metric (same 5,000 images and GT, different joint sets):
compare `pggat_pred` here against `native_full` of the standard run
(outputs/e2e_ap/results_native_full.json). Use a separate --cache_dir: the
standard cache holds refine-on detector output.

    python eval_e2e_ap_norefine.py \
        --checkpoint outputs/pg_gat_finetune_sweep/3362f752/best.pt \
        --k_head outputs/k_head_meng_headline_coco/a636b1c4/best.pt \
        --coco_img_dir data/coco2017/val2017 \
        --coco_ann_file data/coco2017/annotations/person_keypoints_val2017.json \
        --out_dir outputs/e2e_ap_norefine --cache_dir outputs/e2e_ap_norefine/hrnet_cache \
        --hrnet_device cuda --wandb_run_id none
"""
import eval_e2e_ap as base

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
