# All experiment commands

Run from `code_v3/` directory.

## Output structure

Training outputs go to `outputs/{config_name}/{guid}/` with a `latest` symlink:
```
outputs/
  train_scot_no_depth/
    a1b2c3d4/
      best.pt
      final.pt
      history.json
      config.yaml        # copy of config used
      eval_results.json   # saved by evaluator
    latest -> a1b2c3d4
```

Previous frozen results are in `outputs/frozen_checkpoints/`.

To re-run a config from a previous run:
```bash
python trainer.py --config outputs/train_scot_no_depth/a1b2c3d4/config.yaml --name train_scot_no_depth
```

## Baseline — Standard GAT, contrastive only

### kNN with depth
```bash
python trainer.py --config configs/train_knn_only.yaml
python evaluator.py --checkpoint outputs/train_knn_only/latest/best.pt --virtual_dir data/virtual
python evaluator.py --checkpoint outputs/train_knn_only/latest/best.pt --coco_img_dir data/coco2017/val2017 --coco_ann_file data/coco2017/annotations/person_keypoints_val2017.json
```

### kNN no depth
```bash
python trainer.py --config configs/train_knn_no_depth.yaml
python evaluator.py --checkpoint outputs/train_knn_no_depth/latest/best.pt --virtual_dir data/virtual
python evaluator.py --checkpoint outputs/train_knn_no_depth/latest/best.pt --coco_img_dir data/coco2017/val2017 --coco_ann_file data/coco2017/annotations/person_keypoints_val2017.json
```


## Vanilla DMoN — no depth
```bash
python trainer.py --config configs/train_dmon_no_depth.yaml
python evaluator.py --checkpoint outputs/train_dmon_no_depth/latest/best.pt --virtual_dir data/virtual
```


## SA-DMoN v1 — skeleton-aware null model experiments

### Experiment 4 — learnable sigma (diverged)
```bash
python trainer.py --config configs/train_sa_dmon_no_depth.yaml
python evaluator.py --checkpoint outputs/train_sa_dmon_no_depth/latest/best.pt --virtual_dir data/virtual
```

### Experiment 5 — sigma clamped [0.05, 0.5]
```bash
python trainer.py --config configs/train_sa_dmon_no_depth_clamped.yaml
python evaluator.py --checkpoint outputs/train_sa_dmon_no_depth_clamped/latest/best.pt --virtual_dir data/virtual
```

### Experiment 6 — sigma clamped, lambda_spectral 1.0
```bash
python trainer.py --config configs/train_sa_dmon_no_depth_lambda1.yaml
python evaluator.py --checkpoint outputs/train_sa_dmon_no_depth_lambda1/latest/best.pt --virtual_dir data/virtual
```

### Experiment 7 — sigma = median pairwise distance
```bash
python trainer.py --config configs/train_sa_dmon_no_depth_fixed_sigma.yaml
python evaluator.py --checkpoint outputs/train_sa_dmon_no_depth_fixed_sigma/latest/best.pt --virtual_dir data/virtual
```

### Experiment 8 — spectral loss removed (lambda=0)
```bash
python trainer.py --config configs/train_sa_dmon_no_depth_no_spectral.yaml
python evaluator.py --checkpoint outputs/train_sa_dmon_no_depth_no_spectral/latest/best.pt --virtual_dir data/virtual
python evaluator.py --checkpoint outputs/train_sa_dmon_no_depth_no_spectral/latest/best.pt --coco_img_dir data/coco2017/val2017 --coco_ann_file data/coco2017/annotations/person_keypoints_val2017.json
```


## SA-DMoN v2 — decoupled graphs + entropy type loss

### Feature adjacency only (lambda=1.0)
```bash
python trainer.py --config configs/train_sa_dmon_v2_feat_only.yaml
python evaluator.py --checkpoint outputs/train_sa_dmon_v2_feat_only/latest/best.pt --virtual_dir data/virtual
```

### Feature adjacency only (lambda=0.1)
```bash
python trainer.py --config configs/train_sa_dmon_v2_feat_only_low_lambda.yaml
python evaluator.py --checkpoint outputs/train_sa_dmon_v2_feat_only_low_lambda/latest/best.pt --virtual_dir data/virtual
```

### Entropy type loss only
```bash
python trainer.py --config configs/train_sa_dmon_v2_entropy_only.yaml
python evaluator.py --checkpoint outputs/train_sa_dmon_v2_entropy_only/latest/best.pt --virtual_dir data/virtual
```

### Full v2 (both fixes)
```bash
python trainer.py --config configs/train_sa_dmon_v2_full.yaml
python evaluator.py --checkpoint outputs/train_sa_dmon_v2_full/latest/best.pt --virtual_dir data/virtual
```


## SCOT — Skeleton-Constrained Optimal Transport

### SCOT no depth
```bash
python trainer.py --config configs/train_scot_no_depth.yaml
python evaluator.py --checkpoint outputs/train_scot_no_depth/latest/best.pt --virtual_dir data/virtual
python evaluator.py --checkpoint outputs/train_scot_no_depth/latest/best.pt --coco_img_dir data/coco2017/val2017 --coco_ann_file data/coco2017/annotations/person_keypoints_val2017.json
```

### SCOT with depth
```bash
python trainer.py --config configs/train_scot_depth.yaml
python evaluator.py --checkpoint outputs/train_scot_depth/latest/best.pt --virtual_dir data/virtual
python evaluator.py --checkpoint outputs/train_scot_depth/latest/best.pt --coco_img_dir data/coco2017/val2017 --coco_ann_file data/coco2017/annotations/person_keypoints_val2017.json
```

### Residual SCOT (spatial prior + learned correction)
```bash
python trainer.py --config configs/train_residual_scot_no_depth.yaml
python evaluator.py --checkpoint outputs/train_residual_scot_no_depth/latest/best.pt --virtual_dir data/virtual
python evaluator.py --checkpoint outputs/train_residual_scot_no_depth/latest/best.pt --coco_img_dir data/coco2017/val2017 --coco_ann_file data/coco2017/annotations/person_keypoints_val2017.json
```

### Adaptive SCOT (scene-predicted temperature)
```bash
python trainer.py --config configs/train_adaptive_scot_no_depth.yaml
python evaluator.py --checkpoint outputs/train_adaptive_scot_no_depth/latest/best.pt --virtual_dir data/virtual
python evaluator.py --checkpoint outputs/train_adaptive_scot_no_depth/latest/best.pt --coco_img_dir data/coco2017/val2017 --coco_ann_file data/coco2017/annotations/person_keypoints_val2017.json
```

### Unbalanced SCOT (K-free, KL-relaxed marginals)
```bash
python trainer.py --config configs/train_unbalanced_scot_no_depth.yaml
python evaluator.py --checkpoint outputs/train_unbalanced_scot_no_depth/latest/best.pt --virtual_dir data/virtual
python evaluator.py --checkpoint outputs/train_unbalanced_scot_no_depth/latest/best.pt --coco_img_dir data/coco2017/val2017 --coco_ann_file data/coco2017/annotations/person_keypoints_val2017.json
```

### Dustbin SCOT (K-free, explicit rejection)
```bash
python trainer.py --config configs/train_dustbin_scot_no_depth.yaml
python evaluator.py --checkpoint outputs/train_dustbin_scot_no_depth/latest/best.pt --virtual_dir data/virtual
python evaluator.py --checkpoint outputs/train_dustbin_scot_no_depth/latest/best.pt --coco_img_dir data/coco2017/val2017 --coco_ann_file data/coco2017/annotations/person_keypoints_val2017.json
```


## SCOT demo (qualitative)
```bash
# Random image with 2-6 people
python demo.py --checkpoint outputs/train_scot_no_depth/latest/best.pt --coco_img_dir data/coco2017/val2017 --coco_ann_file data/coco2017/annotations/person_keypoints_val2017.json

# Specific person count range
python demo.py --checkpoint outputs/train_scot_no_depth/latest/best.pt --coco_img_dir data/coco2017/val2017 --coco_ann_file data/coco2017/annotations/person_keypoints_val2017.json --min_people 3 --max_people 5
```


## Type-constrained grouping methods (inference only, no training)

All use the standard GAT contrastive checkpoint (knn_no_depth).

### COP-Kmeans
```bash
python eval_cop_kmeans.py --checkpoint outputs/train_knn_no_depth/latest/best.pt --virtual_dir data/virtual
python eval_cop_kmeans.py --checkpoint outputs/train_knn_no_depth/latest/best.pt --coco_img_dir data/coco2017/val2017 --coco_ann_file data/coco2017/annotations/person_keypoints_val2017.json
```

### Type-constrained agglomerative clustering
```bash
python eval_agglomerative.py --checkpoint outputs/train_knn_no_depth/latest/best.pt --virtual_dir data/virtual
python eval_agglomerative.py --checkpoint outputs/train_knn_no_depth/latest/best.pt --coco_img_dir data/coco2017/val2017 --coco_ann_file data/coco2017/annotations/person_keypoints_val2017.json
```

### Skeleton-aware spectral clustering
```bash
python eval_spectral.py --checkpoint outputs/train_knn_no_depth/latest/best.pt --virtual_dir data/virtual
python eval_spectral.py --checkpoint outputs/train_knn_no_depth/latest/best.pt --coco_img_dir data/coco2017/val2017 --coco_ann_file data/coco2017/annotations/person_keypoints_val2017.json
```


## SA-GAT — Skeleton-Aware GAT

### Full SA-GAT (all three modifications)
```bash
python trainer.py --config configs/train_sa_gat_full.yaml
python evaluator.py --checkpoint outputs/train_sa_gat_full/latest/best.pt --virtual_dir data/virtual
python eval_cop_kmeans.py --checkpoint outputs/train_sa_gat_full/latest/best.pt --virtual_dir data/virtual
python evaluator.py --checkpoint outputs/train_sa_gat_full/latest/best.pt --coco_img_dir data/coco2017/val2017 --coco_ann_file data/coco2017/annotations/person_keypoints_val2017.json
python eval_cop_kmeans.py --checkpoint outputs/train_sa_gat_full/latest/best.pt --coco_img_dir data/coco2017/val2017 --coco_ann_file data/coco2017/annotations/person_keypoints_val2017.json
```

### Type-pair attention only
```bash
python trainer.py --config configs/train_sa_gat_type_pair.yaml
python evaluator.py --checkpoint outputs/train_sa_gat_type_pair/latest/best.pt --virtual_dir data/virtual
python eval_cop_kmeans.py --checkpoint outputs/train_sa_gat_type_pair/latest/best.pt --virtual_dir data/virtual
python eval_cop_kmeans.py --checkpoint outputs/train_sa_gat_type_pair/latest/best.pt --coco_img_dir data/coco2017/val2017 --coco_ann_file data/coco2017/annotations/person_keypoints_val2017.json
```

### Position encoding only
```bash
python trainer.py --config configs/train_sa_gat_pos_enc.yaml
python evaluator.py --checkpoint outputs/train_sa_gat_pos_enc/latest/best.pt --virtual_dir data/virtual
python eval_cop_kmeans.py --checkpoint outputs/train_sa_gat_pos_enc/latest/best.pt --virtual_dir data/virtual
python eval_cop_kmeans.py --checkpoint outputs/train_sa_gat_pos_enc/latest/best.pt --coco_img_dir data/coco2017/val2017 --coco_ann_file data/coco2017/annotations/person_keypoints_val2017.json
```

### Repulsion heads only
```bash
python trainer.py --config configs/train_sa_gat_repulsion.yaml
python evaluator.py --checkpoint outputs/train_sa_gat_repulsion/latest/best.pt --virtual_dir data/virtual
python eval_cop_kmeans.py --checkpoint outputs/train_sa_gat_repulsion/latest/best.pt --virtual_dir data/virtual
python eval_cop_kmeans.py --checkpoint outputs/train_sa_gat_repulsion/latest/best.pt --coco_img_dir data/coco2017/val2017 --coco_ann_file data/coco2017/annotations/person_keypoints_val2017.json
```


## SA-GAT + SCOT (PEng direction)
```bash
python trainer.py --config configs/train_sa_gat_scot_no_depth.yaml
python evaluator.py --checkpoint outputs/train_sa_gat_scot_no_depth/latest/best.pt --virtual_dir data/virtual
python evaluator.py --checkpoint outputs/train_sa_gat_scot_no_depth/latest/best.pt --coco_img_dir data/coco2017/val2017 --coco_ann_file data/coco2017/annotations/person_keypoints_val2017.json
python eval_cop_kmeans.py --checkpoint outputs/train_sa_gat_scot_no_depth/latest/best.pt --virtual_dir data/virtual
python eval_cop_kmeans.py --checkpoint outputs/train_sa_gat_scot_no_depth/latest/best.pt --coco_img_dir data/coco2017/val2017 --coco_ann_file data/coco2017/annotations/person_keypoints_val2017.json
```


## K Estimation

### Eigenvalue gap (inference only, no training)
```bash
python eval_k_estimation.py --checkpoint outputs/train_sa_gat_full/latest/best.pt --virtual_dir data/virtual
python eval_k_estimation.py --checkpoint outputs/train_sa_gat_full/latest/best.pt --coco_img_dir data/coco2017/val2017 --coco_ann_file data/coco2017/annotations/person_keypoints_val2017.json
python eval_k_estimation.py --checkpoint outputs/train_knn_no_depth/latest/best.pt --virtual_dir data/virtual
```

### Learned K head — joint training (corrupts embeddings, don't use)
```bash
python trainer.py --config configs/train_sa_gat_k_head.yaml
python eval_k_estimation.py --checkpoint outputs/train_sa_gat_k_head/latest/best.pt --virtual_dir data/virtual
python eval_k_estimation.py --checkpoint outputs/train_sa_gat_k_head/latest/best.pt --coco_img_dir data/coco2017/val2017 --coco_ann_file data/coco2017/annotations/person_keypoints_val2017.json
```

### Learned K head — frozen backbone (recommended)
```bash
python train_k_head_frozen.py --checkpoint outputs/train_sa_gat_full/latest/best.pt --epochs 50
python eval_k_estimation.py --checkpoint outputs/k_head_frozen/latest/best.pt --virtual_dir data/virtual
python eval_k_estimation.py --checkpoint outputs/k_head_frozen/latest/best.pt --coco_img_dir data/coco2017/val2017 --coco_ann_file data/coco2017/annotations/person_keypoints_val2017.json
```
