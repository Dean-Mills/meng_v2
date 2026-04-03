# All experiment commands

Run from `code_v3/` directory.

## Baseline — Standard GAT, contrastive only

### kNN with depth
```bash
python trainer.py --config configs/train_knn_only.yaml
python evaluator.py --checkpoint outputs/checkpoints/knn_only/best.pt --virtual_dir data/virtual
python evaluator.py --checkpoint outputs/checkpoints/knn_only/best.pt --coco_img_dir data/coco2017/val2017 --coco_ann_file data/coco2017/annotations/person_keypoints_val2017.json
```

### kNN no depth
```bash
python trainer.py --config configs/train_knn_no_depth.yaml
python evaluator.py --checkpoint outputs/checkpoints/knn_no_depth/best.pt --virtual_dir data/virtual
python evaluator.py --checkpoint outputs/checkpoints/knn_no_depth/best.pt --coco_img_dir data/coco2017/val2017 --coco_ann_file data/coco2017/annotations/person_keypoints_val2017.json
```


## Vanilla DMoN — no depth
```bash
python trainer.py --config configs/train_dmon_no_depth.yaml
python evaluator.py --checkpoint outputs/checkpoints/dmon_no_depth/best.pt --virtual_dir data/virtual
```


## SA-DMoN v1 — skeleton-aware null model experiments

### Experiment 4 — learnable sigma (diverged)
```bash
python trainer.py --config configs/train_sa_dmon_no_depth.yaml
python evaluator.py --checkpoint outputs/checkpoints/sa_dmon_no_depth/best.pt --virtual_dir data/virtual
```

### Experiment 5 — sigma clamped [0.05, 0.5]
```bash
python trainer.py --config configs/train_sa_dmon_no_depth_clamped.yaml
python evaluator.py --checkpoint outputs/checkpoints/sa_dmon_no_depth_clamped/best.pt --virtual_dir data/virtual
```

### Experiment 6 — sigma clamped, lambda_spectral 1.0
```bash
python trainer.py --config configs/train_sa_dmon_no_depth_lambda1.yaml
python evaluator.py --checkpoint outputs/checkpoints/sa_dmon_no_depth_lambda1/best.pt --virtual_dir data/virtual
```

### Experiment 7 — sigma = median pairwise distance
```bash
python trainer.py --config configs/train_sa_dmon_no_depth_fixed_sigma.yaml
python evaluator.py --checkpoint outputs/checkpoints/sa_dmon_no_depth_fixed_sigma/best.pt --virtual_dir data/virtual
```

### Experiment 8 — spectral loss removed (lambda=0)
```bash
python trainer.py --config configs/train_sa_dmon_no_depth_no_spectral.yaml
python evaluator.py --checkpoint outputs/checkpoints/sa_dmon_no_depth_no_spectral/best.pt --virtual_dir data/virtual
python evaluator.py --checkpoint outputs/checkpoints/sa_dmon_no_depth_no_spectral/best.pt --coco_img_dir data/coco2017/val2017 --coco_ann_file data/coco2017/annotations/person_keypoints_val2017.json
```


## SA-DMoN v2 — decoupled graphs + entropy type loss

### Feature adjacency only (lambda=1.0)
```bash
python trainer.py --config configs/train_sa_dmon_v2_feat_only.yaml
python evaluator.py --checkpoint outputs/checkpoints/sa_dmon_v2_feat_only/best.pt --virtual_dir data/virtual
```

### Feature adjacency only (lambda=0.1)
```bash
python trainer.py --config configs/train_sa_dmon_v2_feat_only_low_lambda.yaml
python evaluator.py --checkpoint outputs/checkpoints/sa_dmon_v2_feat_only_low_lambda/best.pt --virtual_dir data/virtual
```

### Entropy type loss only
```bash
python trainer.py --config configs/train_sa_dmon_v2_entropy_only.yaml
python evaluator.py --checkpoint outputs/checkpoints/sa_dmon_v2_entropy_only/best.pt --virtual_dir data/virtual
```

### Full v2 (both fixes)
```bash
python trainer.py --config configs/train_sa_dmon_v2_full.yaml
python evaluator.py --checkpoint outputs/checkpoints/sa_dmon_v2_full/best.pt --virtual_dir data/virtual
```


## SCOT — Skeleton-Constrained Optimal Transport

### SCOT no depth
```bash
python trainer.py --config configs/train_scot_no_depth.yaml
python evaluator.py --checkpoint outputs/checkpoints/scot_no_depth/best.pt --virtual_dir data/virtual
python evaluator.py --checkpoint outputs/checkpoints/scot_no_depth/best.pt --coco_img_dir data/coco2017/val2017 --coco_ann_file data/coco2017/annotations/person_keypoints_val2017.json
```

### SCOT with depth
```bash
python trainer.py --config configs/train_scot_depth.yaml
python evaluator.py --checkpoint outputs/checkpoints/scot_depth/best.pt --virtual_dir data/virtual
python evaluator.py --checkpoint outputs/checkpoints/scot_depth/best.pt --coco_img_dir data/coco2017/val2017 --coco_ann_file data/coco2017/annotations/person_keypoints_val2017.json
```

### Residual SCOT (spatial prior + learned correction)
```bash
python trainer.py --config configs/train_residual_scot_no_depth.yaml
python evaluator.py --checkpoint outputs/checkpoints/residual_scot_no_depth/best.pt --virtual_dir data/virtual
python evaluator.py --checkpoint outputs/checkpoints/residual_scot_no_depth/best.pt --coco_img_dir data/coco2017/val2017 --coco_ann_file data/coco2017/annotations/person_keypoints_val2017.json
```

### Adaptive SCOT (scene-predicted temperature)
```bash
python trainer.py --config configs/train_adaptive_scot_no_depth.yaml
python evaluator.py --checkpoint outputs/checkpoints/adaptive_scot_no_depth/best.pt --virtual_dir data/virtual
python evaluator.py --checkpoint outputs/checkpoints/adaptive_scot_no_depth/best.pt --coco_img_dir data/coco2017/val2017 --coco_ann_file data/coco2017/annotations/person_keypoints_val2017.json
```

### Unbalanced SCOT (K-free, KL-relaxed marginals)
```bash
python trainer.py --config configs/train_unbalanced_scot_no_depth.yaml
python evaluator.py --checkpoint outputs/checkpoints/unbalanced_scot_no_depth/best.pt --virtual_dir data/virtual
python evaluator.py --checkpoint outputs/checkpoints/unbalanced_scot_no_depth/best.pt --coco_img_dir data/coco2017/val2017 --coco_ann_file data/coco2017/annotations/person_keypoints_val2017.json
```

### Dustbin SCOT (K-free, explicit rejection)
```bash
python trainer.py --config configs/train_dustbin_scot_no_depth.yaml
python evaluator.py --checkpoint outputs/checkpoints/dustbin_scot_no_depth/best.pt --virtual_dir data/virtual
python evaluator.py --checkpoint outputs/checkpoints/dustbin_scot_no_depth/best.pt --coco_img_dir data/coco2017/val2017 --coco_ann_file data/coco2017/annotations/person_keypoints_val2017.json
```


## SCOT demo (qualitative)
```bash
# Random image with 2-6 people
python demo.py --checkpoint outputs/checkpoints/scot_no_depth/best.pt --coco_img_dir data/coco2017/val2017 --coco_ann_file data/coco2017/annotations/person_keypoints_val2017.json

# Specific person count range
python demo.py --checkpoint outputs/checkpoints/scot_no_depth/best.pt --coco_img_dir data/coco2017/val2017 --coco_ann_file data/coco2017/annotations/person_keypoints_val2017.json --min_people 3 --max_people 5
```


## Confidence-gated evaluation (SCOT + kNN fusion, inference only)
```bash
python eval_gated.py --checkpoint outputs/checkpoints/scot_no_depth/best.pt --virtual_dir data/virtual --threshold 0.3
python eval_gated.py --checkpoint outputs/checkpoints/scot_no_depth/best.pt --coco_img_dir data/coco2017/val2017 --coco_ann_file data/coco2017/annotations/person_keypoints_val2017.json --threshold 0.3
```


## Skeleton-aware edge classifier (Option A, for comparison)
```bash
python trainer.py --config configs/train_skeleton_edge_no_depth.yaml
python evaluator.py --checkpoint outputs/checkpoints/skeleton_edge_no_depth/best.pt --virtual_dir data/virtual
python evaluator.py --checkpoint outputs/checkpoints/skeleton_edge_no_depth/best.pt --coco_img_dir data/coco2017/val2017 --coco_ann_file data/coco2017/annotations/person_keypoints_val2017.json
```


## Type-constrained grouping methods (inference only, no training)

All use the standard GAT contrastive checkpoint (knn_no_depth).

### COP-Kmeans
```bash
python eval_cop_kmeans.py --checkpoint outputs/checkpoints/knn_no_depth/best.pt --virtual_dir data/virtual
python eval_cop_kmeans.py --checkpoint outputs/checkpoints/knn_no_depth/best.pt --coco_img_dir data/coco2017/val2017 --coco_ann_file data/coco2017/annotations/person_keypoints_val2017.json
```

### Type-constrained agglomerative clustering
```bash
python eval_agglomerative.py --checkpoint outputs/checkpoints/knn_no_depth/best.pt --virtual_dir data/virtual
python eval_agglomerative.py --checkpoint outputs/checkpoints/knn_no_depth/best.pt --coco_img_dir data/coco2017/val2017 --coco_ann_file data/coco2017/annotations/person_keypoints_val2017.json
```

### Skeleton-aware spectral clustering
```bash
python eval_spectral.py --checkpoint outputs/checkpoints/knn_no_depth/best.pt --virtual_dir data/virtual
python eval_spectral.py --checkpoint outputs/checkpoints/knn_no_depth/best.pt --coco_img_dir data/coco2017/val2017 --coco_ann_file data/coco2017/annotations/person_keypoints_val2017.json
```


## SA-GAT — Skeleton-Aware GAT

### Full SA-GAT (all three modifications)
```bash
python trainer.py --config configs/train_sa_gat_full.yaml
python evaluator.py --checkpoint outputs/checkpoints/sa_gat_full/best.pt --virtual_dir data/virtual
python eval_cop_kmeans.py --checkpoint outputs/checkpoints/sa_gat_full/best.pt --virtual_dir data/virtual
python evaluator.py --checkpoint outputs/checkpoints/sa_gat_full/best.pt --coco_img_dir data/coco2017/val2017 --coco_ann_file data/coco2017/annotations/person_keypoints_val2017.json
python eval_cop_kmeans.py --checkpoint outputs/checkpoints/sa_gat_full/best.pt --coco_img_dir data/coco2017/val2017 --coco_ann_file data/coco2017/annotations/person_keypoints_val2017.json
```

### Type-pair attention only
```bash
python trainer.py --config configs/train_sa_gat_type_pair.yaml
python evaluator.py --checkpoint outputs/checkpoints/sa_gat_type_pair/best.pt --virtual_dir data/virtual
python eval_cop_kmeans.py --checkpoint outputs/checkpoints/sa_gat_type_pair/best.pt --virtual_dir data/virtual
python eval_cop_kmeans.py --checkpoint outputs/checkpoints/sa_gat_type_pair/best.pt --coco_img_dir data/coco2017/val2017 --coco_ann_file data/coco2017/annotations/person_keypoints_val2017.json
```

### Position encoding only
```bash
python trainer.py --config configs/train_sa_gat_pos_enc.yaml
python evaluator.py --checkpoint outputs/checkpoints/sa_gat_pos_enc/best.pt --virtual_dir data/virtual
python eval_cop_kmeans.py --checkpoint outputs/checkpoints/sa_gat_pos_enc/best.pt --virtual_dir data/virtual
python eval_cop_kmeans.py --checkpoint outputs/checkpoints/sa_gat_pos_enc/best.pt --coco_img_dir data/coco2017/val2017 --coco_ann_file data/coco2017/annotations/person_keypoints_val2017.json
```

### Repulsion heads only
```bash
python trainer.py --config configs/train_sa_gat_repulsion.yaml
python evaluator.py --checkpoint outputs/checkpoints/sa_gat_repulsion/best.pt --virtual_dir data/virtual
python eval_cop_kmeans.py --checkpoint outputs/checkpoints/sa_gat_repulsion/best.pt --virtual_dir data/virtual
python eval_cop_kmeans.py --checkpoint outputs/checkpoints/sa_gat_repulsion/best.pt --coco_img_dir data/coco2017/val2017 --coco_ann_file data/coco2017/annotations/person_keypoints_val2017.json
```
