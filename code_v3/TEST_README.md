# Test Suite

## Running Tests

All tests are run from `code_v2/` root, not from inside `tests/`.

---

## Data Pipeline Test

Verifies the full data pipeline from adapter through to preprocessor, for both virtual and COCO sources.

**What it checks:**

**Virtual adapter**
- Loads the correct number of samples from a folder of `<guid>_n.png/json` pairs
- Min/max person count filtering works correctly
- Sample has correct shapes — image `[3, H, W]`, keypoints `[17, 4]` per person
- Visibility values are in `{0, 1, 2}`
- Joints with `v=0` have sentinel position `[-1, -1]`
- Valid joints have `z >= 0`

**COCO adapter**
- Same shape checks as virtual
- Joints with `v=0` have sentinel position `[-1, -1]`
- Valid joints have non-zero MiDaS depth

**Dataset**
- Image is exactly `target_size × target_size` after resize and pad
- Keypoint `x, y` are within `[0, target_size]` after transform
- Sentinel joints `[-1, -1]` are unchanged by the transform
- `z` and `v` columns are untouched by the transform

**Preprocessor**
- Graph has correct number of nodes (only joints with `v > 0`)
- All node features in `[0, 1]`
- No self-loops in edge index
- `joint_types` in range `[0, 16]`
- `person_labels` match expected person count

**Cross-source consistency**
- Virtual and COCO graphs have identical feature dimensions and tensor dtypes

**Visual check**
- Saves an image with keypoints overlaid coloured by person to `--save_dir`

**Run (virtual only):**
```bash
python tests/test_data.py \
  --virtual_dir /home/dean/projects/mills_ds/data/virtual \
  --save_dir /home/dean/projects/mills_ds/outputs/test_outputs/data
```

**Run (virtual + COCO):**
```bash
python tests/test_data.py \
  --virtual_dir /home/dean/projects/mills_ds/data/virtual \
  --coco_img_dir /home/dean/projects/mills_ds/data/coco2017/val2017 \
  --coco_ann_file /home/dean/projects/mills_ds/data/coco2017/annotations/person_keypoints_val2017.json \
  --save_dir /home/dean/projects/mills_ds/outputs/test_outputs/data
```

---

## GAT Isolation Test

> Make sure you have some virtual data with 5 people in it — the test searches for a 5-person scene to use for visualisation.

Verifies the GAT is working correctly in isolation before connecting a grouping head.

**What it checks:**

**Forward pass**
- Output shape is `[N, output_dim]` where N is number of valid joints in the graph
- Output dtype is float32
- No NaN or Inf values

**L2 normalisation**
- All embedding norms are ≈ 1.0 (max deviation < 1e-5)

**Loss computation**
- Contrastive loss returns a scalar tensor
- Loss value is >= 0
- No NaN values

**Gradient flow**
- `loss.backward()` runs without errors
- All parameters receive gradients
- Total gradient norm > 0

**Embedding separability**

The key test — verifies the GAT is actually learning to separate embeddings by person. Reports the following metrics before and after training:

| Metric | What it measures | Want |
|--------|-----------------|------|
| Silhouette score | Overall cluster separation quality | reported only, not used as pass/fail |
| Distance ratio | Inter-person distance / intra-person distance | must improve |
| kNN accuracy | Fraction of 5 nearest neighbours from same person | > 0.9 |

Also saves PCA and t-SNE visualisations to `--save_dir` from a single fixed 5-person scene:
- `embeddings_before.png` — should look like a blob
- `embeddings_after.png` — should show clear clusters per person

If the gap between metrics before and after training is not growing, something is wrong upstream in the data pipeline.

**Run:**
```bash
python tests/test_gat.py \
  --virtual_dir /home/dean/projects/mills_ds/data/virtual \
  --config configs/gat_isolation.yaml \
  --save_dir /home/dean/projects/mills_ds/outputs/test_outputs/gat
```

**Options:**
```
--n_steps    Number of training steps for separability test (default: 100)
--n_graphs   Number of graphs to use for separability test (default: 8)
```

---

## DEC Isolation Test

> DEC was evaluated as a candidate grouping head and ultimately ruled out — see `CHANGES.md` for the full discussion. This test is kept for completeness.

Verifies DEC mechanics work correctly in isolation using fabricated embeddings. No real data or GAT required.

Fabricated embeddings are Gaussian blobs on the unit hypersphere with known ground truth labels. This removes all upstream dependencies and tests exactly what DEC is supposed to do: cluster already-structured embeddings.

**Why fabricated embeddings and not real data?**

DEC is not designed to cluster from scratch — it refines structure that already exists. With random GAT embeddings, `q` is nearly uniform so `KL(p||q) ≈ 0` immediately and nothing is learned. With perfectly trained GAT embeddings, k-means++ already achieves 1.0 accuracy before DEC runs and there's nothing to improve. Fabricated blobs with controlled noise give DEC a task that's genuinely learnable.

**What it checks:**

**Initialisation**
- Cluster centres have shape `[K, D]`
- k-means++ spreads centres out (min pairwise distance > 0)
- No NaN values

**Soft assignment**
- `q` is `[N, K]`, rows sum to 1, values in `[0, 1]`
- `p` is `[N, K]`, rows sum to 1, values in `[0, 1]`
- `p` is sharper than `q` (lower entropy)

**Loss computation**
- KL divergence is a scalar tensor
- Not NaN, not Inf, >= 0

**Gradient flow**
- Cluster centres receive gradients after `loss.backward()`
- Gradient norm > 0

**Assignment accuracy — easy (noise=0.05)**
- Tight, well-separated clusters
- k-means++ already achieves ~1.0 before training — this is an initialisation check
- Accuracy must not degrade

**Assignment accuracy — medium (noise=0.1)**
- The real DEC test — k-means++ gets ~0.62, DEC should push to 1.0
- Accuracy >= 0.75 after training
- Accuracy must not degrade

> Note: the test uses 50 joints per person rather than the real 17. With 17 joints in 128D DEC hits a hard ceiling at 0.8 due to the curse of dimensionality — see `CHANGES.md`. The test uses 50 to verify the mechanism works, not to simulate the real problem.

Accuracy is measured using Hungarian matching — DEC cluster indices are arbitrary so the optimal bijection between predicted clusters and ground truth people is found before scoring.

**Note on K:**
At training time K is taken from ground truth labels. At inference time K must be provided externally. This is a known open problem addressed separately from the grouping task.

**Run:**
```bash
python tests/test_dec.py --config configs/dec_isolation.yaml
```

**Options:**
```
--n_steps        Training steps for accuracy test (default: 1000)
--k              Number of people / clusters (default: 5)
--n_per_person   Joints per person in fabricated data (default: 50)
--noise_easy     Noise for easy test (default: 0.05)
--noise_medium   Noise for medium test (default: 0.1)
```
---

## Slot Attention Isolation Test

Verifies slot attention works correctly in isolation using fabricated embeddings. No real data or GAT required — same approach as `test_dec.py`.

Unlike DEC, slot attention is trained with ground truth labels so it has direct supervision signal. This is why it works at realistic scale (17 joints per person, 128D) where DEC could not.

**What it checks:**

**Forward pass**
- `logits` is `[N, K]`, no NaN or Inf
- `slots` is `[K, D]`, no NaN

**Gradient flow**
- All parameters receive gradients after `loss.backward()`
- Total gradient norm > 0

**Loss computation**
- Cross entropy loss is a scalar tensor
- Not NaN, not Inf, >= 0

**Assignment accuracy — easy (noise=0.05)**
- Tight clusters, 17 joints per person
- Best accuracy >= 0.9 during training

**Assignment accuracy — hard (noise=0.15)**
- Overlapping clusters, 17 joints per person — same scale as the real problem
- Best accuracy >= 0.9 during training

Accuracy is measured using Hungarian matching. Pass/fail is based on best accuracy seen during training rather than final step accuracy — the model is evaluated every 50 steps and the peak is recorded. This is because slot attention can find the correct solution but overshoot past it; the best accuracy is a more honest measure of whether the mechanism works.

**Notes on training stability**

Slot attention is more sensitive to training dynamics than DEC. Two things matter:

- `num_iterations: 7` — more refinement steps per forward pass means slots converge more reliably to consistent assignments, producing cleaner gradient signal
- Cosine LR schedule (1e-3 → 1e-5) — fast learning early, settles late, reduces oscillation

Both are genuine training decisions, not just test fixes. They carry over to full training.

**Run:**
```bash
python tests/test_slot_attention.py --config configs/slot_attention_isolation.yaml
```

**Options:**
```
--n_steps       Training steps (default: 500)
--k             Number of people / slots (default: 5)
--n_per_person  Joints per person (default: 17)
--noise_easy    Noise for easy test (default: 0.05)
--noise_hard    Noise for hard test (default: 0.15)
```

---

## Graph Partitioning Isolation Test

Verifies the edge classifier works correctly in isolation using fabricated embeddings. No real data or GAT required — same approach as the other isolation tests.

The edge classifier predicts for every pair of joints whether they belong to the same person. Groups are recovered by thresholding predictions and running connected components on the resulting affinity graph.

**What it checks:**

**Forward pass**
- `logits` is `[E]` where E = N*(N-1)/2 — one score per pair
- `pairs` is `[E, 2]` — the joint index pairs (i, j) with i < j
- No NaN or Inf in logits

**Gradient flow**
- All parameters receive gradients after `loss.backward()`
- Total gradient norm > 0

**Loss computation**
- Binary cross entropy is a scalar tensor
- Not NaN, not Inf, >= 0
- Reports edge F1, edge accuracy, and grouping accuracy

**Assignment accuracy — easy (noise=0.05)**
- Best edge F1 >= 0.9
- Best grouping accuracy >= 0.9

**Assignment accuracy — hard (noise=0.15)**
- Same thresholds at realistic scale — 17 joints per person, 128D

Two metrics are reported throughout training:

- **Edge F1** — how well the classifier labels individual pairs as same/different person. F1 rather than accuracy because pairs are heavily class-imbalanced (far more different-person pairs than same-person pairs with 5 people).
- **Grouping accuracy** — end-to-end metric. Predicted edges are thresholded at 0.5, connected components recover person groups, Hungarian matching scores them against ground truth. This is what actually matters.

Pass/fail uses best seen during training (recorded every 50 steps) for consistency with the slot attention test.

**Note on class imbalance**

With 5 people × 17 joints there are 5 × C(17,2) = 680 same-person pairs and 2890 different-person pairs. The loss uses `pos_weight = n_neg / n_pos` (clamped at 10) to prevent the model collapsing to predicting all-different.

**Run:**
```bash
python tests/test_graph_partitioning.py --config configs/graph_partitioning_isolation.yaml
```

**Options:**
```
--n_steps       Training steps (default: 500)
--k             Number of people (default: 5)
--n_per_person  Joints per person (default: 17)
--noise_easy    Noise for easy test (default: 0.05)
--noise_hard    Noise for hard test (default: 0.15)
```

---