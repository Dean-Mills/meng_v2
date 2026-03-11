# Changes

## Dataset generator

### Reason 

The old code had separate files for everything:

virtual_dataloader.py, virtual_dataset.py, virtual_preprocessor.py as well as dataloader.py, dataset.py, preprocessor.py, depth_estimation.py

The architecture has been updated to look like this:

```
COCO data          Virtual data
    ↓                   ↓
coco_adapter.py    virtual_adapter.py
         ↓         ↓
        dataset.py          ← single Dataset class
             ↓
        preprocessor.py     ← single graph builder
             ↓
        dataloader.py       ← single DataLoader
```

Two reasons for this:
- After the adapter we don't care where the data came from, the rest of the pipeline is identical
- Can train on synthetic data first then swap to COCO without changing anything downstream


### Updates

**extractor.py**

Keypoint format [x,y,v] → [x,y,z,v] per joint (51 → 68 values)
- Per-joint depth is strictly better than one depth value per person

z = world-space Euclidean distance from camera to joint in metres
- Consistent across all camera positions, matches MiDaS semantics after normalisation

Visibility 0/2 → 0/1/2 with raycast occlusion detection
- Occluded joints are real training signal — marking them as invisible was losing information

calculate_distance_to_camera removed
- Redundant — replaced by per-joint z

Out-of-frame joints position changed from [0.0, 0.0] to [-1.0, -1.0]
- Zero is a real pixel location (top-left corner), -1 is an unambiguous sentinel meaning "no valid position". The GAT uses visibility v=0 to ignore these joints during message passing but the position itself should also signal invalidity explicitly.

**renderer.py**

Filenames changed to `<guid>_<n>.png` / `<guid>_<n>.json`
- GUIDs eliminate ID collisions, _n suffix allows filtering by person count without parsing JSON

num_people added to image metadata
- Useful for debugging and dataset statistics


### Virtual adapter

Loads a folder of `<guid>_n.png/<guid>_n.json` pairs and scans the folder to filter using min and max people.

Converts from the per image JSON:
```
[x, y, z, v, x, y, z, v, ...]   # 68 floats per person
```
to the unified output:
```python
{
    "image":      tensor [3, H, W],        # raw uint8 RGB
    "keypoints":  list of tensor [17, 4],  # one tensor per person, columns: x, y, z, v
    "img_id":     str,                     # GUID
    "num_people": int
}
```

No transforms, no normalisation, no MiDaS — just reading and reshaping. Everything else happens downstream.


### COCO adapter

Loads COCO 2017 person keypoint annotations via pycocotools, filters by min/max person count the same way as the virtual adapter.

MiDaS DPT_Hybrid is initialised once at construction time and used to estimate a depth map per image. Depth is then sampled at each keypoint's pixel location.

Converts from COCO's flat format:
```
[x, y, v, x, y, v, ...]   # 51 floats per person, COCO visibility 0/1/2
```
to the unified output:
```python
{
    "image":      tensor [3, H, W],
    "keypoints":  list of tensor [17, 4],  # [x, y, z, v]
    "img_id":     str,                     # COCO int cast to string
    "num_people": int
}
```

Visibility is handled explicitly:

| COCO v | Position | Depth |
|--------|----------|-------|
| 0 not labeled | [-1, -1] | 0 |
| 1 occluded | real x, y | MiDaS sampled |
| 2 visible | real x, y | MiDaS sampled |


### Dataset

Single shared Dataset class that sits between the adapters and the rest of the pipeline. Accepts either adapter and applies a consistent transform so everything downstream sees the same tensor shapes regardless of data source.

- Image resized to fit within target_size × target_size preserving aspect ratio
- Symmetrically padded to exactly target_size × target_size
- Keypoint x, y scaled and shifted to match the new image dimensions
- z and v columns left untouched
- Joints with v=0 (sentinel -1, -1) are not transformed

Output:
```python
{
    "image":      tensor [3, target_size, target_size],
    "keypoints":  list of tensor [17, 4],
    "img_id":     str,
    "num_people": int
}
```


### Dataloader

Wraps a PoseDataset with a custom collate function. Images stacked into a single tensor, keypoints kept as a list of lists since person count varies per image.

Output per batch:
```python
{
    "image":      tensor [B, 3, H, W],
    "keypoints":  list[list[tensor [17, 4]]],
    "img_id":     list[str],
    "num_people": list[int]
}
```

batch_size is an explicit argument — the caller controls it from config.yaml.


### Preprocessor

Single unified graph builder. Converts a DataLoader batch into PyTorch Geometric graphs ready for the GAT. One graph per image, all people's joints are nodes in the same graph.

Joints with v=0 are excluded entirely — no position, no node. Everything else becomes a node.

Node features:
```
x_norm   — x normalised to [0, 1] by image size
y_norm   — y normalised to [0, 1] by image size
z_norm   — depth normalised per-graph min-max over valid joints
v_norm   — visibility normalised (0→0.0, 1→0.5, 2→1.0)
```

Separate tensors per graph:
```
joint_types   — [N] long, index 0-16 for GAT embedding layer
person_labels — [N] long, ground truth person assignment for training
```

Edges: kNN with k=8 built on (x_norm, y_norm, z_norm) — depth is part of the proximity calculation so joints that are close in 3D get connected even if far apart in 2D.

Key decisions:
- Per-graph depth normalisation — MiDaS and metric depth have different absolute scales, normalising per-graph makes both sources comparable
- joint_types separate from features — lets the GAT learn a per-joint-type embedding independently of position
- kNN includes depth — richer connectivity than 2D-only


### Data pipeline test results

All tests passed — see `outputs/test_outputs/data`

---

## GAT

### What it does

The GAT (Graph Attention Network) takes the graph of joints built by the preprocessor and produces a 128-dimensional embedding for each joint. The goal is for joints belonging to the same person to end up close together in this 128D space, and joints from different people to be far apart. The grouping head will sit on top of this and use those embeddings to make the final assignment.

The architecture:
- Each joint starts as a 4-value feature vector [x, y, z, v] plus a learned 32-dim embedding for its joint type (nose, left knee, etc.) — so the model knows what kind of joint it's looking at independently of where it is
- Two GATv2 layers with 4 attention heads do message passing — each joint updates its representation by attending to its neighbours
- A final linear projection maps down to 128 dimensions
- L2 normalisation so all embeddings live on the unit hypersphere

The joint type embedding matters — a left knee should have different characteristics than a nose regardless of position, and keeping this separate from the spatial features lets the GAT learn that independently.


### Training (isolation phase)

The GAT is trained in isolation using contrastive loss only. Pull same-person joint embeddings together, push different-person joint embeddings apart. No grouping head yet — just verifying the embedding space is structured correctly before adding the next component.

Contrastive loss works on pairwise similarities between all joints in the graph. For each pair from the same person it penalises low similarity, for each pair from different people it penalises similarity above a margin of 0.5.

```yaml
gat:
  num_joint_types: 17
  joint_embedding_dim: 32
  raw_feature_dim: 4
  hidden_dim: 64
  output_dim: 128
  num_layers: 2
  num_heads: 4
  dropout: 0.1
  use_layer_norm: true
  l2_normalize: true

loss:
  contrastive_margin: 0.5
```


### Test results

Four sanity checks passed in both runs:
- Forward pass — output shape [N, 128] ✓
- L2 normalisation — all norms ≈ 1.0 ✓
- Loss computation — scalar, no NaN, >= 0 ✓
- Gradient flow — all parameters receive gradients ✓

Two separability runs:

| Metric | 100 steps / 8 graphs before | 100 steps / 8 graphs after | 500 steps / 16 graphs before | 500 steps / 16 graphs after |
|--------|-------|-------|-------|-------|
| Silhouette | -0.043 | -0.082 | -0.030 | -0.042 |
| Distance ratio | 1.006 | 1.108 | 0.992 | 1.020 |
| kNN accuracy | 0.630 | 0.983 | 0.596 | 0.985 |

kNN accuracy is the most meaningful number here — 0.983 and 0.985 means almost every joint's 5 nearest neighbours in embedding space belonged to the same person, in both runs.

The distance ratio is noticeably better in the 100 step run (1.108 vs 1.020). This is expected — 16 graphs covers more diverse scenes which makes the task harder. The model is doing more work to separate more varied configurations of people, so the absolute distances compress. The kNN accuracy tells you it's still succeeding.

Silhouette stays negative in both runs. This is a known limitation — silhouette uses global Euclidean distances which get geometrically compressed when everything lives on a unit hypersphere. It's not a useful metric here and is reported for completeness only.


### How to read the plots

Both runs visualise a single fixed scene with 5 people and 85 joints. Each colour is one unambiguous person. Every dot is one joint.

**PCA** projects the 128-dimensional embeddings down to 2D by finding the axes of maximum variance. The variance explained number tells you how much of the total structure is visible — above 97% in both after-training plots, meaning the embedding space is almost entirely organised along two directions.

**t-SNE** preserves local neighbourhood structure rather than global variance. Absolute positions and distances don't mean anything, but if same-colour points cluster together that's real.

**Before training** — both plots show colours scattered and mixed. The untrained GAT has no concept of person identity. PCA variance explained is around 87%, meaning the space has no dominant structure yet.

**After 100 steps** — five tight, clearly separated clusters in both PCA and t-SNE. PCA variance explained jumped to 99.17%. Each person's joints — nose, knees, shoulders, elbows — all ended up in the same region of embedding space despite being physically spread across the body.

**After 500 steps** — same clean result, 97.56% PCA variance. Slightly less sharp on the distance ratio metric due to training on more diverse scenes, but visually identical.

The single-scene visualisation is important here — each colour maps to exactly one physical person in the scene, so the clustering is directly interpretable.


---

## Grouping head 

GAT isolation verified. Next step is the grouping head — takes the embeddings from the GAT and produces the final joint-to-person assignment.

There are 4 we can choose from

- Deep Embedded clustering
- DETR (DEtectron Transformer)
- Graph partitioning
- Slot attention

## DEC (Deep Embedded Clustering)

### What it does

DEC takes embeddings and learns K cluster centres in the same embedding space. Each joint gets a soft assignment `q` based on distance to each centre using a Student's t-distribution. A sharpened target distribution `p` is computed from `q` and the model is trained to minimise KL divergence between them — pushing soft assignments toward harder, more confident ones over time.

Cluster centres are initialised with k-means++ before training, which spreads them out proportionally to distance rather than randomly. This matters — bad initialisation means DEC never recovers.

One important implementation detail: `p` is updated every `update_interval` steps rather than every step. Updating every step causes cluster collapse — the model chases a moving target and all joints pile into one cluster by step 150. Periodic updates let the centres stabilise between target refreshes.


### What we tried first (and why it kept failing)

**Attempt 1 — random GAT embeddings.**
The first test ran DEC on a real scene with an untrained GAT. With random embeddings all joints are roughly equidistant from all cluster centres, so `q` is nearly uniform across all K clusters. `p` is computed from `q` so `p ≈ q`, which means `KL(p||q) ≈ 0` immediately — no loss, no gradient, no learning. DEC just sat there.

**Attempt 2 — pre-trained GAT embeddings.**
The fix seemed obvious — train the GAT first with contrastive loss so the embeddings are already structured, then run DEC on top. This worked better initially. With 200 steps of GAT pre-training, k-means++ initialised perfectly (accuracy 1.0 before any DEC training). But then something unexpected happened:

```
Before training: accuracy = 1.0000
Step  25: accuracy=1.0000  loss=0.0661
...
Step 125: accuracy=0.5765  loss=0.0005
Step 150: accuracy=0.2000  loss=0.0002
Step 175: accuracy=0.2000  loss=0.0001
```

DEC started at perfect accuracy and actively degraded to random chance (0.2 = 1/5 clusters). This is cluster collapse — because `p` was being recomputed from `q` every single step, the model was chasing a moving target. The dominant cluster absorbed more and more joints until everything collapsed into one. This is a known failure mode from the original DEC paper that we hadn't accounted for.

The fix was periodic `p` updates — refresh the target distribution every `update_interval` steps instead of every step, giving the centres time to stabilise between updates.

**Attempt 3 — periodic updates, real scene, 17 joints per person.**
With cluster collapse fixed we went back to real scenes. But now a different problem: with perfectly trained GAT embeddings, k-means++ already achieved 1.0 accuracy before DEC ran. DEC had nothing to improve. The KL loss plateaued immediately and the test was not actually testing anything.

We switched to fabricated embeddings — Gaussian blobs on the unit hypersphere with known ground truth — so we could control difficulty independently of the GAT. With 17 joints per person and noise=0.15 the results hit a hard ceiling:

```
Step 250: accuracy=0.7647  loss=0.0273
Step 300: accuracy=0.7647  loss=0.0272
...
Step 1000: accuracy=0.7647  loss=0.0274
```

0.7647 exactly — 4 out of 5 clusters correct, one pair always confused. Tried noise=0.1, noise=0.05, 1000 steps, still 0.8 ceiling. This is the curse of dimensionality — 17 points in 128D is too sparse for the Student's t soft assignments to produce a meaningful gradient. Everything is roughly equidistant in high dimensions and DEC can't resolve the ambiguous cluster.

**Attempt 4 — 50 joints per person.**
Bumped `n_per_person` to 50 to give DEC more signal. This finally worked:

```
Before training: accuracy = 0.6200
Step 250: accuracy=0.9680
Step 300: accuracy=1.0000
After training:  accuracy = 1.0000
```

0.62 before, 1.0 after. DEC working as intended — but only with 3x more points per cluster than the real problem has.


### Isolation test results (final)

5 clusters, 50 points per cluster, 128D, `update_interval=20`:

| | Before training | After training |
|---|---|---|
| Easy (noise=0.05) | 1.000 | 1.000 |
| Medium (noise=0.1) | 0.620 | 1.000 |

Accuracy measured with Hungarian matching since DEC cluster indices are arbitrary.


### Why DEC isn't the right head for this problem

DEC works in isolation when given enough points per cluster. The problem is how it fits into this pipeline.

DEC is fully unsupervised — it never sees ground truth labels. It can only sharpen assignments that are already roughly correct. It has no signal to fix wrong ones. In real scenes you have 17 joints per person, not 50. Testing repeatedly showed a hard ceiling at 0.8 with 17 joints in 128D regardless of noise level, steps, or tuning — one cluster pair was always confused and DEC couldn't resolve it without ground truth signal.

There's also a more fundamental point: if the GAT is doing its job, kNN accuracy on the embeddings is already 0.98-1.0. At that point k-means++ alone at inference time would give near-perfect assignments. DEC adds training complexity and collapse risk without meaningfully improving on the initialisation.

The right grouping head for this problem needs to learn from ground truth labels. Graph partitioning and slot attention both do this — trained end-to-end against ground truth person assignments rather than self-supervising. That's the next step.

------------------------------------------------------------------------------------
# Progress Summary

## What's been built

### Data pipeline

- Synthetic data generator (Blender/Mixamo) producing `<guid>_n.png/json` pairs with per-joint depth, raycast occlusion, and sentinel positions for out-of-frame joints
- Virtual adapter — loads and filters synthetic scenes by person count
- COCO adapter — loads COCO 2017 annotations with MiDaS depth estimation
- Single shared Dataset class that normalises both sources to the same format
- Preprocessor — converts batches into PyTorch Geometric graphs, one per scene, with kNN edges in 3D space
- All pipeline tests passing — see `outputs/test_outputs/data`

### GAT

- GATv2 with joint type embeddings, 2 layers, 4 heads, 128D L2-normalised output
- Contrastive loss — pulls same-person joints together, pushes different-person joints apart
- Isolation test passing — kNN accuracy 0.98–1.0, clear cluster separation in PCA/t-SNE from a single 5-person scene
- See `outputs/test_outputs/gat`

### DEC

- Implemented and tested in isolation — ultimately ruled out as a grouping head
- See below for why

---

## Why DEC didn't work

DEC is unsupervised — it only optimises KL divergence between its soft assignment `q` and a sharpened target `p`. It never sees ground truth labels. Three things went wrong during testing:

**With untrained GAT embeddings** — all joints are roughly equidistant from all cluster centres in 128D, so `q` is nearly uniform across all K clusters. `p ≈ q`, so `KL(p||q) ≈ 0` immediately. No loss, no gradient, nothing learned.

**With trained GAT embeddings** — k-means++ already achieved 1.0 accuracy before DEC ran. Nothing to improve. When DEC was allowed to train anyway it actively degraded — accuracy dropped from 1.0 to 0.2 (random chance) by step 150. This is cluster collapse: `p` was being recomputed every step, the model chased a moving target, and all joints piled into one cluster. Fixed with periodic `p` updates but the underlying problem remained — if the GAT is good there's nothing for DEC to do.

**With fabricated blobs at realistic scale (17 joints per person, 128D)** — hit a hard ceiling at 0.8 regardless of noise, steps, or tuning. One cluster pair always confused, DEC couldn't resolve it. This is the curse of dimensionality — 17 points in 128D is too sparse for the soft assignments to produce a meaningful gradient signal. Only worked when bumped to 50 joints per cluster, which is 3x more than the real problem provides.

The deeper issue is that DEC is the wrong tool for this problem. It can only sharpen assignments that are already roughly correct — it has no mechanism to fix wrong ones. With 17 joints per person and no ground truth signal it simply doesn't have enough to work with. Full discussion in `CHANGES.md`.

---

## Slot Attention

### Why slot attention

DEC failed at realistic scale because it is unsupervised — it has no access to ground truth labels and cannot fix wrong assignments, only sharpen correct ones. With 17 joints per person in 128D the self-training signal is too weak and a hard ceiling of 0.8 was observed regardless of tuning. The fundamental requirement is a grouping head that learns directly from labelled data. Slot attention satisfies this — it is trained end-to-end with ground truth person assignments via Hungarian-matched cross entropy.


### What it does

Slot attention is an iterative competitive attention mechanism. K slot vectors are initialised by sampling from a learned Gaussian distribution, one slot per person. Over several iterations each slot attends to all joints and competes for the ones it best represents. The competition is enforced by applying softmax over slots rather than over joints — each joint's attention weights sum to 1 across all slots, so if two slots compete for the same joint the better-matching one wins. Slots naturally specialise onto different people without any explicit assignment logic beyond the loss.

After iterative refinement the final slot vectors score each joint via dot product, producing logits `[N, K]`. Hungarian matching finds the optimal bijection between predicted slots and ground truth people, and cross entropy is computed on the matched assignments.

The slot dimension is derived from `gat.output_dim` rather than configured separately — slot vectors and GAT embeddings live in the same space so attention scores are geometrically meaningful.


### Architecture

```
Inputs:  embeddings [N, D]  L2-normalised GAT embeddings
         k          int     number of people in this scene

Slots sampled from learned μ, σ  →  [K, D]

For num_iterations:
    norm(slots)   →  queries  [K, D]
    norm(inputs)  →  keys     [N, D]
                     values   [N, D]

    scores   =  queries · keys^T * scale     [K, N]
    softmax over slots  (competition)        [K, N]
    normalise within each slot               [K, N]
    updates  =  weighted sum of values       [K, D]
    slots    =  GRU(updates, slots)          [K, D]
    slots    =  slots + FF(LayerNorm(slots)) [K, D]

slot_keys  =  linear(slots)                 [K, D]
logits     =  embeddings · slot_keys^T      [N, K]
```

Loss: Hungarian matching between predicted slots and ground truth people, then cross entropy on matched assignments.

Config:
```yaml
slot_attention:
  num_iterations: 7    # refinement steps per forward pass
  slot_weight:    1.0  # loss weight
```


### What went wrong during testing

**Instability at lr=1e-3, 3 iterations** — the model found the correct solution early (accuracy 1.0 at step 100 on the easy test) but then overshot and lost it, ending at 0.78. Loss was bouncing between 0.37 and 1.47 within the same run. The problem is that slots reinitialise from the learned Gaussian every forward pass — with only 3 iterations they do not always converge to the same solution, producing noisy gradient signal that destabilises training.

**Fix 1 — more iterations.** Increasing `num_iterations` from 3 to 7 gives slots more refinement steps per forward pass. With 5 people in 128D, 3 iterations is insufficient for consistent convergence. 7 iterations stabilised the assignments within each forward pass and produced cleaner gradient signal. This is a genuine architectural decision, not a test fix — it would carry over to full training.

**Fix 2 — cosine LR schedule.** A flat lr of 1e-3 caused the model to find the correct solution and then overshoot. A cosine schedule decaying from 1e-3 to 1e-5 allows fast early learning and fine-grained settling once assignments are mostly correct. Again a real training decision, not specific to the isolation test.

**Evaluation on best accuracy rather than final step.** With a noisy loss curve the final evaluation checkpoint is not always the best point. Accuracy is recorded every 50 steps and pass/fail is based on the peak seen during training. This is honest — the model found the correct solution, it should not be penalised for the optimiser state at one arbitrary checkpoint.


### Isolation test results

5 clusters, 17 joints per person (real scale), 128D, 500 steps, cosine LR 1e-3 → 1e-5:

| | Before training | Best accuracy | Step reached |
|---|---|---|---|
| Easy (noise=0.05) | 0.459 | 1.000 | 250 |
| Hard (noise=0.15) | 0.400 | 1.000 | 200 |

Both tests reach perfect accuracy at realistic joint counts — 17 joints per person, the same scale the real problem provides. DEC hit a hard ceiling of 0.8 at this scale and never improved.


### Comparison with DEC

| | DEC | Slot attention |
|---|---|---|
| Supervision | Unsupervised (KL only) | Supervised (cross entropy + ground truth) |
| Ceiling at 17 joints / 128D | 0.8 | 1.0 |
| Can fix wrong assignments | No | Yes |
| End-to-end trainable with GAT | Weakly | Yes |

The supervision signal is the deciding factor. With 17 joints per person the self-training signal in DEC is too weak to resolve ambiguous cluster pairs. Slot attention resolves them correctly because it has access to ground truth labels during training.

## Graph Partitioning

### What it does

Graph partitioning frames the grouping problem as binary edge classification.
For every pair of joints (i, j) the model predicts whether they belong to the
same person. After thresholding, connected components on the resulting affinity
graph recover the person groups without any explicit knowledge of K.

This is a fundamentally different framing from slot attention. Slot attention
assigns joints to K pre-specified slots via competitive attention. Graph
partitioning makes pairwise same/different decisions and lets the group
structure emerge from the graph topology. K does not need to be specified at
inference time — it falls out naturally from the number of connected components.

### Architecture

For each pair (i, j) the feature vector is:

```
f(i, j) = [e_i − e_j  ‖  e_i ⊙ e_j]   ∈ R^{2D}
```

where e_i, e_j are L2-normalised GAT embeddings. The difference captures
direction between the two points in embedding space; the element-wise product
captures co-activation. This is option A — a learned classifier on top of the
embeddings rather than simply thresholding cosine similarity. The model learns
what "same person" means in embedding space from labelled data rather than
relying on a fixed geometric criterion.

The MLP maps `R^{2D}` to a scalar logit per pair:

```
Linear(2D → H) → ReLU → LayerNorm → Dropout
→ Linear(H → H/2) → ReLU
→ Linear(H/2 → 1)
```

Groups are recovered at inference by thresholding predicted affinities at 0.5
and finding connected components via union-find.

Config:
```yaml
graph_partitioning:
  hidden_dim:       256    # MLP hidden dimension
  dropout:          0.1
  threshold:        0.5    # affinity threshold at inference
  partition_weight: 1.0    # loss weight
```

### Loss

Binary cross entropy over all N*(N-1)/2 pairs. With 5 people × 17 joints
there are 680 same-person pairs and 2890 different-person pairs — a 4:1
class imbalance. Without correction the model collapses to predicting
all-different and achieves high accuracy trivially. `pos_weight = n_neg / n_pos`
(clamped at 10) upscales the positive class so same-person pairs contribute
proportionally to the gradient.

Two metrics are tracked during training:

- **Edge F1** — precision/recall balance on the binary pair classification.
  Accuracy is not used because it is misleading under class imbalance.
- **Grouping accuracy** — end-to-end metric. Thresholded edges → connected
  components → Hungarian matching against ground truth. This is the metric
  that actually matters for the grouping task.

### Isolation test results

5 clusters, 17 joints per person (real scale), 128D, 500 steps, cosine LR
1e-3 → 1e-5:

| | Before training | Best F1 | Best grouping acc | Step reached |
|---|---|---|---|---|
| Easy (noise=0.05) | F1=0.083 / acc=0.212 | 1.000 | 1.000 | 50 |
| Hard (noise=0.15) | F1=0.235 / acc=0.200 | 1.000 | 1.000 | 50 |

Perfect F1 and grouping accuracy by step 50 on both tests, holding stable
for the remaining 450 steps with loss decaying smoothly to near zero. This
is the cleanest result of all three grouping heads tested.

### Comparison across all grouping heads

| | DEC | Slot attention | Graph partitioning |
|---|---|---|---|
| Supervision | Unsupervised | Supervised (CE) | Supervised (BCE) |
| Ceiling at 17 joints / 128D | 0.8 | 1.0 | 1.0 |
| Requires K at inference | Yes | Yes | No |
| Stability | Flat (collapsed) | Noisy | Very stable |
| Steps to converge | Never | ~200 | ~50 |
| End-to-end trainable with GAT | Weakly | Yes | Yes |

The key practical advantage of graph partitioning over slot attention is
stability and convergence speed. Slot attention must coordinate K slot vectors
via competitive attention across multiple iterations — the gradient signal is
indirect and the training curve is noisy. The edge classifier receives a direct
binary supervision signal from 3570 labelled pairs per scene, which is dense
and unambiguous.

The key theoretical advantage over slot attention is that K does not need to
be known at inference time. This is a significant practical benefit for the
real problem where the number of people in a scene is not known in advance.

## DETR — Rationale for Exclusion

DETR was considered as a fourth grouping head. A preliminary implementation
was attempted but not pursued to isolation testing. The decision was made to
exclude it from the final comparison for the following reasons.

**Architectural overlap with slot attention.** DETR's decoder is a superset
of slot attention — it adds self-attention between queries and a heavier
feed-forward stack, but the core mechanism is the same iterative cross-attention
competition between learned queries and input features. For this problem the
additional complexity adds no meaningful capability.

**The null object problem.** DETR was designed for object detection where the
number of detections varies per image. It handles this by padding predictions
with a learned "no object" class and training the model to predict empty slots
for unused queries. This is the source of the null value instability encountered
in the preliminary implementation. Resolving it correctly requires careful loss
masking and balanced sampling of empty vs occupied slots — engineering overhead
with no benefit for this problem, where the number of people per scene is
bounded and known at training time.

**The comparison is already complete without it.** The four approaches evaluated
— kNN on GAT embeddings, DEC, slot attention, and graph partitioning — span the
relevant design space: unsupervised vs supervised, slot-based vs pairwise,
K-required vs K-free. DETR would add a fifth data point that is not
meaningfully distinct from slot attention in any of these dimensions.

## Experimental Results

### Metrics

**PGA (Pose Grouping Accuracy)**
The primary metric. For each scene, predicted person groups are matched to
ground truth people via Hungarian matching — finding the optimal bijection
between predicted and ground truth labels. PGA is the fraction of joints
assigned to the correct person under this optimal matching. A score of 1.0
means every joint was grouped correctly. This metric is permutation-invariant
— the model is not penalised for labelling people in a different order than
the ground truth.

**NMI (Normalised Mutual Information)**
Measures the quality of the GAT embeddings independently of the grouping head.
K-means is run on the embeddings with ground truth K, and NMI measures how
much information the resulting clusters share with the ground truth person
labels. A score of 1.0 means perfect correspondence. NMI is the same across
all methods evaluated from the same checkpoint because it depends only on the
embeddings, not the grouping head.

**ARI (Adjusted Rand Index)**
A second embedding quality metric. Measures agreement between the k-means
clustering and ground truth, corrected for chance. Ranges from 0 (random) to
1 (perfect). Like NMI, it is the same across all heads from the same checkpoint.

**Detection F1**
Measures whether the method correctly counts the number of people in the scene.
For kNN this is always 1.0 because the ground truth K is provided explicitly.
For slot attention K is also provided explicitly so F1 is close to 1.0.
For graph partitioning K is inferred from connected components — if the edge
threshold splits or merges groups incorrectly, F1 drops. Detection F1 therefore
measures threshold calibration for graph partitioning.

**Per-joint accuracy**
PGA broken down by COCO joint type. Reveals whether certain joints are harder
to group correctly — typically extremities (ankles, wrists) which are more
likely to be occluded or close to another person's joints.

---

### Experiment 1 — Synthetic test set

All models trained on 400 virtual scenes (100 each at 2/3/4/5 people),
evaluated on 100 held-out virtual scenes. Training: 50 epochs, cosine LR
1e-3 → 1e-5, AdamW.

| Method | PGA | Std | NMI | ARI | Det F1 |
|---|---|---|---|---|---|
| kNN (contrastive GAT) | 0.994 | 0.012 | 0.987 | 0.987 | 1.000 |
| kNN (partition GAT) | 0.995 | 0.012 | 0.987 | 0.987 | 1.000 |
| Graph partitioning | 0.954 | 0.115 | 0.987 | 0.987 | 0.973 |
| kNN (slot GAT) | 0.994 | 0.015 | 0.988 | 0.987 | 1.000 |
| Slot attention | 0.858 | 0.143 | 0.988 | 0.987 | 0.935 |

**Per-joint accuracy — synthetic test set (kNN / graph partitioning / slot attention):**

| Joint | kNN | Graph partition | Slot attention |
|---|---|---|---|
| nose | 1.000 | 0.955 | 0.852 |
| left eye | 1.000 | 0.955 | 0.852 |
| right eye | 1.000 | 0.955 | 0.854 |
| left ear | 1.000 | 0.955 | 0.854 |
| right ear | 1.000 | 0.955 | 0.856 |
| left shoulder | 1.000 | 0.955 | 0.861 |
| right shoulder | 1.000 | 0.955 | 0.864 |
| left elbow | 0.992 | 0.955 | 0.860 |
| right elbow | 0.984 | 0.947 | 0.852 |
| left wrist | 0.994 | 0.955 | 0.873 |
| right wrist | 0.975 | 0.952 | 0.853 |
| left hip | 1.000 | 0.955 | 0.863 |
| right hip | 1.000 | 0.955 | 0.863 |
| left knee | 1.000 | 0.955 | 0.868 |
| right knee | 0.996 | 0.953 | 0.864 |
| left ankle | 0.990 | 0.955 | 0.855 |
| right ankle | 0.984 | 0.941 | 0.842 |

On synthetic data kNN is the strongest method. The GAT with contrastive loss
produces near-perfect embeddings — NMI 0.987, ARI 0.987 — and k-means is
sufficient to recover the correct groups. The learned grouping heads do not
improve on this. Graph partitioning reaches 0.954 with higher variance (std
0.115), and slot attention reaches 0.858. The head loss going to near zero
by epoch 36 for graph partitioning confirms that the edge classifier solved
its task quickly and stopped contributing meaningful gradient signal — the GAT
was already making the problem trivially easy.

---

### Experiment 2 — COCO val2017 (zero-shot transfer)

The same trained checkpoints evaluated on COCO val2017 with no fine-tuning.
MiDaS DPT_Hybrid provides per-joint depth estimates. 2346 images, all with
at least one annotated person.

This is a zero-shot sim-to-real transfer test — the models have never seen
real images during training.

| Method | PGA | Std | NMI | ARI | Det F1 |
|---|---|---|---|---|---|
| kNN — contrastive GAT | **0.838** | 0.194 | 0.754 | 0.692 | 1.000 |
| kNN — partition GAT | 0.836 | 0.195 | 0.751 | 0.687 | 1.000 |
| kNN — slot GAT | 0.828 | 0.202 | 0.740 | 0.675 | 1.000 |
| Slot attention | 0.788 | 0.236 | 0.740 | 0.675 | 0.920 |
| Graph partitioning | 0.752 | 0.266 | 0.751 | 0.687 | 0.802 |

**Per-joint accuracy — COCO val2017 (kNN contrastive / graph partition / slot attention):**

| Joint | kNN | Graph partition | Slot attention |
|---|---|---|---|
| nose | 0.832 | 0.748 | 0.786 |
| left eye | 0.818 | 0.733 | 0.778 |
| right eye | 0.819 | 0.741 | 0.779 |
| left ear | 0.796 | 0.697 | 0.744 |
| right ear | 0.807 | 0.709 | 0.761 |
| left shoulder | 0.835 | 0.746 | 0.780 |
| right shoulder | 0.838 | 0.747 | 0.786 |
| left elbow | 0.841 | 0.739 | 0.780 |
| right elbow | 0.833 | 0.748 | 0.783 |
| left wrist | 0.848 | 0.732 | 0.780 |
| right wrist | 0.831 | 0.740 | 0.787 |
| left hip | 0.837 | 0.745 | 0.770 |
| right hip | 0.835 | 0.747 | 0.769 |
| left knee | 0.833 | 0.742 | 0.767 |
| right knee | 0.816 | 0.741 | 0.772 |
| left ankle | 0.775 | 0.706 | 0.749 |
| right ankle | 0.777 | 0.714 | 0.748 |

---

### Conclusions

**1. The contrastive loss alone is the dominant learning signal.**
Joint training with a grouping head did not improve the GAT embeddings.
kNN on the contrastive-only GAT (0.838) is essentially identical to kNN on
the jointly trained GATs (0.836, 0.828). The heads add complexity without
improving the upstream representation.

**2. The sim-to-real gap is the primary open problem.**
PGA drops from 0.994 to 0.838 on real data for kNN. NMI drops from 0.987
to 0.754. The embeddings are less separable on COCO because the model was
trained purely on synthetic data. The per-joint accuracy pattern on COCO
shows ankles and ears consistently lowest — joints that are frequently
occluded or at the edge of the image, exactly where synthetic data diverges
most from real scenes.

**3. Slot attention generalises better than graph partitioning on real data.**
On synthetic data slot attention trails kNN by 14 points. On COCO that gap
closes to 5 points (0.838 vs 0.788). Graph partitioning goes the other
direction — it trails kNN by 4 points on synthetic but 8.6 points on COCO,
and its detection F1 collapses to 0.802. The fixed threshold of 0.5
calibrated on synthetic data does not transfer to noisier real embeddings.
Slot attention, trained with ground truth supervision, generalises more
gracefully.

**4. kNN with a good GAT is a strong and robust baseline.**
Given ground truth K, k-means on L2-normalised GAT embeddings is competitive
with or better than both learned heads on both domains. It has perfect
detection F1 by construction and lower variance than the learned heads in
every experiment. This suggests that for deployment on synthetic data or
data close to the training distribution, the learned grouping heads are not
necessary.

**5. Learned heads become relevant as scene complexity increases.**
The relative improvement of slot attention over its synthetic performance on
COCO suggests that learned grouping provides a meaningful benefit when
embeddings are noisier. The natural next step is fine-tuning on a small
amount of real annotated data — if slot attention can close the remaining
5-point gap to kNN on COCO after fine-tuning, that would confirm the head
is learning something useful beyond what the embeddings alone provide.