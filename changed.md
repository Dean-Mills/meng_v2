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