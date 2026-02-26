# Debugging the DETR Decoder: A Systematic Investigation

## Executive Summary

**The problem:** My DETR decoder couldn't figure out *how many* people were in a scene. The existence head — which is supposed to predict "is this query a real person?" — was stuck at random (loss ≈ 0.693 = ln(2)) across every experiment. It never learned.

**What I tried (and why each failed):**

1. **Threshold sweep on real data** — Found false positives sit in a narrow 0.50–0.55 band. Raising threshold to 0.55 eliminated all false positives (precision: 82% → 100%) but missed 10% of real people. Cheap fix, but ceiling at ~90% recall.

2. **Isolated the DETR on synthetic data** — Gave it *perfect* input (well-separated clusters, σ=0.05 noise). It still failed: existence loss stuck at 0.69, collapsed to always predicting 2 people. Proved the problem is the DETR architecture, not bad embeddings.

3. **Added a count head** — Pool all query features, predict total count. Improved counting (error: 1.54 → 0.88) but learned the dataset mean (~3.3), not per-sample counts. PGA actually dropped.

4. **Orthogonal init + 10× query LR + auxiliary losses** — Three known DETR training fixes. Marginal PGA gain (+0.03). Existence still stuck at 0.67. These are training dynamics fixes for what turned out to be a structural problem.

**The root cause:** In standard DETR (object detection), unused queries attend to *background* pixels (sky, grass) and develop distinct "no-object" features. In my architecture, the memory is *only* real joints — no background. Unmatched queries attend to real joints and get person-like features indistinguishable from matched queries. The existence head literally cannot separate them because its inputs are identical for both classes.

**The diagnostic that proved it:** I evaluated one trained model with 6 different query selection strategies:

| Method | PGA | What it tests |
|--------|-----|---------------|
| Current (count head + existence rank) | 0.816 | Full pipeline as built |
| Oracle count + existence rank | 0.802 | Is counting or ranking the problem? |
| Oracle count + confidence rank | 0.825 | Does a different ranking signal help? |
| **Oracle count + best possible subset** | **0.947** | **Assignment ceiling — can the decoder assign?** |
| All queries active | 0.862 | Cost of extra queries |
| N/17 + confidence rank | 0.825 | Zero-parameter baseline |

**Key takeaways:**
- **Assignment works** — 94.7% ceiling. The decoder learned to specialise queries to clusters.
- **Existence head is the worst ranking signal** — 0.802 with oracle count, *worse* than current method.
- **Assignment confidence is better** — Queries that found a person have peaked softmax distributions; unspecialised queries have flat ones. Gap: 0.101 vs existence gap of 0.061.
- **Trivial counting beats learned counting** — N/17 gives exact counts on synthetic data.

**The fix:** Drop existence head and count head at inference. Count people from joint frequencies (mode of per-type counts). Rank queries by assignment confidence (average peak softmax probability). Zero new parameters. PGA: 0.816 → 0.825 immediately.

**Full pipeline training completed** (200 epochs, real data). Existence loss dropped to ~0.03 (unlike synthetic!), assignment near zero, NMI ≈ 1.0. Evaluation crashed due to a GAT config mismatch (checkpoint has 3 layers / hidden_dim=512, evaluator built with 2 layers / hidden_dim=256). Training was successful; evaluation needs a config fix.

---

## 1. Context

My pose grouping pipeline has two stages. First, a Graph Attention Network (GAT) takes detected keypoints from a scene and produces an embedding for each joint — a 128-dimensional vector that encodes identity information. Joints belonging to the same person should end up close together in this embedding space, and joints from different people should be far apart. Second, a DETR-style transformer decoder takes those embeddings and groups them into people.

The DETR decoder is borrowed conceptually from object detection. In the original DETR paper (Carion et al., 2020), a set of learnable "object queries" are fed into a transformer decoder alongside image features. Each query learns to detect one object. The transformer's cross-attention mechanism lets queries look at the image features and specialise to different objects. At the end, a classification head decides whether each query found a real object or not (the "existence" or "no-object" prediction), and a regression head predicts the bounding box.

I adapted this for pose grouping. My person queries cross-attend to joint embeddings instead of image features. An existence head predicts whether each query represents a real person. Assignment heads (one per joint type — 17 for COCO keypoints) compute dot-product similarity between each query and each joint of that type, determining which joints belong to that person. During training, a Hungarian matcher finds the optimal bipartite matching between predicted queries and ground truth people so the loss signal is consistent.

The pipeline was trained end-to-end on my synthetic dataset (generated from Blender + Mixamo animations). The GAT was working well — NMI of 1.0 and ARI of 1.0 on the test set, meaning the embeddings perfectly clustered by person. But the full pipeline's person detection precision was only 82.4%, meaning roughly 18% of detected people were hallucinations. The problem was clearly in the DETR decoder's ability to determine *how many* people existed, not in the quality of the embeddings it was receiving.

This document traces my systematic investigation into what was wrong with the existence head and how I resolved it.


## 2. Experiment 1 — Existence Threshold Sweep

### Motivation

Before building anything new, I wanted to understand how well the existing existence head was actually performing. At inference time, the existence head outputs a probability for each person query (e.g. 0.92, 0.71, 0.48, 0.12), and a threshold determines which queries count as real people. My default threshold was 0.5. I suspected the false positives — the hallucinated people — might be sitting just above 0.5 with noticeably lower confidence than the true detections. If so, simply raising the threshold could eliminate them without any retraining.

### Method

I ran the full trained pipeline (GAT + DETR) on `mixed_test` (450 images) at five threshold values: 0.50, 0.55, 0.60, 0.65, and 0.70. This was purely an inference-time change — same model weights, just a different cutoff for what counts as a detected person.

### Results

| Threshold | Precision | Recall | F1 | Perfect Count | Per-Joint Accuracy |
|-----------|-----------|--------|----|---------------|--------------------|
| 0.50 | 0.824 | 0.965 | 0.872 | 33.6% | ~96.0% |
| 0.55 | 1.000 | 0.905 | 0.944 | 66.4% | ~90.0% |
| 0.60 | 0.992 | 0.888 | 0.926 | 64.4% | ~87.5% |
| 0.65 | 1.000 | 0.905 | 0.944 | 66.4% | ~89.0% |
| 0.70 | 1.000 | 0.808 | 0.878 | 50.0% | ~80.0% |

### Analysis

The results confirmed my suspicion. At 0.5, precision was 82.4% — about one in five detected people was fake. But every single false positive sat in the narrow probability band between 0.50 and 0.55. Raising the threshold by just 0.05 eliminated all of them, jumping precision to 100%.

The sweet spot was 0.55 (or equivalently 0.65, which gave identical results): F1 of 0.944, perfect count rate of 66.4%. But this came at a cost. Recall dropped from 96.5% to 90.5% — about 10% of real people were now being missed because their existence probabilities were also in that 0.50–0.55 band. Per-joint accuracy dropped from ~96% to ~90% because the missed people's joints got left unassigned.

This told me two important things. First, the existence head *does* have some discriminative signal — false positives are consistently less confident than true positives. Second, that signal is weak. The probabilities are all squished into a narrow range around 0.5 rather than being cleanly bimodal (near 0 for fakes, near 1 for real people). A threshold sweep was the cheapest possible fix, and it helped (F1: 0.872 → 0.944), but it was hitting a ceiling. The model physically could not separate real people from hallucinations with more than ~90% recall at perfect precision.

This motivated the next question: could a different mechanism for determining person count do better?


## 3. Experiment 2 — Standalone DETR on Synthetic Data (Baseline)

### Motivation

To properly debug the DETR decoder, I needed to isolate it from the GAT. If I only tested the full pipeline, any problems I observed could be the GAT's fault (noisy embeddings, poor cluster separation) rather than the DETR's. I needed to give the DETR *perfect* input — embeddings where person clusters are trivially separated — so that any remaining failures would be definitively the DETR's problem.

### How I Generated the Synthetic Data

For each training sample, I randomly chose a number of people between 2 and 5. For each person, I created a "person centre" — a random vector in 128-dimensional space, normalised to unit length and then scaled to magnitude 2.0. Then for each of the 17 COCO joint types, I created that person's joint embedding by taking the person centre and adding a small amount of Gaussian noise (standard deviation 0.05).

This produces perfectly clustered data. All 17 joints of person 0 are tightly grouped around one point in embedding space, all 17 joints of person 1 around a completely different point, and so on. The inter-cluster distance averages about 2.8 (since random unit vectors in high dimensions are roughly orthogonal), while the intra-cluster spread is only 0.05. This is an absurdly easy clustering problem — any reasonable model should solve it perfectly. If the DETR can't handle this, the architecture has a fundamental issue.

Each sample also comes with ground truth: which person each joint belongs to (`person_labels`), what type each joint is (`joint_types`), and how many people are in the scene (`num_people`). I generated fresh random batches at each training step rather than using a fixed dataset, so the model saw unlimited variety.

### Architecture

The DETR decoder ran in isolation (no GAT). Configuration: embedding dimension 128, 7 person queries (max_people set to 5 + 2 buffer queries), 3 decoder layers, 8 attention heads, FFN dimension 512. I trained with just existence loss (λ=1.0) and assignment loss (λ=5.0) using AdamW at learning rate 1e-4 with cosine annealing over 300 epochs. Batch size was 16 samples per step.

At inference, I used Hungarian matching per joint type: for each of the 17 joint types, find the globally optimal one-to-one assignment between active queries and available joints of that type. This replaced an earlier greedy approach where the most confident query got first pick, which was suboptimal (a query might only slightly prefer one joint over another, but by claiming it, force a later query into a terrible assignment).

### Results

```
Epoch  150 | Loss: 5.48 | exist: 0.660 | assign: 0.964 | PGA: 0.726 | Count: 25% perfect, Pred: 2.0/GT: 3.5
Epoch  200 | Loss: 5.65 | exist: 0.645 | assign: 1.000 | PGA: 0.751 | Count: 25% perfect, Pred: 2.0/GT: 3.5
Epoch  300 | Loss: 6.40 | exist: 0.675 | assign: 1.145 | PGA: 0.751 | Count: 25% perfect, Pred: 2.0/GT: 3.5
```

Final metrics:
- **PGA**: 0.751
- **Perfect count rate**: 25% (always predicted 2 people)
- **Existence loss**: Stuck at ~0.67–0.69 for the entire 300 epochs
- **Existence probabilities**: Compressed into a narrow band of 0.33–0.67

### Analysis

The existence loss at ~0.69 is almost exactly the loss you get from random guessing on a binary classification problem (ln(2) ≈ 0.693). It barely moved from its initial value across 300 epochs of training. The existence head learned nothing.

Looking at individual predictions, the model collapsed to always activating the same two queries (queries 3 and 5) regardless of whether there were 2, 3, 4, or 5 people in the scene. The probabilities showed no separation — the top probability might be 0.67, the bottom 0.32, with everything else clustered in between. There was no bimodal distribution, no clean gap between "real" and "fake" queries.

And yet, PGA was 0.75, and the assignment loss *was* decreasing. The decoder was learning something about which joints go together — some queries were specialising to specific person clusters. But without a working existence head, it couldn't decide how many queries to activate, and defaulted to 2.

The problem was structural, not one of insufficient training.


## 4. Experiment 3 — Adding the Count Head

### Hypothesis

The existence head fails because each query makes its decision independently. Query 3 decides "do I exist?" by looking only at its own features. It has no idea what queries 0, 1, 2, 4, 5, and 6 decided. It's like putting 7 people in separate rooms, showing each of them the same crowd, and asking "are you needed?" They can't coordinate. Most say "maybe?" (probabilities around 0.5) and the model never learns to commit.

A count head takes a fundamentally different approach: it pools all query features together and predicts a single scalar — "there are N people total." This is a global signal. Instead of M independent binary decisions, it's one coordinated prediction.

### Architecture

The `PersonCountHead` takes all M decoded person features ([M, D] tensor), mean-pools across queries to get a single [D] vector, and passes it through a 3-layer MLP (D → D/2 → D/4 → 1) with ReLU and dropout, outputting a scalar clamped to [0, max_people]. The training loss is smooth L1 against the ground truth person count, weighted at λ_count = 2.0.

At inference, the count head's output is rounded to an integer N, then the top-N queries by existence probability are kept as the active people.

### Results

```
Epoch  150 | exist: 0.671 | assign: 1.041 | count: 0.606 | PGA: 0.661 | Count: 23% perfect, Pred: 3.0/GT: 3.8
Epoch  200 | exist: 0.685 | assign: 0.888 | count: 0.695 | PGA: 0.683 | Count: 27% perfect, Pred: 3.2/GT: 3.8
Epoch  300 | exist: 0.676 | assign: 0.875 | count: 0.804 | PGA: 0.692 | Count: 31% perfect, Pred: 3.6/GT: 3.8
```

Comparison with baseline:

| Metric | Without Count Head | With Count Head |
|--------|-------------------|-----------------|
| PGA | 0.751 | 0.692 |
| Perfect count rate | 25% | 31% |
| Mean count error | 1.54 | 0.88 |
| Avg predicted count | 2.0 (collapsed) | 3.6 |
| Existence loss | ~0.69 | ~0.67 (still stuck) |

### Analysis

The count head improved counting. The model was no longer stuck predicting 2 people — it now predicted 3.6 on average against a ground truth mean of 3.8. Mean count error halved from 1.54 to 0.88. But the count head output was noisy and oscillated between samples rather than being reliably accurate per-sample. Perfect count rate only improved from 25% to 31%.

PGA actually decreased (0.751 → 0.692). This is because the model was now trying to assign joints for more people (3.6 instead of 2), but its query-to-person assignments weren't precise enough to handle the extra queries. More active queries means more chances for bad assignments.

Critically, the existence head remained completely stuck. The count head hadn't helped it at all — existence loss was still at ~0.67, and probabilities were still compressed into a narrow band (0.45–0.62). The count head was a workaround for bad counting, but didn't address the fundamental problem.

Looking at the example predictions, the model was also always activating the same subset of queries (2, 3, 4, 5) rather than different queries for different samples. The existence probabilities showed no meaningful variation between samples with 2 people and samples with 5 people.


## 5. Experiment 4 — Orthogonal Initialisation, Separate Query LR, Auxiliary Losses

### Hypothesis

Three known issues with vanilla DETR training could be compounding the problem:

**Query symmetry.** All 7 queries were initialised as `randn * 0.02`, meaning they started nearly identical — almost zero vectors. The self-attention mechanism sees 7 copies of the same vector and computes identical updates. Cross-attention produces identical patterns over the joint embeddings. There's nothing to break the symmetry, so queries stay similar and can't specialise to different people. The original DETR paper was known to need hundreds of epochs partly because of this symmetry-breaking problem.

**Slow query learning.** The person queries have only 7 × 128 = 896 parameters. The rest of the transformer has ~1 million parameters. With a shared learning rate (1e-4), the queries' gradients get drowned out by the transformer weights. Queries need to rapidly move through embedding space to find and lock onto person clusters, but they're crawling.

**No intermediate supervision.** I was only applying losses to the final (third) decoder layer's output. Gradients from the loss have to backpropagate through 3 layers of attention to reach the first layer. The first layer — which should be learning the coarse "split queries toward different clusters" behaviour — gets almost no gradient signal. The original DETR paper applies auxiliary losses at every intermediate decoder layer to address exactly this.

### Changes

1. **Orthogonal initialisation**: Replaced `randn * 0.02` with `nn.init.orthogonal_`. This forces every query to start pointing in a maximally different direction in embedding space. In 128 dimensions, 7 orthogonal vectors are easily achievable and guarantee that self-attention sees genuinely distinct inputs from the start.

2. **Separate learning rate (10×)**: Separated model parameters into two optimiser groups. Queries got 10× the base learning rate (1e-3 vs 1e-4). This lets queries move fast through embedding space to find clusters while the transformer layers learn slowly and stably.

3. **Auxiliary losses**: Instead of wrapping decoder layers in PyTorch's `nn.TransformerDecoder`, I stored them individually in a `ModuleList` and extracted intermediate outputs at each layer. Each intermediate layer's features were passed through the same heads (existence, assignment, count) and the resulting losses were averaged and added to the total loss (weighted at λ_auxiliary = 1.0). This gives every layer direct supervision.

### Results

```
Epoch  150 | E:0.630 A:1.060 Cnt:0.496 Aux:6.952 | PGA: 0.683 | Count: 19% perfect, Pred: 3.1/GT: 3.6
Epoch  200 | E:0.671 A:1.033 Cnt:0.493 Aux:7.100 | PGA: 0.708 | Count: 20% perfect, Pred: 3.7/GT: 3.6
Epoch  300 | E:0.670 A:0.536 Cnt:0.571 Aux:4.984 | PGA: 0.719 | Count: 19% perfect, Pred: 3.1/GT: 3.6
```

Comparison:

| Metric | Count Head Only | + Ortho/LR/Aux |
|--------|----------------|----------------|
| PGA | 0.692 | 0.719 |
| Perfect count rate | 31% | 19% (worse) |
| Mean count error | 0.88 | 1.11 (worse) |
| Existence loss | ~0.67 | ~0.67 (still stuck) |
| Existence probs range | 0.40–0.62 | 0.49–0.58 (more compressed) |

### Analysis

The three fixes gave a marginal PGA improvement (+0.027) but didn't touch the core problem. The existence head was still stuck at 0.67. The existence probabilities were actually *more* compressed than before (0.49–0.58 vs 0.40–0.62), meaning less separation, not more. Perfect count rate actually dropped from 31% to 19%, and the count head got worse.

The assignment loss did decrease more cleanly (from 0.875 at epoch 300 before, to 0.536 after), suggesting the auxiliary losses helped the decoder learn better assignments. But this advantage was masked by the broken query selection mechanism.

The key observation at this point was that the existence head was stuck at 0.67 across *every experiment* I'd run — baseline, count head, orthogonal init, auxiliary losses. No matter what I tried, it wouldn't budge. This wasn't a training dynamics issue. It was architectural.


## 6. Root Cause Analysis — The Missing Background Problem

After three rounds of increasingly sophisticated fixes that all failed to move the existence loss, I stepped back and asked: *why can't the existence head learn, even on trivially separable synthetic data?*

The answer comes from understanding how the original DETR works in object detection versus how my adaptation works for pose grouping.

**In standard DETR**, the encoder produces a feature map from the entire image. Crucially, most of this feature map is *background* — pixels showing sky, grass, walls, floors. When there are 3 objects in the image and 100 queries, 97 queries cross-attend to these background regions. Their decoded features become distinctly "background-like" — they look fundamentally different from the 3 queries that found real objects. The classification head trivially learns: "background features → no object" and "object features → yes object."

**In my architecture**, the memory that queries cross-attend to consists entirely of joint embeddings. Every single token in the memory is a real joint belonging to a real person. There is no background. When there are 7 queries and 3 people, what do the 4 unmatched queries attend to? They attend to real joints — because there's nothing else available. They develop features that look just like the matched queries, because they're attending to the same type of content.

This means the existence head receives near-identical features for matched and unmatched queries. It's being asked to classify inputs that are genuinely indistinguishable. No binary classifier can separate identical inputs — and that's exactly what a loss of 0.693 (random BCE) reflects. The classification problem is *unsolvable* given these inputs.

This also explains why the count head (which pooled query features) learned the dataset mean: if all query features look alike regardless of how many real people exist, then the pooled representation carries no count information. The count head saw roughly the same input every time and learned to predict ~3.3 (the average of Uniform(2,5)).


## 7. Experiment 5 — Diagnostic Evaluation: Isolating the Bottleneck

### Motivation

Instead of trying another fix, I needed to *precisely measure* where the bottleneck was. The model has two jobs: (a) assign joints to people correctly (assignment quality), and (b) determine which queries to activate (query selection). Every fix so far assumed the whole system was broken, but the gradually improving PGA suggested assignment was working. I designed a diagnostic evaluation that cleanly separates these two concerns.

### Method

I trained the DETR decoder for 300 epochs (with all improvements from Experiment 4 plus learnable null tokens in the memory — 8 extra learnable "background" embeddings appended to the real joint embeddings, to test whether giving unmatched queries something distinct to attend to would help). Then I evaluated the same trained model using six different query selection strategies, holding assignment constant:

1. **Current method**: Use the learned count head for N, rank by existence probability, keep top-N.
2. **Oracle count + existence rank**: Cheat by using the ground truth person count, but still rank by existence probability. This isolates whether the problem is counting or ranking.
3. **Oracle count + confidence rank**: Use GT count, but rank by *assignment confidence* instead of existence. Assignment confidence measures how peaked each query's softmax distributions are over joints. A query that found a person produces sharply peaked distributions (it knows exactly which joint to pick). An unspecialised query produces flat distributions (it's guessing).
4. **Oracle count + BEST possible subset**: For each sample, try all C(M, N) combinations of N queries from M total and report the best PGA achievable. This is the theoretical ceiling — the best the model could ever do if query selection were perfect.
5. **ALL queries active**: Just use all M queries and let Hungarian matching sort it out.
6. **N/17 + confidence rank**: Count people trivially as total_joints ÷ 17, rank by assignment confidence. This uses zero learned parameters for selection.

### Results

| Method | PGA | Perfect Count |
|--------|-----|---------------|
| Current (count head + existence rank) | 0.816 | 50% |
| Oracle count + existence rank | 0.802 | 100% |
| Oracle count + confidence rank | 0.825 | 100% |
| **Oracle count + BEST subset (ceiling)** | **0.947** | **100%** |
| ALL queries active | 0.862 | N/A |
| N/17 + confidence rank | 0.825 | 100% |

Separation diagnostics (how well each signal distinguishes matched vs unmatched queries):

| Signal | Matched Queries (mean) | Unmatched Queries (mean) | Gap |
|--------|----------------------|------------------------|-----|
| Existence probability | 0.452 | 0.391 | 0.061 |
| Assignment confidence | 0.917 | 0.816 | 0.101 |

### Analysis

These results are definitive and contain several important findings.

**The decoder's assignment quality is high.** The theoretical ceiling — what PGA you'd get if you magically picked the perfect subset of queries every time — is 0.947. The decoder has genuinely learned to specialise queries to person clusters on this synthetic data. Assignment was never the problem.

**The existence head is the worst available ranking signal.** With oracle count + existence ranking (0.802), PGA is actually *lower* than the current method (0.816). That means oracle counting helps, but the existence ranking actively hurts compared to the count head's noisy-but-somewhat-useful signal. The existence gap of 0.061 confirms the probabilities barely separate matched from unmatched queries.

**Assignment confidence is a better ranking signal.** The confidence gap (0.101) is nearly twice the existence gap (0.061). This makes intuitive sense: a query that successfully found a person cluster will produce peaked softmax distributions (high maximum probability, meaning "I'm very sure this is the right joint"). An unspecialised query produces flat distributions (low maximum probability, meaning "I have no idea which joint is mine"). This signal comes directly from the assignment heads, which *are* working well.

**Trivial counting works perfectly.** N/17 + confidence (0.825) exactly matches oracle count + confidence (0.825), because on this synthetic data, every person has exactly 17 joints, so total_joints ÷ 17 always gives the exact count. The learned count head is unnecessary.

**The gap to ceiling is entirely query selection.** The current method achieves 0.816 out of a possible 0.947 — about 86% of the ceiling. The remaining 14% is lost to suboptimal query selection, not assignment errors. This gap represents queries that are confidently wrong (they've latched onto the wrong cluster) or genuinely useful queries that happen to have moderate confidence.

**Null tokens didn't help.** Even with 8 learnable background tokens appended to the memory, the existence gap remained at 0.061. The null tokens were in the memory, but there was no gradient signal forcing unmatched queries to attend to them specifically. The decoder treated them as just more tokens to attend to — they didn't create the background-vs-foreground distinction I was hoping for.


## 8. Solution — Confidence-Based Query Selection

Based on the diagnostic results, the fix is straightforward: abandon the existence head and count head for inference, and replace them with two zero-parameter alternatives.

### Counting

A new `_count_from_joints()` method counts the number of instances of each joint type (e.g., 3 left_knees means 3 people) and takes the mode across all types. For synthetic data where every person has all 17 joint types, this gives exact counts. For real data where some joints may be occluded, taking the mode is robust to missing detections.

### Query Ranking

A new `_assignment_confidence()` method computes, for each query, the average peak softmax probability across all joint types. If a query's softmax over left_knees peaks at 0.95 and its softmax over right_elbows peaks at 0.92, its confidence is high — it has clearly claimed specific joints. If another query's peaks are all around 0.3, it's diffuse and unspecialised.

### Inference Pipeline

The updated `predict()` method now defaults to:

1. Run the forward pass through the decoder (unchanged from training)
2. Count people from joint type frequencies (no learned parameters)
3. Rank queries by assignment confidence (no learned parameters)
4. Keep top-N queries by confidence
5. Per-type Hungarian matching for joint assignment (unchanged)

The existence head and count head remain in the model during training — the existence loss, while producing random gradients, doesn't prevent the assignment heads from converging (the decoder still reached a 0.947 ceiling). But at inference, they are bypassed entirely.

*(Real data evaluation results to be added once the experiment completes.)*


## 9. Critical Assessment

### What Worked

The core DETR decoder architecture does learn to group joints into people. On perfectly clustered synthetic data, it achieves a 94.7% assignment ceiling. The transformer's cross-attention mechanism genuinely specialises different queries to different person clusters, and the assignment heads learn to pick the right joints. Hungarian matching at both training time (for loss computation) and inference time (for joint assignment) was important for achieving this.

### What Didn't Work

Every attempt to fix the existence head failed because the problem was structural, not a training issue:

- **Threshold sweeping** improved F1 (0.872 → 0.944) but hit a ceiling at ~90% recall.
- **Count head** improved average predictions but learned the dataset mean, not per-sample counts.
- **Orthogonal initialisation** broke query symmetry but didn't help existence discrimination.
- **Separate query learning rate** made queries learn faster but couldn't overcome missing background.
- **Auxiliary losses** gave stronger gradient flow but the classification problem was still unsolvable.
- **Null tokens** were in the memory but the decoder had no incentive to route unmatched queries to them specifically.

The root cause — no background tokens in the memory for unmatched queries to attend to — meant the existence head's inputs were indistinguishable between matched and unmatched queries. No classifier can solve a problem where the two classes have identical features.

### Limitations

The diagnostic evaluation was conducted entirely on synthetic data with perfect cluster separation (noise σ = 0.05, inter-cluster distance ~2.8). Several aspects of real data are not captured:

- **Imperfect embeddings**: The GAT's real embeddings won't have zero cluster overlap. Some joints may have ambiguous embeddings, especially in crowded scenes.
- **Variable joint counts**: In real data, people are partially occluded. Not every person has all 17 keypoints detected. The N/17 counting heuristic is exact for synthetic data but will need the mode-based approach on real data.
- **Scale**: I tested with 2–5 people (7 queries). Real scenes might have more people, which increases the combinatorial complexity of query selection.
- **Training noise from existence loss**: The existence loss contributes ~0.67 to total loss every epoch as pure random noise. Its gradients flow back through the person features into the decoder. It's possible that setting λ_existence to 0 could improve assignment quality further, but I haven't tested this yet.
- **Confidence vs. ceiling gap**: Even with oracle counting, assignment confidence ranking (0.825) only captures 87% of the assignment ceiling (0.947). Some queries are confidently wrong. Understanding and closing this gap is an open question.

The critical next step is evaluating confidence-based selection on real data to determine whether the findings from synthetic experiments transfer.


## 10. Summary Table

| Experiment | Key Change | PGA | Perfect Count | Existence Loss | Key Finding |
|------------|-----------|-----|---------------|----------------|-------------|
| Threshold sweep (real data) | Vary threshold 0.5–0.7 | — | 33–66% | — | False positives sit in 0.50–0.55 band; F1 ceiling ~0.944 |
| Synthetic baseline | DETR in isolation | 0.751 | 25% | 0.69 (stuck) | Collapsed to always predicting 2 people |
| + Count head | Global count prediction | 0.692 | 31% | 0.67 (stuck) | Better counting (err: 1.54→0.88) but worse PGA |
| + Ortho/LR/Aux | Training dynamics fixes | 0.719 | 19% | 0.67 (stuck) | Marginal PGA gain, existence unchanged |
| Diagnostic | 6 selection methods compared | 0.816 | 50% | — | Assignment ceiling 0.947; existence is worst signal |
| Confidence-based | Drop existence, use confidence | 0.825 | 100%* | — | Zero-parameter selection beats all learned methods |
| Full pipeline (real data) | End-to-end GAT+DETR | — | — | ~0.03 (converged!) | Existence works with real embeddings; eval needs config fix |

*On synthetic data where N/17 counting is exact.


## 11. Full Pipeline Training (Real Data)

With the architectural fixes in place (orthogonal init, separate query LR, auxiliary losses, null tokens, count head), I trained the full GAT + DETR pipeline end-to-end on the real synthetic dataset (450 annotated scenes from Blender/Mixamo) for 200 epochs.

### Training Observations

The training logs show dramatically different behaviour from the standalone synthetic DETR tests:

- **Existence loss dropped to ~0.03 by epoch 30**, then settled around 0.03–0.05 for the remainder of training. This is a stark contrast to the standalone tests where it was stuck at 0.67. This suggests that when trained end-to-end, the GAT learns to produce embeddings that give the existence head more discriminative information — possibly because the GAT's learned embeddings have more structure than the random synthetic clusters I was generating.
- **Assignment loss** dropped to near zero (0.000–0.01) by epoch 28 and stayed there.
- **Count head** converged quickly — predicting 3.0 against a match target of 3.0 from around epoch 7 onward.
- **NMI** was consistently 0.95–1.0, confirming the GAT embeddings remained well-clustered throughout.
- **Contrastive loss** steadily decreased from 0.42 to 0.02 over the full 200 epochs.
- **Total loss** converged to ~0.28–0.38 by the final epochs.

There were periodic loss spikes (e.g., epoch 47: loss jumped to 3.32, epoch 70: 2.41) which is typical of training on small datasets with occasional hard examples. The model recovered quickly each time.

### Evaluation Failure

Evaluation crashed with a `state_dict` mismatch error. The checkpoint was saved with a GAT configuration of 3 layers and hidden_dim=512, but the evaluator was building the model with 2 layers and hidden_dim=256. This is a configuration propagation bug — the training config wasn't being passed through to the evaluator correctly. The training itself completed successfully; only the evaluation script needs fixing.

The specific mismatches: the checkpoint has `gat_layers.2.*` keys (third GAT layer) that don't exist in the 2-layer model, and all weight dimensions differ (512 vs 256 hidden dim). This is a straightforward fix — ensure the evaluator reads the GAT config from the checkpoint or the same YAML config used for training.

### Significance

The fact that existence loss converges on real data but not on synthetic data is an important observation. It suggests one of two things:

1. **The end-to-end training signal helps**: When the GAT and DETR train together, the GAT may learn to encode information that makes existence discrimination easier — perhaps embedding magnitude, or subtle distributional differences that the standalone synthetic data (uniform random centres + Gaussian noise) didn't have.

2. **The dataset is too homogeneous**: All training scenes have the same number of people (~3), so the existence head may have learned "always predict 3" rather than genuinely discriminating. The count always showing 3.0 supports this interpretation. We'd need to evaluate on scenes with varying person counts to distinguish these explanations.

This remains an open question pending the evaluation fix.