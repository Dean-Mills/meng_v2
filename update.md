Yes, that's exactly the point — and it's actually the core argument for why this is better than a two-stage approach.
Graph partitioning — the affinity prediction is just an MLP on pairs of GAT embeddings, trained with binary cross-entropy (same person = 1, different = 0). Gradients flow back through the affinity head into the GAT, so the GAT learns to produce embeddings where same-person joints are pulled together and different-person joints are pushed apart. The GAT is essentially learning a pose-aware metric space optimized for partitioning.
Slot attention — fully differentiable by design. The slots compete via softmax attention to "claim" joints, and the loss (whatever you use — Hungarian matching against ground truth groups) propagates back through the attention weights into the GAT. Same effect — the GAT learns embeddings that are easy for slots to cleanly separate.
The key insight worth highlighting in your thesis is exactly what you said: the grouping objective shapes the embedding space. You're not training the GAT to produce good embeddings independently and then grouping them separately. The whole thing is trained jointly so the GAT learns representations that are inherently group-discriminative. That's the research contribution — the end-to-end formulation where joint embedding and person grouping are co-optimized.
Compare that to classical approaches where you'd use a pre-trained pose estimator, extract features separately, then cluster — those are decoupled and the features aren't optimized for grouping at all.

Between the two I'd lean toward slot attention as your primary approach with graph partitioning as a comparison baseline. Slot attention is more recent, less explored in pose tracking, and the "no K required" property is genuinely useful. Graph partitioning is well understood so it makes a clean ablation — "here's what you get with a simpler grouping head vs slot attention."
Does that framing feel right for what you want to argue in the thesis?



