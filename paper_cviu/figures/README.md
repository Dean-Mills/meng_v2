# Figure index for the CVIU paper

Every candidate figure, whether or not it ends up in the paper. Status: **new** = made for the paper (2026-09-10),
**diss** = exists in `dissertation/figures/` (would need a two-column re-layout), **sup** = supplement or cut candidate.
"Print size" is the check against the 522 pt text width (figure\*) or the 252 pt column with the 8 pt font floor.
Regenerate any `new` figure from `code_v3/` with the script named (all write a `.pdf`, a `.png` preview and,
where numbers are plotted, a `.json` sidecar with every plotted value).

| # | File | Status | Section (skeleton §12) | What it shows | Source / data | Print size |
|---|---|---|---|---|---|---|
| F1 | `concept_modular.pdf` | new | Intro, Fig. 1 | (a) joint-trained grouping inside one detector; (b) three detectors → keypoint-only interface → one frozen PG-GAT + K-head + k-means; real insets from COCO 281759 | `src/concept_modular.tex` (TikZ) + `src/fig_concept_insets.py`; insets `concept_inset_ungrouped/grouped.pdf` | figure\*, 540 pt natural → scale 0.97, text 8.7 pt ✓ |
| F2 | `edge_categories.pdf` | new | Method 3.2/3.3 | the four edge categories from one anchor joint, the same-type edge to the other person, hop distances; head joints have no skeleton path (16-edge table, guide §5 item 13) | `src/fig_edge_categories.py` (tables from `sa_gat.py`) | column, 3.45 in ✓ |
| F3 | `attention_edges.pdf` | new | Method 3.3 or Discussion 6.1 | (a) attention share by edge category vs uniform, baseline vs PG-GAT; (b) within-person attention per layer; (c) top-2 attended edges on scene 554328 | `src/fig_attention_edges.py` (1,248 multi-person val scenes; `--redraw` reuses the JSON + npz) | figure\*, 7.2 in ✓ |
| F4 | `embedding_space_pca.pdf` | new | Discussion 6.1 | scene 554328, three checkpoints: PCA of the embeddings (misassigned ringed) + cosine-similarity matrices | `src/fig_embedding_space.py --proj pca --img_id 554328` | figure\*, 7.2 in ✓ |
| F4b | `embedding_space_both.pdf`, `embedding_space_tsne.pdf` | sup | - | same with a t-SNE row / t-SNE only; t-SNE exaggerates separation and is not reproducible run to run | same script, `--proj both` / `tsne` | figure\* |
| F5 | `detector_swap.pdf` | new | Results 5.4 | scene 281759 × HigherHRNet / YOLO11x / RTMO-L: pooled keypoints, native grouping, frozen PG-GAT grouping (all PGA 1.00 on this scene) | `src/fig_detector_swap.py` (HRNet from the AP cache; YOLO, RTMO run live) | figure\*, 7.2 in ✓ |
| F6 | `crowd_size.pdf` | new | Results 5.5 / Limitations | (a) end-to-end PGA and (b) AP per annotated person count 1..8+, native vs PG-GAT, K ≤ 5 range marked, n per bin | `src/fig_crowd_size.py` (per-image dump + AP results) | column, 3.45 × 4.0 in ✓ |
| D1 | `dissertation/figures/e2e_interface.pdf` | diss | Method 3.1 | the detector-independent interface block diagram (three detectors, pooling, PG-GAT, K-head, evaluation branch) | `dissertation/figures/src/e2e_interface.tex` | re-layout for 522 pt; check fonts |
| D2 | `dissertation/figures/pggat_architecture.pdf` | diss | Method 3.3 | the PG-GAT layer: node features, edge encoder, standard + repulsion GATv2Conv, projection | `dissertation/figures/src/pggat_architecture.tex` | re-layout; could absorb F2 |
| D3 | `dissertation/figures/autonomous_pipeline.pdf` | diss / sup | Method 3.5 | the autonomous inference pipeline; largely duplicated by F1(b) and D1 | TikZ | probably cut |
| D4 | `dissertation/figures/e2e_examples.pdf` | diss | Results 5.4 | four scored scenes × (AE, PG-GAT oracle K, PG-GAT K̂); already at exemplar level | `make_figures.py fig_e2e_examples` | 6.1 in → figure\* |
| D5 | `dissertation/figures/predk_recall.pdf` | diss | Results 5.4 / Discussion 6.3 | predicted-K advantage vs keypoint recall, four operating points (sign flip) | `make_figures.py` | column ✓ |
| D6 | `dissertation/figures/transfer_scatter.pdf` | diss | Discussion 6.2 | synthetic-val vs COCO-val PGA for 27 configurations with fine-tuning arrows | `make_figures.py` | column or figure\* |
| D7 | `dissertation/figures/k_confusion.pdf` | diss / sup | Discussion 6.3 | K-head confusion matrix on HigherHRNet detections | `make_figures.py` | column ✓ |
| D8 | `dissertation/figures/skeleton_graph.pdf` | diss / sup | Method 3.2 | COCO 17-joint skeleton + kNN graph on a two-person synthetic scene; overlaps F2 | `make_figures.py` | column |
| D9 | `dissertation/figures/synth_examples.pdf` | diss | Setup 4.1 | four synthetic scenes, K = 2..5 | `make_figures.py` | column |
| D10 | `dissertation/figures/context_example.pdf` | diss / sup | - | one COCO image ungrouped vs grouped; superseded by F1 | `make_figures.py` | - |
| D11 | `dissertation/figures/arch_sweep.pdf`, `ft_sweep.pdf` | sup | supplement | the 32-run architecture sweep and the 12-run fine-tune sweep | `make_figures.py` (W&B) | - |

## Numbers behind the new figures (also in the JSON sidecars)

- F3: PG-GAT standard heads, attention share vs uniform: same-type 0.000 / 0.041; skeletal neighbour 0.292 / 0.173; same limb 0.100 / 0.098; cross-body 0.519 / 0.628; self-loop 0.089 / 0.059; within-person 0.856 / 0.553 (layers 0.74 → 0.90 → 0.93). Standard GAT: within-person 0.765 / 0.717. Repulsion head: self-loop 0.675 / 0.589, same-type 0.325 / 0.411 (state in the text).
- F4: scene 554328 (K = 3, 34 keypoints; the scene closest to the dataset means among 180 candidates): per-scene PGA 0.882 / 0.912 / 1.000, misassigned 4 / 3 / 0, against COCO val PGA 0.8841 / 0.9010 / 0.9715.
- F5: scene 281759, K = 5: pooled 80 / 81 / 82 keypoints, matched 73 / 75 / 74, K̂ = 5 for all three, PGA native 1.00 and PG-GAT 1.00 for all three.
- F6: PGA per K (AE / oracle / K̂): 1: 0.999/1.000/0.999; 2: 0.987/0.935/0.971; 3: 0.980/0.944/0.941; 4: 0.977/0.934/0.929; 5: 0.971/0.922/0.898; 6: 0.969/0.923/0.921; 7: 0.968/0.880/0.879; 8+: 0.926/0.876/0.882. AP per K (native pooled / PG-GAT K̂): 1: 0.640/0.639; 2: 0.583/0.565; 3: 0.630/0.577; 4: 0.583/0.540; 5: 0.572/0.498; 6: 0.483/0.449; 7: 0.442/0.352; 8+: 0.447/0.359.

## Conventions

Person colours `#c23b3b #2b5fa3 #3c8a4e #b58a2a #7a4fa3 …`; module colours as in the dissertation diagrams (red = trained here, blue = external detector, grey = deterministic); serif (Times) fonts, ≥ 8 pt at print size; PDF for the paper, PNG for preview only. Category colours: same type red, skeletal neighbour green, same limb blue, cross-body grey.
