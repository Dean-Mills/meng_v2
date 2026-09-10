# CVIU exemplars (2024–2026): structural analysis

Purpose: how recent *Computer Vision and Image Understanding* (Elsevier) papers on pose / skeleton keypoints are **written** (structure, length, metrics, framing), as a template for a paper proposing a detector-independent pose-grouping module (graph attention on detected keypoints + k-means read-out, evaluated with PGA rather than COCO AP).

Compiled 2026-09-09 by the research agent; two scope corrections applied afterwards (comparison references and the negative-results list now name only MEng-scope items). Sources: Crossref (complete listing of CVIU records dated 2024-01-01 onward: 819 records, ISSN 1077-3142), Unpaywall, OpenAlex, OpenAIRE, Semantic Scholar, arXiv API, Ghent University Academic Bibliography, web search. Files named `<firstauthor><year>_<slug>.pdf`.

## Catalogue

| # | Title | First author | Year, vol., art. | DOI (resolves) | OA status on ScienceDirect | Local file | Source of PDF |
|---|---|---|---|---|---|---|---|
| 1 | Exploring event-based human pose estimation with 3D event representations | Xiaoting Yin (Zhejiang U.) | 2024, vol. 249, 104189 | [10.1016/j.cviu.2024.104189](https://doi.org/10.1016/j.cviu.2024.104189) | Subscription (closed) | `yin2024_event_hpe.pdf` (25 pp, arXiv layout) | arXiv [2311.04591v4](https://arxiv.org/abs/2311.04591) — same title/authors; arXiv comment: "Accepted to Computer Vision and Image Understanding" |
| 2 | Fourier analysis on robustness of graph convolutional neural networks for skeleton-based action recognition | Nariki Tanaka (Chiba U.) | 2024, vol. 240, 103936 | [10.1016/j.cviu.2024.103936](https://doi.org/10.1016/j.cviu.2024.103936) | Open access, CC BY-NC-ND (hybrid) | `tanaka2024_fourier_gcn_skeleton.pdf` (18 pp, arXiv layout) | arXiv [2305.17939v2](https://arxiv.org/abs/2305.17939) — same title/authors/abstract. ScienceDirect PDF is OA but automated download returned HTTP 403 |
| 3 | 3D Pose Nowcasting: Forecast the future to improve the present | Alessandro Simoni (U. Modena & Reggio Emilia / U. Florence) | 2025, vol. 251, 104233 | [10.1016/j.cviu.2024.104233](https://doi.org/10.1016/j.cviu.2024.104233) | Open access, CC BY (hybrid) | `simoni2025_pose_nowcasting.pdf` (11 pp, arXiv layout) | arXiv [2308.12914v2](https://arxiv.org/abs/2308.12914) — same title/authors. ScienceDirect PDF is OA but automated download returned HTTP 403 |
| 4 | An end-to-end pipeline for team-aware, pose-aligned augmented reality in cycling broadcasts | Winter Clinckemaillie (Ghent U.–imec) | 2026, vol. 263, 104602 | [10.1016/j.cviu.2025.104602](https://doi.org/10.1016/j.cviu.2025.104602) | Open access, CC BY (hybrid); special issue "CV for Sports" | `clinckemaillie2026_cycling_pose_ar.pdf` (16 pp, **published CVIU layout**, PDF/A) | Publisher version deposited at [Ghent University Academic Bibliography](https://biblio.ugent.be/publication/01KE6XXA6QTBSTTB6HPJXZPXZV) |
| 5 | Comparing Human Pose Estimation through deep learning approaches: An overview | Gaetano Dibenedetto (U. Bari) | 2025, vol. 252, 104297 | [10.1016/j.cviu.2025.104297](https://doi.org/10.1016/j.cviu.2025.104297) | Open access, CC BY (hybrid) | **not downloaded** | Only copy is the ScienceDirect PDF (no arXiv, no repository copy found via Unpaywall/OpenAlex/OpenAIRE); ScienceDirect blocks automated download (HTTP 403). Download manually from the DOI if wanted. Survey; covers 2D multi-person top-down vs bottom-up and COCO AP benchmarks |

All four downloaded files start with `%PDF-` and open with `pdfinfo`. Other files in this folder (`tvc_peng2025_*.pdf`, `README_visual_computer.md`) were written by a separate process for *The Visual Computer* and are not part of this catalogue.

### Candidates rejected or not obtainable

The complete Crossref listing of CVIU 2024–2026 was screened on title keywords (pose, keypoint, skeleton, joint, grouping, association, human, body, person, crowd, limb, articulated, part, GNN/GAT). **No CVIU paper in 2024–2026 addresses 2D bottom-up multi-person pose estimation or keypoint grouping/association.** The nearest candidates and why they are not in the analysed set:

| Candidate | DOI | Why rejected / not obtained |
|---|---|---|
| Niu, Lü, Xue, Wang — *Skeleton Cluster Tracking for robust multi-view multi-person 3D human pose estimation*, 2024, vol. 246, 104059 | 10.1016/j.cviu.2024.104059 | Closest topically (multi-person association by clustering a skeleton pool, "Skeleton Pooling–Clustering–Tracking"), but subscription only; no arXiv, repository or author-page copy found (Unpaywall/OpenAlex/OpenAIRE/S2/arXiv/web). Worth obtaining through the library |
| Son, Lee, Kim — *STELA: Spatial–temporal enhanced learning with an anatomical graph transformer for 3D human pose estimation*, 2025, vol. 257, 104381 | 10.1016/j.cviu.2025.104381 | Graph-attention over joints (GNN-on-pose), but subscription only; no OA copy anywhere |
| Wang, Wu, Kang, Liu, Yang — *MuRE: Multi-Relationship Encoder for 3D human pose estimation*, 2026, vol. 267, 104707 | 10.1016/j.cviu.2026.104707 | Subscription only; no OA copy |
| Li et al. — *Static graph convolution with learned temporal and channel-wise graph topology generation for skeleton-based action recognition*, 2024, vol. 244, 104012 | 10.1016/j.cviu.2024.104012 | Listed as green OA (U. Wollongong figshare 27818367) but the figshare record contains no files; publisher closed |
| Zhou et al. *SlowFastFormer* (vol. 243, 103992); Xiang et al. *DBMHT* (vol. 249, 104147); Xu et al. *Spatio-Temporal Dynamic Interlaced Network* (vol. 251, 104258) | — | 2D-to-3D lifting transformers on keypoint sequences; all subscription, no arXiv |
| Xiang et al. — *A GCN and Transformer complementary network for skeleton-based action recognition* (vol. 249, 104213) | 10.1016/j.cviu.2024.104213 | Closed, no OA copy; less relevant than Tanaka 2024 |
| Figari Tomenotti et al. — *Head pose estimation with uncertainty…* (vol. 243, 103999, CC BY) | 10.1016/j.cviu.2023.103999 | Off topic (head pose) |
| Mohottala et al. — *Spatio-temporal GNN based child action recognition… systematic analysis* (vol. 259, 104410) | 10.1016/j.cviu.2025.104410 | Closed; application survey |

Not verifiable: ScienceDirect article pages (highlights, graphical abstracts, published page counts of papers 1–3) — every request to sciencedirect.com returned HTTP 403 ("Are you a robot?"). Whether the arXiv texts of papers 1–3 are identical to the CVIU versions: paper 1 (v4, Sep 2024) is labelled accepted; paper 2 (v2, Dec 2023, published online Jan 2024) is very likely final; paper 3 (v2, Nov 2023) predates acceptance by roughly a year and the CVIU version may contain revisions.

---

## Per-paper structural analysis

### 1. Yin et al. 2024 — event-based HPE with 3D event representations (`yin2024_event_hpe.pdf`)

- **Layout / length.** arXiv v4 uses the elsarticle two-column CVIU preprint template (mimics the journal header). 25 pages: body p1–19, references p19–21, then four pages of supplementary figures/tables (Fig. 11–14, Table 13) after the references with no appendix heading. Journal extension of a conference paper (Chen et al., 2022) — stated explicitly on p3–4.
- **References.** ≈56 (author–year, Elsevier Harvard style; OpenAlex lists 56).
- **Abstract.** 213 words.
- **Section headings.** 1 Introduction (p1–4) · 2 Related Work (p4–6): 2.1 Human Pose Estimation, 2.2 Event-based Human Pose Estimation, 2.3 Point Clouds vs. Voxel Grid · 3 Methodology (p6–11): 3.1 Overview, 3.2 Dataset, 3.3 HPE based on 3D Rasterized Event Point Cloud, 3.4 HPE based on 3D Decoupled Event Voxel · 4 Experiments (p11–18): 4.1 Experiment Setups, 4.2 Ablation Studies, 4.3 Comparison of Event Representations · 5 Limitations (p18–19) · 6 Conclusion (p19) · References.
- **Contributions.** Two enumerated lists at the end of the introduction: four items from the conference version, then "This paper is an extension of our conference work … adding the following contributions" with five further items (new representation, new attention module, new dataset, best results, two representations shown to work on the wild dataset).
- **Related work.** Separate section; organised by task (general HPE → event-based HPE) and then by input-representation family (point cloud vs voxel). About 1.5 pages.
- **Overview figure.** Fig. 1 (p2, in the introduction) contrasts the 2D event-frame paradigm with the proposed 3D representations — a conceptual figure before the method. The actual block diagrams are Fig. 3 (RasEPC pipeline, p8) and Fig. 4 (DEV pipeline, p10), each placed at the head of its method subsection; 3.1 "Overview" is text.
- **Experimental layout.** 4.1 datasets (DHP19, MMHPSD, EV-3DPW, EV-JAAD with split protocols) and metric, then implementation details → 4.2 nine ablation tables (Tables 1–9, p12–14) → 4.3 main comparisons (Tables 10–12, p15–17) plus a latency-vs-error plot (Fig. 6) and qualitative figures (Figs 7–9). **Ablations precede the main comparison.**
- **Metrics.** MPJPE2D (pixels) and MPJPE3D (mm); latency (ms) on edge devices. No COCO AP/OKS, no PCK. Multi-person is handled top-down with given boxes, so no grouping metric.
- **Negative results / limitations.** Dedicated Section 5 "Limitations" (~170 words): only HPE explored; multi-person handled top-down without end-to-end optimisation. Competitor failure cases shown qualitatively (Fig. 9, dim garage). A noise-robustness table (Table 13) is relegated to the trailing pages.
- **Comparison framing.** The phrase "state-of-the-art" does not occur. Comparisons are controlled: different event representations "under the same backbone", with † marking the authors' own re-implementations of reference methods (DHP19, PointNet, Pose-ResNet18/50, HATS, Ev-FlowNet, EST).
- **Highlights / graphical abstract.** Not in the arXiv version; not verifiable on ScienceDirect.
- **Availability statements.** Code and dataset GitHub link in the abstract/footnote (MasterHow/EventPointPose); no formal Data availability or CRediT block in the arXiv version.

### 2. Tanaka et al. 2024 — Fourier analysis of GCN robustness on skeleton keypoints (`tanaka2024_fourier_gcn_skeleton.pdf`)

- **Layout / length.** arXiv v2 single-column preprint, 18 pages: body p1–16, references p16–18. Analysis paper (no new model).
- **References.** 34 (numbered).
- **Abstract.** 152 words, followed by a keyword line.
- **Section headings.** 1 Introduction (p1–2) · 2 Related work (p2–3): 2.1 Robustness of Skeleton-based Action Recognition, 2.2 Fourier analysis of CNN-based image classification · 3 Fourier Analysis for Skeleton-based Action Recognition (p3–6): 3.1 Spatiotemporal Graph for Skeletal Sequence Data, 3.2 Standard & Adversarial Training, 3.3 Discrete & Graph Fourier Transforms, 3.4 Joint Fourier Transform and Fourier Heatmap · 4 Experiment (p6–15): 4.1 Experimental Setting (4.1.1 Dataset, 4.1.2 Model, 4.1.3 Adversarial Attack, 4.1.4 Adversarial Training, 4.1.5 Evaluation Metric), 4.2 Results (4.2.1 Frequency Analysis of Adversarial Training, 4.2.2 Frequency Analysis of Adversarial Attack, 4.2.3 Robustness Trade-off between High- and Low-Frequency Perturbations, 4.2.4 Robustness to Common Corruptions) · 5 Conclusions (p15–16) · References.
- **Contributions.** Three bullets at the end of the introduction; the third bullet is a *negative* finding ("Challenges are revealed in comprehensively explaining the robustness … using Fourier analysis").
- **Related work.** Separate section (~1 page), two subsections by topic: the application domain's robustness literature, then the analysis-tool literature.
- **Overview figure.** Fig. 1 (p2, introduction): block-diagram flow of the joint Fourier transform on skeleton data — before the method (Section 3 starts p3).
- **Experimental layout.** Setting first (dataset NTU RGB+D, two models, attack, training, metric), then results as three analyses; Tables 1–2 (clean / adversarial accuracy) open the results, corruption tables (5–7) close them. No separate ablation subsection (the whole paper is an analysis).
- **Metrics.** Classification accuracy, but a *matched* accuracy: only samples correctly classified by both compared models are perturbed and scored (4.1.5). No AP/PCK.
- **Negative results / limitations.** No separate section; the negative finding is in the abstract, the contribution list, 4.2.4 and the Conclusions (~430 words, which also carry "remains an open problem" and future work).
- **Comparison framing.** Chosen references: "We chose ST-GCN as a baseline GCN and TCA-GCN as one of the state-of-the-art GCNs" — one baseline plus one strong model, not a leaderboard.
- **Highlights / graphical abstract.** Not in the arXiv version; not verifiable.
- **Availability statements.** Uses "official codes"; no code release, Data availability or CRediT block in the arXiv version.

### 3. Simoni et al. 2025 — 3D Pose Nowcasting (`simoni2025_pose_nowcasting.pdf`)

- **Layout / length.** arXiv v2 in CVPR-style two-column format, 11 pages: body p1–9, references p9–11.
- **References.** 65 (numbered).
- **Abstract.** 123 words.
- **Section headings.** 1 Introduction (p1–2) · 2 Related Work (p2–3) · 3 Proposed Method (p4–5): 3.1 Depth and Past Pose Input Processing, 3.2 Pose Estimation and Forecasting Branches, 3.3 Losses · 4 Experimental Validation (p5–9): 4.1 Datasets, 4.2 Metrics, 4.3 Training, 4.4 Results (paragraphs "Results on SimBa", "Results on ITOP"), 4.5 Execution Time Analysis · 5 Conclusion and Future Work (p9) · References.
- **Contributions.** Three bullets at the end of the introduction ("Summarizing, the main contributions of our paper are").
- **Related work.** Separate section (~1 page) with bold paragraph leads by task: Robot Pose Estimation from Depth; Human Pose Estimation from Depth; Pose Forecasting.
- **Overview figure.** Fig. 1 (p1) teaser; Fig. 2 "Overview of the proposed 3D Pose Nowcasting framework" on p3, before Section 3 (p4); Fig. 3 (p4) is the branch architecture.
- **Experimental layout.** Datasets → Metrics (with equation) → Training → Results (Table 1 robot pose estimation, Table 2 estimation + forecasting, Table 3 per-joint human results vs literature, Table 4 human estimation + forecasting) → execution time. No ablation subsection; the "w/o forecasting" rows inside the main tables serve as the ablation.
- **Metrics.** ADD (mean L2 joint error, cm) and mAP@δ for δ ∈ {2,4,6,8,10} cm — defined by an equation in 4.2. Note that this "mAP" is a distance-threshold accuracy (PCK-like), not COCO AP; the paper adopts the metric conventional for the ITOP benchmark.
- **Negative results / limitations.** None dedicated. The conclusion states results on ITOP are "comparable with the current literature competitors" (i.e. not best) and lists future work; no failure cases.
- **Comparison framing.** Explicit SOTA claims ("state-of-the-art results" on SimBa; Table 3 "compared to the state-of-the-art" on ITOP).
- **Highlights / graphical abstract.** Not in arXiv version; not verifiable.
- **Availability statements.** No code/data availability or CRediT block in the arXiv version.

### 4. Clinckemaillie et al. 2026 — pose-aligned AR in cycling broadcasts (`clinckemaillie2026_cycling_pose_ar.pdf`)

- **Layout / length.** Published CVIU two-column layout. 16 pages: body p1–12, CRediT/Funding/Declaration/Acknowledgments p12, Appendices A–D p12–14, Data availability p13, references p15–16. Front matter shows received/revised/accepted/online dates (19 Sep → 5 Nov → 2 Dec → 4 Dec 2025) and the CC BY line; a title footnote marks the special issue.
- **References.** ≈56 (author–year, Elsevier Harvard style).
- **Abstract.** 185 words; six keywords printed in the margin before the abstract.
- **Section headings.** 1 Introduction (p1–2) · 2 Related work (p2–3): 2.1 Cyclist detection and team recognition, 2.2 AR visualization · 3 Datasets (p3–4): 3.1 Cyclist detection, 3.2 Team recognition (Jersey crops), 3.3 CyclingTrack · 4 Methodology (p4–7): 4.1 Cyclist detection, 4.2 One-shot team recognition, 4.3 Cyclist tracking, 4.4 3D bounding box calculation, 4.5 AR visualization · 5 Experiments & results (p7–10): 5.1 Cyclist detection, 5.2 One-shot team recognition, 5.3 Cyclist tracking, 5.4 Inference time and optimization · 6 User experience evaluation (p10–12): 6.1 Study design, 6.2 Survey results, 6.3 Takeaways and implications · 7 Limitations (p12) · 8 Conclusion (p12) · CRediT · Funding · Declaration of competing interest · Acknowledgments · Appendix A–D · Data availability · References.
- **Related work.** Separate section organised by pipeline stage; each subsection ends with an explicit "Research gap." paragraph.
- **Overview figure.** Fig. 5 "Overview of the proposed pipeline" on p5, at the head of Section 4 (which starts p4) — after the datasets section and before the dense method text. Fig. 1 (p2) is an example/teaser.
- **Experimental layout.** Datasets are a stand-alone Section 3 *before* the method; results subsections mirror the method subsections one-to-one; then runtime; then a user study. Comparison tables are per component (detector variants, tracker variants). No ablation section as such.
- **Metrics.** mAP@0.5 and inference time (detection); classification accuracy (team recognition); HOTA / MOTA / IDF1 (tracking); FPS after TensorRT; 1–5 Likert scores (user study). The pose component (TokenHMR mesh recovery) is used as-is and not evaluated quantitatively. No pose AP/PCK.
- **Negative results / limitations.** Dedicated Section 7 "Limitations" (bulleted, ~200 words). The user study's negative outcome — most participants preferred conventional labels over the proposed AR overlays — is reported plainly in 6.3 and repeated in the limitations.
- **Comparison framing.** Chosen reference families (YOLOv8/YOLOv11/RF-DETR; ByteTrack-family trackers); "SOTA" appears only inside cited titles.
- **Highlights / graphical abstract.** Neither is printed in the published PDF (Elsevier shows them only on the article web page); not verifiable.
- **Availability statements.** CRediT statement (per-author roles), Funding, Declaration of competing interest, Acknowledgments, and "Data availability: Data will be made available on request." GitHub links are to third-party tools only.

---

## Cross-paper synthesis

**Typical skeleton (published two-column layout ≈ 12–16 pages; arXiv layouts 11–25).**

| Section | Content pattern | Rough length |
|---|---|---|
| Abstract | 120–215 words (median ≈ 170); keywords line follows | — |
| 1 Introduction | Problem, gap, approach, bullet list of 3–5 contributions at the end (Yin lists conference contributions and journal-extension contributions separately) | 1.5–3 pp |
| 2 Related work | Always a separate section, 2–3 subsections by task or by method/representation family; two papers end each subsection with an explicit gap statement | 1–2 pp |
| (3 Datasets) | Optional stand-alone section when datasets are a contribution (Clinckemaillie); otherwise a subsection of the method (Yin 3.2) or of experiments | 0–1.5 pp |
| 3/4 Method | Overview figure at or before the first method subsection in all four papers (p2, p3, p5, and p2/p8/p10); numbered equations; losses last | 2–5 pp |
| 4/5 Experiments | Setup first (datasets + protocols → metric definition → training/implementation), then results. Order of main-comparison vs ablation is flexible (Yin: ablations first; Simoni: ablation rows inside the main tables; Tanaka: analyses only). Efficiency/latency subsection common (3 of 4) | 4–8 pp |
| Limitations | Dedicated section in 2 of 4 (Yin §5, Clinckemaillie §7), otherwise folded into the conclusion | 0.3–0.5 pp |
| Conclusion | Short; restates contributions and future work | 0.3–0.5 pp |
| Back matter (published) | CRediT, funding, competing interests, acknowledgments, appendices, data availability, then references | 0.5–2 pp |

**Reference count.** 34–65; three of four cluster at 55–65. Elsevier author–year (Harvard) style in the published layout.

**Observations for a detector-independent grouping paper evaluated with PGA rather than COCO AP.**

1. **A task-specific metric is normal in CVIU, provided it is defined formally.** None of the four papers reports COCO AP/OKS or PCK; each defines its own metric in a named subsection (Simoni "4.2 Metrics" with an equation; Tanaka "4.1.5 Evaluation Metric" with a matched-accuracy protocol) and, where the metric is a threshold accuracy, says so. PGA should get the same treatment: its own subsection, an equation, the relation to AP (AP conflates detection quality and grouping), and one sentence on why AP is not reported. Simoni's use of "mAP" for a distance-threshold accuracy shows the reader will tolerate a bespoke metric when it is conventional for the benchmark and stated up front.
2. **Controlled comparison against explicitly chosen references reads as the house style, not a weakness.** Yin compares representations "under the same backbone" with † re-implementations and never uses the phrase "state-of-the-art"; Tanaka picks one baseline and one strong model by name; Clinckemaillie compares named model families per component. Only Simoni claims SOTA. A grouping module evaluated on fixed detected keypoints against a named joint-trained reference (HigherHRNet's associative embedding on identical matched detections, plus each swapped detector's native grouping) fits this pattern; frame it as a controlled comparison, name the references, and reserve any SOTA language for the end-to-end result if it is warranted.
3. **Negative results and a dedicated Limitations section are accepted, even in the abstract.** Tanaka's headline finding is partly negative and appears in the abstract and contribution bullets; Clinckemaillie reports that users preferred the baseline. A "Limitations" section placed between the results and the conclusion (Yin §5, Clinckemaillie §7) is the natural home for the in-scope negative results (learned grouping heads, SA-DMoN, hyperbolic embedding, TriGAT, visual-feature augmentation) and for the detector-dependence caveats.
4. **Datasets/protocol before metric before results; ablation position is free.** All four open the experiments with datasets and split protocols, then the metric, then implementation. Ablations may precede the main comparison (Yin) when the design space is what is being demonstrated — a useful precedent for a paper whose story is the grouping-module design (read-out choice, K handling) rather than a single headline number.
5. **Overview diagram early, and one per pipeline stage if the method has stages.** Every paper places a schematic before or at the start of the method text; Yin adds a per-branch block diagram at the head of each method subsection. For PG-GAT: one end-to-end diagram (detector → keypoint graph → GAT → k-means read-out) before the method, and a second, denser diagram for the attention block where the equations start.
6. **Published back matter to plan for.** CRediT roles, funding, declaration of competing interests, a data-availability line and appendices before the references (Clinckemaillie). Highlights and graphical abstract are not visible in any PDF here; check the CVIU Guide for Authors rather than inferring from these exemplars.
