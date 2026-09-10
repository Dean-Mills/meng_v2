# The Visual Computer (Springer) — fallback-venue notes

Compiled 2026-09-09. Every fact below is tied to a page that was actually fetched on that date (URL given). "NOT FOUND" means the item was looked for on the cited page(s) and is not there. Inferences are labelled as such.

Access note: `link.springer.com` serves a JavaScript "Client Challenge" page to non-browser clients (curl, headless Chromium from the snap sandbox), so journal pages were read via WebFetch's cookie-fallback URL and via the `r.jina.ai` reader proxy; the cited URL is always the underlying Springer page.

---

## Part A — Submission guidelines

### A1. LaTeX template

| Item | Finding | Source |
|---|---|---|
| Journal-specific template | None. Verbatim: "At present there is no official template for the journal, just a general latex one, therefore please use the Instructions for Authors for the generic formatting and then submit." | https://link.springer.com/journal/371/submission-guidelines |
| Generic template named | "Springer Nature LaTeX authoring template" — class `sn-jnl.cls` (the `sn-jnl` class, **not** `svjour3`; `svjour3` is not mentioned on any page fetched) | https://www.springernature.com/gp/authors/campaigns/latex-author-support |
| Download zip | https://cms-resources.apps.public.k8s.springernature.io/springer-cms/rest/v1/content/18782940/data/v12 (901,814 bytes, downloaded and unpacked; folder `sn-article-template/`) | same page |
| Overleaf copy | https://www.overleaf.com/latex/templates/springer-nature-latex-template/gsvvftmrppwq | same page |
| Template version | "Version 3.1 December 2024" (line 1 of `sn-article.tex`; user-manual §11) | template zip |
| Zip contents | `sn-jnl.cls`, `sn-article.tex`, `sn-article.pdf` (12 pp sample), `sn-bibliography.bib`, `user-manual.pdf`, `fig.eps`, `empty.eps`, `bst/` with 9 files: `sn-basic`, `sn-mathphys-num`, `sn-mathphys-ay`, `sn-aps`, `sn-vancouver-num`, `sn-vancouver-ay`, `sn-apa`, `sn-apacite`, `sn-chicago`, `sn-nature` | template zip |
| Default documentclass line in `sn-article.tex` | `\documentclass[pdflatex,sn-mathphys-num]{sn-jnl}% Math and Physical Sciences Numbered Reference Style` (line 65) | template zip |
| Other documented options | `[pdflatex,sn-basic]`, `[pdflatex,sn-nature]`, `[pdflatex,sn-mathphys-ay]`, `[pdflatex,sn-aps]`, `[pdflatex,sn-vancouver-num]`, `[pdflatex,sn-vancouver-ay]`, `[pdflatex,sn-apa]`, `[pdflatex,sn-chicago]`; `referee` (double spacing), `lineno`; `Numbered` toggles numbered vs author-year for `sn-basic`/`sn-chicago` (user-manual §7.9: `\documentclass[sn-basic,Numbered]{sn-jnl}`) | template zip |
| pdflatex note | "If you are submitting to eJP using the Springer Nature LaTeX template you will need to use the pdflatex option in the preamble to allow PDF compilation." | https://www.springernature.com/gp/authors/campaigns/latex-author-support |
| Which .bst for which journal | Template does not say. User-manual: "For submissions to all other journals please check the submission guidelines on the respective journal website for information on the particular bibliography style that is being used for that journal." | template zip, `user-manual.pdf` |

**Local TeX Live / CTAN**

| Check | Result |
|---|---|
| `kpsewhich sn-jnl.cls` | not found (exit 1) |
| `tlmgr info sn-jnl` | no package (Debian TeX Live, tlmgr in user mode prints nothing) |
| `kpsewhich svjour3.cls` | not found |
| https://ctan.org/pkg/sn-jnl | HTTP 404 |
| https://ctan.org/search?phrase=sn-jnl | "The search found no matching documents on CTAN" |
| Conclusion | `sn-jnl` is **not on CTAN**; obtain from the Springer zip URL above (or Overleaf). Nothing was installed. |

### A2. Reference style

| Item | Finding | Source |
|---|---|---|
| Citation style | Numbered. "Reference citations in the text should be identified by numbers in square brackets." Examples: "Negotiation research spans many disciplines [3]." / "This effect has been widely studied [1-3, 7]." | https://link.springer.com/journal/371/submission-guidelines |
| List order | "The entries in the list should be numbered consecutively." | same |
| DOIs | "If available, please always include DOIs as full DOI links in your reference list (e.g. 'https://doi.org/abc')." | same |
| Journal abbreviations | "Always use the standard abbreviation of a journal's name according to the ISSN List of Title Word Abbreviations" | same |
| Example entry (journal) | "Hamburger, C.: Quasimonotonicity, regularity and duality for nonlinear systems of partial differential equations. Ann. Mat. Pura Appl. 169, 321–354 (1995)" | same |
| Example entry (chapter) | "Broy, M.: Software engineering — from auxiliary to key technologies. In: Broy, M., Denert, E. (eds.) Software Pioneers, pp. 10–13. Springer, Heidelberg (2002)" | same |
| Named .bst | NOT FOUND on the journal page. | — |
| Inference | The "Surname, I.: Title. Journal vol, pages (year)" pattern is the template's "Math and Physical Sciences" numbered style, i.e. `sn-mathphys-num` — which is also the template default. Both downloaded exemplars use exactly this format (e.g. "He, K., Gkioxari, G., Dollár, P., Girshick, R.: Mask r-cnn. In: ..."). | template `bst/`, exemplar PDFs |

### A3. Length norms

| Item | Finding | Source |
|---|---|---|
| Page / word limit | NOT FOUND (no limit stated). | https://link.springer.com/journal/371/submission-guidelines |
| Observed lengths (typeset) | Tian 2025: 15 pp; Peng (EHFusion) 2025: 23 pp; Peng (scoliosis GCN) 2025: 13 pp; template sample: 12 pp | exemplar PDFs (`pdfinfo`) |

### A4. Abstract, keywords, highlights, graphical abstract

| Item | Finding | Source |
|---|---|---|
| Abstract | "Please provide an abstract of 150 to 250 words." | https://link.springer.com/journal/371/submission-guidelines |
| Keywords | "Please provide 4 to 6 keywords which can be used for indexing purposes." | same |
| Highlights | NOT FOUND | same |
| Graphical abstract | NOT FOUND | same |
| Journal-specific extra | "Authors have to supply a brief biographical summary of between 50 and 100 words and a black and white photograph, passport sized." | same |

### A5. Peer review

| Item | Finding | Source |
|---|---|---|
| Journal | "This journal follows a single-blind reviewing procedure." | https://link.springer.com/journal/371/submission-guidelines |
| Publisher default | "Most journals use a single-anonymized peer review process; that is, author identities are known to peer reviewers, but peer reviewers identities are not revealed to the authors." | https://link.springer.com/brands/springer/journal-policies |

### A6. Declarations

| Item | Finding | Source |
|---|---|---|
| Heading / placement | "The following statements should be included under the heading 'Statements and Declarations' for inclusion in the published paper." | https://link.springer.com/journal/371/submission-guidelines |
| Required | Competing Interests (required); Funding (required, also entered in submission system); Data availability ("All original research must include a data availability statement"); Ethics approval / consent "if applicable"; Author contributions (recommended, separate section) | same |
| Template | `\section*{Declarations}` with bullet list: Funding; Conflict of interest/Competing interests; Ethics approval and consent to participate; Data availability; Author contribution (lines 1101–1125 of `sn-article.tex`) | template zip |
| As typeset in exemplars | After Conclusions, before References, in the order: Acknowledgements → Author Contributions → Funding → Data Availability → **Declarations** (Conflict of interest) → Open Access licence block → References | exemplar PDFs |

### A7. Preprint and generative-AI policy

| Item | Finding | Source |
|---|---|---|
| Preprint (journal page) | NOT FOUND on the journal's own guidelines page. | https://link.springer.com/journal/371/submission-guidelines |
| Preprint (publisher policy) | "Posting of preprints is not considered prior publication and will not jeopardize consideration at Springer Nature journals." / "Authors should disclose details of preprint posting, including DOI and licensing terms, upon submission of the manuscript." / "Once the preprint is published, it is the author's responsibility to ensure that the preprint record is updated with a publication reference." CC-licence restriction on the preprint: NOT FOUND on this page. | https://link.springer.com/brands/springer/journal-policies |
| Generative AI | "Large Language Models (LLMs), such as ChatGPT, do not currently satisfy our authorship criteria. Notably an attribution of authorship carries with it accountability for the work, which cannot be effectively applied to LLMs." / "Use of an LLM should be properly documented in the Methods section." / "AI assisted copy editing" (readability improvements only) need not be declared. | https://link.springer.com/journal/371/submission-guidelines |

### A8. Open access

| Item | Finding | Source |
|---|---|---|
| APC | "The current APC for The Visual Computer is £2390.00 GBP / $3290.00 USD / €2690.00 EUR." | https://link.springer.com/journal/371/how-to-publish-with-us |
| Model | "The Visual Computer is a hybrid journal. Once the article is accepted for publication, authors will have the option to choose how their article is published" — subscription (no fee) or open access. Journal home: "Publishing model: Hybrid". | same; https://link.springer.com/journal/371 |
| Journal metrics on home page | Impact Factor 3.4 (2025); 5-year 3.2; downloads 745k (2025); "Submission to first decision (median): 3 days" as displayed; Editor-in-Chief Prof. Nadia Magnenat-Thalmann | https://link.springer.com/journal/371 |
| Transformative agreements | NOT FOUND on the pages fetched | — |

### A9. Figures and tables

| Item | Finding | Source |
|---|---|---|
| Resolution | Line art min 1200 dpi; halftone min 300 dpi; combination art min 600 dpi | https://link.springer.com/journal/371/submission-guidelines |
| Fonts | Helvetica or Arial (sans serif); "Keep lettering consistently sized throughout your final-sized artwork, usually about 2–3 mm (8–12 pt)." | same |
| Colour | "Color art is free of charge for online publication." Ensure figures remain distinguishable in B&W print. | same |
| Formats | EPS (vector), TIFF (halftone); MS Office files acceptable | same |
| Figure captions | "Each figure should have a concise caption describing accurately what the figure depicts." Captions go in the text file, not in the figure file. | same |
| Table captions | "For each table, please supply a table caption (title) explaining the components of the table." "Footnotes to tables should be indicated by superscript lower-case letters (or asterisks for significance values and other statistical data)." | same |
| Placement (template) | `\caption` is placed **before** `\begin{tabular}` (table caption above) and **after** `\includegraphics` (figure caption below) — `sn-article.tex` lines 399–419 and 613–619 | template zip |
| Placement (as typeset) | Same in both exemplar PDFs: table titles above, figure captions below. | exemplar PDFs |

### A10. Aims and scope (verbatim, relevant sentences)

Source: https://link.springer.com/journal/371/aims-and-scope

- "The Visual Computer publishes high-quality research on all aspects of computer graphics and visual computing, with a special focus on the rapidly evolving integration of artificial intelligence, computer vision, and graphics."
- "Modern visual computing stands at the convergence of graphics (generative, simulation, rendering), vision (analysis, perception, scene understanding) and AI (representation learning, generative models, multimodal models)."
- "The Visual Computer is one of the few journals that explicitly embraces this intersection and publishes research that combines all three."
- Listed area: "Computer Vision for Graphics, including detection, segmentation, **pose estimation**, and scene understanding"
- Also listed: "Animation, Simulation, and Digital Humans"; "Multimodal AI for Graphics".

No sentence mentions "human pose" specifically; "pose estimation" appears once, in the list above.

---

## Part B — Exemplars (2024–2026)

Search coverage: four Crossref queries filtered to `container-title:The Visual Computer`, 2024-01-01 to 2026-12-31 (`api.crossref.org/works?...query.bibliographic=` with "multi-person pose estimation keypoint", "graph convolutional network human pose skeleton", "bottom-up pose estimation keypoint grouping association", "graph attention network human pose joints", "multi-person"), each returning 20–25 records, cross-checked with Semantic Scholar and OpenAlex. **No 2024–2026 paper in this journal on keypoint grouping / association was found.** Closest fits: (1) Tian 2025 — 2D multi-person pose (open access but PDF not retrievable from this machine, see below); (2) Peng 2025 EHFusion — topology-based *grouping of joints* for 3D pose, CC BY; (3) Peng 2025 — GCN on 3D skeleton sequences, CC BY. dblp could not be consulted (https://dblp.org returns an Anubis bot-check page). arXiv title search for the Tian paper returned 0 results (`export.arxiv.org/api/query`).

Downloaded files (both verified with `file` and `pdfinfo` as PDF 1.4, Creator "Springer"):

| File | Bytes | Pages |
|---|---|---|
| `tvc_peng2025_ehfusion.pdf` | 2,366,798 | 23 |
| `tvc_peng2025_scoliosisgcn.pdf` | 1,627,767 | 13 |

### B1. Peng, Zhou, Mok — EHFusion (downloaded)

| Field | Value | Source |
|---|---|---|
| Title | EHFusion: an efficient heterogeneous fusion model for group-based 3D human pose estimation | Crossref https://api.crossref.org/works/10.1007/s00371-024-03724-5 |
| Authors | Jihua Peng, Yanghong Zhou, P. Y. Mok | same |
| Year / volume / pages | Vis. Comput. 41(8), 5323–5345, print June 2025; online 27 Nov 2024 (Crossref `issued` year 2024; file named 2025 after the volume year) | same |
| DOI | 10.1007/s00371-024-03724-5 — resolves (`https://doi.org/...` → 302 → `https://link.springer.com/10.1007/s00371-024-03724-5`) | curl -I |
| Open access | Yes, CC BY 4.0 (Crossref license VOR; OpenAlex `oa_status: hybrid`, license cc-by) | Crossref; https://api.openalex.org/works/https://doi.org/10.1007/s00371-024-03724-5 |
| PDF source used | PolyU Institutional Research Archive (published version, CC BY): https://ira.lib.polyu.edu.hk/bitstream/10397/112581/1/s00371-024-03724-5.pdf (record http://hdl.handle.net/10397/112581). Springer's own PDF link is blocked to curl by the JS challenge. | https://ira.lib.polyu.edu.hk/handle/10397/112581 |
| Relevance | 2D→3D lifting with joints partitioned into topology-based groups and fused (HFF module); not multi-person, not a GNN, but the closest open-access "grouping" paper in the journal. | PDF |

Structure notes (from `pdftotext`):

- **Pages**: 23 typeset (two-column Springer layout).
- **References**: 66 (Crossref `reference-count` 66; last numbered entry 66).
- **Abstract**: 252 words (own count), 4 keywords.
- **Section order**: 1 Introduction (pp. 1–3) → 2 Related work (p. 3; 2.1–2.4) → 3 Method (pp. 4–9: 3.1 Problem formulation, 3.2 Heterogeneous feature fusion, 3.3 Motion amplitude information, 3.4 Camera intrinsic embedding, 3.5 Model optimization) → 4 Experimental results and discussion (pp. 9–20: 4.1 Datasets and evaluation protocol, 4.2 Ablation studies, 4.3 Comparison with state-of-the-art methods, 4.4 Discussion, 4.5 Qualitative results) → 5 Conclusions (p. 20) → Acknowledgements, Funding, Data availability, Declarations, Open Access → References (pp. 20–23).
- **Overview diagram before method text**: Yes. Fig. 1 "Architecture of the proposed EHFusion model" is on p. 3 inside the Introduction, before Sec. 2 and Sec. 3; Fig. 2 (full end-to-end network) opens the Method on p. 5.
- **Results / ablation layout**: Ablations come **first** (4.2, p. 10): Table 1 rows Baseline / +MAI / +… with columns MPJPE (mm) ↓, FLOPs ↓, Parameters ↓, Training time (min/epoch) ↓. Then SOTA comparisons (4.3): Tables 2, 4, 5 are per-action Human3.6M tables (Protocol #1 MPJPE with CPN-detected 2D input; Protocol #2 P-MPJPE; Protocol #1 with GT 2D input), Table 6 HumanEva-I. A separate 4.4 Discussion holds hyperparameter/design ablations (Tables 7–11: MAI channel sizes, whether to encode MAI separately, feature-fusion settings, CIE hyperparameters) plus Figs. 6–8 comparison plots. Table 12 is a final comparison. Qualitative Fig. 9 last.
- **Metrics**: MPJPE and P-MPJPE (mm), FLOPs, parameter count, training time. No COCO AP.
- **Limitations / negatives**: Explicit paragraph headed "Limitations and future research" inside the Conclusions: predicted poses are relative only (no pixel coordinates / action categories), single input modality. Ablation tables also show configurations that do not help, discussed in 4.4.

### B2. Peng, Wang, Sun, Lv, Wang, Li, An — GCN scoliosis screening (downloaded)

| Field | Value | Source |
|---|---|---|
| Title | Graph convolutional networks for 3D skeleton-based scoliosis screening using gait sequences | Crossref https://api.crossref.org/works/10.1007/s00371-025-03983-w |
| Authors | Zizhao Peng, Zihan Wang, Mengying Sun, Zheng Lv, Yan Wang, Ping Li, Fengwei An | same |
| Year / volume / pages | Vis. Comput. 41(9), 6823–6835, print July 2025; online 27 May 2025 | same |
| DOI | 10.1007/s00371-025-03983-w — resolves (302 → `https://link.springer.com/10.1007/s00371-025-03983-w`) | curl -I |
| Open access | Yes, CC BY 4.0 (Crossref; OpenAlex `oa_status: hybrid`, cc-by) | Crossref; https://api.openalex.org/works/https://doi.org/10.1007/s00371-025-03983-w |
| PDF source used | PolyU IRA (published version, CC BY): https://ira.lib.polyu.edu.hk/bitstream/10397/115570/1/s00371-025-03983-w.pdf (record http://hdl.handle.net/10397/115570) | https://ira.lib.polyu.edu.hk/handle/10397/115570 |
| Relevance | GCN over 3D skeleton sequences (skeleton-as-graph, spatial-temporal); the "GNN on human skeletons" category. Application paper, not pose estimation. | PDF |

Structure notes:

- **Pages**: 13. **References**: 27 (Crossref 27). **Abstract**: 217 words; 4 keywords.
- **Section order**: 1 Introduction → 2 Related works → 3 Dataset (3.1 Overview, 3.2 Setup, 3.3 Preprocess) → 4 Methodology (pp. 5–8, incl. 4.3 Scoliosis recognition) → 5 Experiment (5.1 Setup, 5.2 Result; pp. 8–10) → 6 Discussion (6.1 Grouping strategy, 6.2 Overfitting; p. 10) → 7 Conclusion (p. 10) → Acknowledgements, Author Contributions, Funding, Data Availability, Declarations, Open Access → References (pp. 11–13).
- **Overview diagram before method text**: Partly. Fig. 1 is the data-collection setup (p. 5); the architecture figure is Fig. 3 "Overall architecture" on p. 6, *inside* Sec. 4, not before it.
- **Results / ablation layout**: Table 3 "Comparison results" against prior screening studies with columns Screening test / Accuracy / Sensitivity / Specificity (with confidence intervals); Fig. 5 confusion matrix; Fig. 6 feature visualisation; Table 4 "Grouping strategy experiment result" (rows = joint groups, columns Accuracy % / Sensitivity %) is the ablation, placed in the Discussion with Fig. 8 defining the groups.
- **Metrics**: accuracy, sensitivity, specificity. No COCO AP, no MPJPE.
- **Limitations / negatives**: A dedicated "6.2 Overfitting" subsection: "the model typically starts to overfit before reaching 200 epochs" and "Despite these efforts, further refinement is still needed to enhance the model's ability to generalize to new data scenarios." Also reports that adding arm joints has a "significant negative impact on the model's sensitivity".

### B3. Tian et al. — 2D multi-person pose refinement (metadata verified; PDF NOT downloaded)

| Field | Value | Source |
|---|---|---|
| Title | Local feature enhancement for robust 2D multi-person pose estimation via pose refinement network | Crossref https://api.crossref.org/works/10.1007/s00371-025-04256-2 |
| Authors | Weili Tian, Jin Zhan, Zhaokang Guan, Chensheng Yi, Fangyuan Lei, Xiaoyong Liu, Huihui Li, Yufeng Zeng | same |
| Year / volume | Vis. Comput. 42(1), article 24, Jan 2026; online 7 Dec 2025 | same |
| DOI | 10.1007/s00371-025-04256-2 — resolves (302 → link.springer.com) | curl -I |
| Open access | Yes, CC BY-NC-ND 4.0 (Crossref VOR license; OpenAlex hybrid) | Crossref; OpenAlex |
| Legitimate PDF | https://link.springer.com/content/pdf/10.1007/s00371-025-04256-2.pdf (publisher OA). **Not downloadable from this machine**: curl (any headers) gets the 3 KB "Client Challenge" page; snap Chromium headless produced empty output even for example.com; OpenAlex lists no repository copy; arXiv title search returned 0 results. Full text was read through the `r.jina.ai` reader proxy for the notes below. | curl, OpenAlex, arXiv API |

Structure notes (from proxy-read full text; not re-checked against a local PDF):

- 15 typeset pages; 47 references (Crossref 47); abstract ≈165 words.
- Sections: 1 Introduction → 2 Related Work → 3 The Proposed Method → 4 Experimental Results and Comparison → 5 Conclusions and Future Work.
- Fig. 1 is the architecture overview (HRNet backbone, DCM, HSM, KRN) at the start of Sec. 3.
- Datasets: COCO val2017 / test-dev2017 and CrowdPose. Metrics: AP, AP50, AP75, APM, APL (COCO); AP, AP50, AP75, APE, APM, APH (CrowdPose). Headline: 73.1 AP COCO val (640×640, multi-scale), 71.8 test-dev, 68.7 CrowdPose; compared against HigherHRNet at 640×640 input.
- Ablations: Table 5 (three DCM variants, AP on both datasets), Table 6 (DCM / HSM toggled independently, both datasets), Fig. 7 (attention variants SE / SAM / CBAM / HSM on CrowdPose).
- Limitations: no section; one sentence concedes lower AP50 than two competitors on CrowdPose. Declarations: Funding list; "Experimental data are provided within the manuscript. The code and models ... are available at https://github.com/Twl-GZ/Human-pose."; "The authors declare no competing interests."

---

## Not verified / gaps

- Page or word limit: none stated on the guidelines page.
- Highlights and graphical abstract: not mentioned anywhere fetched.
- `svjour3`: not mentioned on any fetched page; the current guidance is the generic `sn-jnl` template.
- Which `.bst` the journal wants is not named; `sn-mathphys-num` is an inference from the journal's reference examples and both exemplars' typeset reference format.
- CC-licence restriction on preprints: not on the Springer journal-policies page fetched.
- Transformative-agreement coverage: not on the pages fetched.
- Tian 2025 PDF binary not obtained (see B3).
- dblp listing not consulted (bot-check page).
