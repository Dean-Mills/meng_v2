# Journal template bundles (downloaded 2026-09-09)

- `cviu/` - Elsevier's CVIU LaTeX template, unpacked from `YCVIU_latex_template-updated.zip`
  (source: https://www.elsevier.com/__data/promis_misc/YCVIU%20latex%20template-updated.zip, the "latex template" link in the CVIU Guide for Authors).
  Contents: `ycviu.sty` (journal layout: two-column, 522 pt text width), `ycviu-template-with-authorship.tex`
  (`\documentclass[times,twocolumn,final,authoryear]{elsarticle}` + `\usepackage{ycviu}`, `\bibliographystyle{model2-names}`),
  the `-referees` variant (1.5 line spacing), `model2-names.bst` (author-year), the authorship-confirmation form (PDF), a sample figure,
  and the bundle's own `elsarticle.cls` (v2.1, 2013 - older than the installed TeX Live class; decide which to compile with).
- `springer/sn-article-template.zip` - Springer Nature generic `sn-jnl` template v3.1 (Dec 2024), for The Visual Computer fallback only
  (source: https://www.springernature.com/gp/authors/campaigns/latex-author-support). Not unpacked; not on CTAN; nothing installed.

The verbatim author-guide facts with URLs are in `../cviu_author_guide_facts.md`; the exemplar analyses are in `../exemplars/`.
