# Reference verification report

Assembled 2026-07-15. Every downloaded PDF was opened and its printed title/authors compared to the BibTeX entry; each entry's venue/year/authors were audited against the record.

## Bottom line

- **87 PDFs checked** — **0 wrong-paper mismatches.** Every downloaded PDF is the correct paper for its citation key.
- **2 bib entries need a real fix** (below), **9 have minor/optional notes**, 80 are clean.
- 4 entries have no PDF (3 software, 1 paywalled) — bib fields for those were still audited and are correct.

## A. Confirmed bib errors — recommend fixing

Both confirmed against the paper's own title page.

### `ranftl2020midas`
- Last author 'Koltun, Vladislav' misspelled; should be 'Koltun, Vladlen' (confirmed by PDF).

```diff
- author={Ranftl, Ren{\'e} and Lasinger, Katrin and Hafner, David and Schindler, Konrad and Koltun, Vladislav},
+ author={Ranftl, Ren{\'e} and Lasinger, Katrin and Hafner, David and Schindler, Konrad and Koltun, Vladlen},
```

### `sun2018integral`
- Third author 'Liu, Fangyin' has wrong surname; should be 'Wei, Fangyin'.
- Missing 4th author 'Liang, Shuang' (corresponding author on PDF).

```diff
- author={Sun, Xiao and Xiao, Bin and Liu, Fangyin and Wei, Yichen},
+ author={Sun, Xiao and Xiao, Bin and Wei, Fangyin and Liang, Shuang and Wei, Yichen},
```

## B. Minor / judgment-call notes (no action strictly required)

| Key | Note |
|---|---|
| `chen2022hpesurvey` | Citation key says 2022 but field year is 2020 (CVIU, online March 2020) - key label mismatch only, minor. |
| `doosti2020hopenet` | Page range 6608-6617 differs by one from IEEE/CVF (6607-6616); databases disagree; core citation correct |
| `liu2023grouppose` | Author list truncated with 'and others' (omits Yao Zhao, Jingdong Wang) — minor |
| `ning2020lighttrack` | Downloaded PDF is arXiv v1 preprint (2 authors Ning, Huang); cited CVPRW 2020 version adds Jian Pei, matching bib (verified on CVF). |
| `xie2016dec` | No pages field (ICML 2016 / PMLR v48 is pp.478-487) — minor incompleteness, not an error |
| `xu2023vitposepp` | Year arguably 2024: cited issue TPAMI vol.46 no.2 is the Feb-2024 issue; 2023 is the early-access year. Vol 46, no 2, pp 1212-1230 otherwise correct. |
| `yang2023edpose` | Middle-author order differs: PDF lists Shilong Liu before Feng Li; bib lists Feng Li before Shilong Liu (same author set) |
| `yuan2021hrformer` | Official NeurIPS listing titles it 'High-Resolution Vision Transformer...'; bib/PDF/arXiv omit 'Vision' (minor) |
| `zheng2023hpesurvey` | ACM Computing Surveys vol 56 issue 1 officially dated Jan 2024; year 2023 is online/accepted year - minor. |

## C. Full check — all references

`title` / `author` = PDF-vs-bib match (exact / minor / no_pdf). `bib` = entry audit.

| # | Key | Title | Author | Bib | Detail |
|---|---|---|---|---|---|
| 1 | `andriluka2014mpii` | exact | yes | ok | CVPR 2014, pages 3686-3693, all four authors correct. |
| 2 | `andriluka2018posetrack` | exact | yes | ok | Title/authors match; CVPR 2018, pages 5167-5176 correct (arXiv 1710.10000). |
| 3 | `arthur2007kmeanspp` | exact | yes | ok | SODA 2007, Arthur & Vassilvitskii, pp 1027-1035 confirmed. |
| 4 | `ba2016layernorm` | exact | yes | ok | arXiv:1607.06450 by Ba, Kiros, Hinton; matches; arXiv-only paper, citation accurate. |
| 5 | `blender` | — | — | ok | Software/misc citation (Blender Foundation, 2024); no PDF expected; well-formed. |
| 6 | `braso2021centergroup` | exact | yes | ok | Brasó, Kister, Leal-Taixé match; ICCV 2021, pages 11853-11863 correct. |
| 7 | `brody2022gatv2` | exact | yes | ok | arXiv 2105.14491 'Published at ICLR 2022'; all 3 authors; ICLR 2022 correct. |
| 8 | `cao2017openpose` | exact | yes | ok | OpenPose; arXiv 1611.08050 matches title + 4 authors; CVPR 2017 correct. |
| 9 | `chen2020simclr` | exact | yes | ok | ICML 2020 SimCLR (4 authors); pages 1597-1607 correct for PMLR 119. |
| 10 | `chen2022hpesurvey` | exact | yes | ok | CVIU vol 192, 102897, year 2020 all correct. |
| 11 | `cheng2020higherhrnet` | exact | yes | ok | Title/authors (Cheng, Xiao, Wang, Shi, Huang, Zhang) match; CVPR 2020 correct. |
| 12 | `cho2014gru` | exact | yes | ok | arXiv 1406.1078; EMNLP 2014, all 7 authors, pp 1724-1734 correct. |
| 13 | `chopra2005contrastive` | exact | yes | ok | Chopra, Hadsell, LeCun; CVPR 2005 correct. |
| 14 | `ci2019optimizing` | exact | yes | ok | Ci, Wang, Ma, Wang match; ICCV 2019, pages 2262-2271 correct. |
| 15 | `deng2009imagenet` | exact | yes | ok | All 6 authors match; CVPR 2009, pp.248-255 correct. |
| 16 | `doosti2020hopenet` | exact | yes | ok | Doosti et al.; CVPR 2020 correct; 1-page offset in pages only. |
| 17 | `fang2017rmpe` | exact | yes | ok | arXiv 1612.00137 matches; ICCV 2017, pages 2334-2343 correct. |
| 18 | `fey2019pyg` | exact | yes | ok | 'Published as a workshop paper at ICLR 2019'; Fey & Lenssen; match. |
| 19 | `geng2021dekr` | exact | yes | ok | DEKR CVPR 2021, all five authors correct. |
| 20 | `geng2023pct` | exact | yes | ok | Title/authors match; CVPR 2023, pages 660-671 correct. |
| 21 | `gilmer2017mpnn` | exact | yes | ok | ICML 2017, all 5 authors match. |
| 22 | `hadsell2006contrastive` | exact | yes | ok | DrLIM; Hadsell, Chopra, LeCun; CVPR 2006, pages 1735-1742 correct. |
| 23 | `hamilton2017graphsage` | exact | yes | ok | GraphSAGE; Hamilton/Ying/Leskovec; NeurIPS 2017 correct. |
| 24 | `he2017maskrcnn` | exact | yes | ok | He/Gkioxari/Dollár/Girshick match; ICCV 2017, pp.2961-2969 correct. |
| 25 | `hendrycks2016gelu` | exact | yes | ok | Hendrycks, Gimpel; arXiv:1606.08415 correct. |
| 26 | `jiang2023rtmpose` | exact | yes | ok | Matches all 8 authors; arXiv:2303.07399 tech report, correct. |
| 27 | `jiang2024rtmw` | exact | yes | ok | arXiv:2407.08634v1 (Jiang, Xie, Li); matches. |
| 28 | `jin2020hgg` | exact | yes | ok | HGG ECCV 2020, all seven authors correct. |
| 29 | `khirodkar2024sapiens` | exact | yes | ok | Title/authors match; ECCV 2024 correct. |
| 30 | `kipf2017gcn` | exact | yes | ok | 'Published at ICLR 2017'; Kipf & Welling confirmed. |
| 31 | `kochurov2020geoopt` | exact | yes | ok | arXiv:2005.02819; matches; arXiv preprint citation standard. |
| 32 | `kreiss2019pifpaf` | exact | yes | ok | Kreiss/Bertoni/Alahi; CVPR 2019, pages 11977-11986 correct. |
| 33 | `kuhn1955hungarian` | exact | yes | ok | H.W. Kuhn; Naval Research Logistics Quarterly vol 2, no 1-2, pp.83-97, 1955 correct. |
| 34 | `li2018hyperband` | exact | yes | ok | All 5 authors; JMLR vol 18, no. 185, pp. 1-52, 2018 confirmed. |
| 35 | `li2019crowdpose` | exact | yes | ok | arXiv 1812.00324 matches all 6 authors; CVPR 2019, pages 10863-10872 correct. |
| 36 | `li2021rle` | exact | yes | ok | ICCV 2021, pages 11025-11034 correct; author set matches. |
| 37 | `li2021tokenpose` | exact | yes | ok | ICCV 2021, pages 11313-11322, all seven authors correct. |
| 38 | `li2022simcc` | exact | yes | ok | Title/authors match; ECCV 2022, pages 89-106 correct. |
| 39 | `lin2014coco` | exact | minor | ok | Same paper; bib matches 8-author ECCV version, arXiv PDF lists 2 extra authors (Bourdev, Girshick) - expected preprint difference. |
| 40 | `liu2020gastnet` | exact | yes | ok | GAST-Net arXiv:2003.14179; author set matches; correct. |
| 41 | `liu2023grouppose` | exact | minor | ok | First author Liu, Huan; ICCV 2023 confirmed; venue/year correct. |
| 42 | `lloyd1982kmeans` | exact | yes | ok | Lloyd; IEEE Trans. Info. Theory vol 28 no 2, pp.129-137, 1982 correct. |
| 43 | `locatello2020slotattention` | exact | yes | ok | All 8 authors; NeurIPS 2020 correct. |
| 44 | `loshchilov2017cosineannealing` | exact | yes | ok | PDF states 'Published at ICLR 2017'; authors match; correct. |
| 45 | `loshchilov2019adamw` | exact | yes | ok | 'Published at ICLR 2019'; Loshchilov & Hutter; matches. |
| 46 | `lu2024rtmo` | exact | yes | ok | RTMO CVPR 2024, pages 1491-1500, all six authors correct. |
| 47 | `macqueen1967kmeans` | exact | yes | ok | Title/author (J. MacQueen) match; Fifth Berkeley Symposium vol 1, pages 281-297, 1967 correct. |
| 48 | `maher2025gec` | exact | yes | ok | J. Image and Graphics Vol.13 No.2, 2025; authors Maher, Fathalla, Shaheen match. |
| 49 | `newell2016hourglass` | exact | yes | ok | Newell, Yang, Deng; ECCV 2016, pages 483-499 correct. |
| 50 | `newell2017associative` | exact | yes | ok | Newell/Huang/Deng; NeurIPS 2017 correct. |
| 51 | `newman2006modularity` | exact | yes | ok | M.E.J. Newman; PNAS vol 103, no 23, pp.8577-8582, 2006 correct. |
| 52 | `nibali2018dsnt` | exact | yes | ok | Nibali, He, Morgan, Prendergast; arXiv:1801.07372 correct. |
| 53 | `nickel2018lorentz` | exact | yes | ok | arXiv 1806.03417 matches; ICML 2018, pages 3776-3785 correct. |
| 54 | `ning2020lighttrack` | exact | minor | ok | Same paper; CVPRW 2020 and 3-author list correct; preprint predates added author. |
| 55 | `oord2018infonce` | exact | yes | ok | arXiv:1807.03748, 2018 correct (CPC is legitimately arXiv-only). |
| 56 | `papandreou2018personlab` | exact | yes | ok | Title/authors match; ECCV 2018, pages 269-286 correct. |
| 57 | `ranftl2020midas` | exact | minor | **FIX** | Same paper; TPAMI 44(3):1623-1637, 2022 correct, but wrong first name 'Vladislav' for Vladlen Koltun. |
| 58 | `sala2018learningmrep` | exact | yes | ok | Sala, De Sa, Gu, Re; ICML 2018, pages 4460-4469 correct. |
| 59 | `sandler2018mobilenetv2` | exact | yes | ok | 5 authors match; CVPR 2018, pages 4510-4520 correct. |
| 60 | `sarkar2011treeembed` | exact | yes | ok | Rik Sarkar; Graph Drawing (GD) 2011, pp.355-366 correct. |
| 61 | `schroff2015facenet` | exact | yes | ok | Schroff, Kalenichenko, Philbin; CVPR 2015 pp. 815-823 confirmed. |
| 62 | `scikitlearn` | exact | yes | ok | PDF header confirms JMLR 12 (2011) 2825-2830; authors match. |
| 63 | `shi2000ncut` | exact | yes | ok | Rendered page 1 confirms title + Shi & Malik; TPAMI 22(8):888-905, 2000 correct (PDF is author tech-report copy). |
| 64 | `shi2019twostreamadaptive` | exact | yes | ok | 2s-AGCN CVPR 2019, pages 12026-12035, all four authors correct. |
| 65 | `shi2022petr` | exact | yes | ok | Title/authors match; CVPR 2022, pages 11069-11078 correct. |
| 66 | `sun2018integral` | exact | minor | **FIX** | Same paper (ECCV 2018, pp 529-545); true author list Sun, Xiao, Wei (Fangyin), Liang, Wei (Yichen). |
| 67 | `sun2019hrnet` | exact | yes | ok | HRNet; Sun, Xiao, Liu, Wang; CVPR 2019 correct. |
| 68 | `toshev2014deeppose` | exact | yes | ok | Toshev/Szegedy; CVPR 2014, pages 1653-1660 correct. |
| 69 | `tsitsulin2023dmon` | exact | yes | ok | Authors match; JMLR 24 (2023) vol 24 no 127 pp.1-21 correct. |
| 70 | `ultralytics2024yolo11` | — | — | ok | Software @misc; YOLO11 2024 Ultralytics; Jocher and Qiu standard authors; fields correct. |
| 71 | `vandermaaten2008tsne` | exact | yes | ok | PDF header confirms JMLR 9 (2008) 2579-2605; authors match. |
| 72 | `velickovic2018gat` | exact | yes | ok | 'Published at ICLR 2018'; all six authors match. |
| 73 | `vonluxburg2007spectral` | exact | yes | ok | Statistics and Computing vol 17 no 4, pp 395-416, 2007 correct. |
| 74 | `wandb` | — | — | ok | Software @misc entry (Biewald, W&B, 2020); no PDF expected; correct. |
| 75 | `wei2016cpm` | exact | yes | ok | CVPR 2016, pp 4724-4732, authors Wei/Ramakrishna/Kanade/Sheikh confirmed. |
| 76 | `xiao2018simplebaselines` | exact | yes | ok | Xiao, Wu, Wei; ECCV 2018, pages 466-481 correct. |
| 77 | `xiao2022querypose` | exact | yes | ok | Same 7 authors (slight order diff only); NeurIPS 2022 correct. |
| 78 | `xie2016dec` | exact | yes | ok | Xie/Girshick/Farhadi; ICML 2016 correct; optional pages absent. |
| 79 | `xiu2018poseflow` | exact | yes | ok | All 5 authors; BMVC 2018 correct. |
| 80 | `xu2022vitpose` | exact | yes | ok | arXiv 2204.12484 matches 4 authors; NeurIPS 2022 vol 35, pages 38571-38584 correct. |
| 81 | `xu2023vitposepp` | exact | yes | **FIX** | arXiv:2212.04246v3; authors match; only year 2023-vs-2024 issue-date question. |
| 82 | `yan2018stgcn` | exact | yes | ok | ST-GCN AAAI 2018, pages 7444-7452, all three authors correct. |
| 83 | `yang2019face` | exact | yes | ok | Title/authors (Yang, Zhan, Chen, Yan, Loy, Lin) match; CVPR 2019 correct. |
| 84 | `yang2020gcnve` | exact | yes | ok | CVPR 2020, all 6 authors confirmed. |
| 85 | `yang2023edpose` | exact | minor | ok | ED-Pose 'Published at ICLR 2023'; first author + set match; minor Li/Liu order swap. |
| 86 | `yang2023instanceaware` | — | — | ok | No PDF; Pattern Recognition vol 136, art 109232 (2023) confirmed via ScienceDirect. |
| 87 | `yang2024xpose` | exact | yes | ok | Yang/Zeng/Zhang/Zhang; ECCV 2024 correct. |
| 88 | `yuan2021hrformer` | exact | yes | ok | All 7 authors; NeurIPS 2021 vol 34 pp. 7281-7293 confirmed. |
| 89 | `zhang2019pose2seg` | exact | yes | ok | arXiv 1803.10683 matches all 9 authors; CVPR 2019, pages 889-898 correct. |
| 90 | `zhao2019semanticgcn` | exact | yes | ok | Verified directly from PDF page 1: title + all 5 authors (Zhao, Peng, Tian, Kapadia, Metaxas) match; CVPR 2019, pages 3425-3435 correct. |
| 91 | `zheng2023hpesurvey` | exact | yes | ok | ACM CSUR vol 56, no 1, pp 1-37 confirmed; year 2023 defensible. |

## D. No-PDF entries (audited, correct)

- `blender` — Software/misc citation (Blender Foundation, 2024); no PDF expected; well-formed.
- `ultralytics2024yolo11` — Software @misc; YOLO11 2024 Ultralytics; Jocher and Qiu standard authors; fields correct.
- `wandb` — Software @misc entry (Biewald, W&B, 2020); no PDF expected; correct.
- `yang2023instanceaware` — No PDF; Pattern Recognition vol 136, art 109232 (2023) confirmed via ScienceDirect.
