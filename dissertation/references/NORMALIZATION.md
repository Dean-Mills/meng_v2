# Bibliography normalization — proposed changes

Assembled 2026-07-15 from Crossref + DBLP + the downloaded PDFs. **Nothing has been written to `diss_thesis.bib`.** A candidate file `diss_thesis_normalized.bib` was produced with the *applied* changes below; review it with:

```
git diff --no-index dissertation/diss_thesis.bib dissertation/references/diss_thesis_normalized.bib
```

**Applied automatically (safe):** 55 DOIs added, 4 author fixes, 9 missing page ranges filled.  
**Left for your decision (below):** page-numbering conflicts, optional DOIs, and a few judgment calls — none applied.


## A. Author fixes applied

| Key | Was | Now |
|---|---|---|
| `liu2023grouppose` | Liu, Huan and Chen, Qiang and Tan, Zichang and Liu, Jiangjiang and Wang, Jian and Su, Xiangbo and Li, Xiaolong and Yao, Kun and Han, Junyu and Ding, Errui and others | Liu, Huan and Chen, Qiang and Tan, Zichang and Liu, Jiang-Jiang and Wang, Jian and Su, Xiangbo and Li, Xiaolong and Yao, Kun and Han, Junyu and Ding, Errui and Zhao, Yao and Wang, Jingdong |
| `ranftl2020midas` | Koltun, Vladislav | Ranftl, Ren{\'e} and Lasinger, Katrin and Hafner, David and Schindler, Konrad and Koltun, Vladlen |
| `sun2018integral` | Sun, Xiao and Xiao, Bin and Liu, Fangyin and Wei, Yichen | Sun, Xiao and Xiao, Bin and Wei, Fangyin and Liang, Shuang and Wei, Yichen |
| `yang2023edpose` | Yang, Jie and Zeng, Ailing and Li, Feng and Liu, Shilong and Zhang, Ruimao and Zhang, Lei | Yang, Jie and Zeng, Ailing and Liu, Shilong and Li, Feng and Zhang, Ruimao and Zhang, Lei |

## B. DOIs added

55 publisher DOIs (IEEE casing normalised to uppercase). Source: c=Crossref, d=DBLP.

| Key | DOI | src |
|---|---|---|
| `andriluka2014mpii` | 10.1109/CVPR.2014.471 | c |
| `andriluka2018posetrack` | 10.1109/CVPR.2018.00542 | d |
| `braso2021centergroup` | 10.1109/ICCV48922.2021.01164 | c |
| `cao2017openpose` | 10.1109/CVPR.2017.143 | d |
| `chen2022hpesurvey` | 10.1016/j.cviu.2019.102897 | c |
| `cheng2020higherhrnet` | 10.1109/CVPR42600.2020.00543 | c |
| `cho2014gru` | 10.3115/v1/D14-1179 | d |
| `chopra2005contrastive` | 10.1109/CVPR.2005.202 | d |
| `ci2019optimizing` | 10.1109/ICCV.2019.00235 | c |
| `deng2009imagenet` | 10.1109/CVPR.2009.5206848 | c |
| `doosti2020hopenet` | 10.1109/CVPR42600.2020.00664 | c |
| `fang2017rmpe` | 10.1109/ICCV.2017.256 | d |
| `geng2021dekr` | 10.1109/CVPR46437.2021.01444 | d |
| `geng2023pct` | 10.1109/CVPR52729.2023.00071 | c |
| `hadsell2006contrastive` | 10.1109/CVPR.2006.100 | c |
| `he2017maskrcnn` | 10.1109/ICCV.2017.322 | c |
| `jin2020hgg` | 10.1007/978-3-030-58571-6_42 | d |
| `khirodkar2024sapiens` | 10.1007/978-3-031-73235-5_12 | d |
| `kreiss2019pifpaf` | 10.1109/CVPR.2019.01225 | d |
| `kuhn1955hungarian` | 10.1002/nav.3800020109 | c |
| `li2019crowdpose` | 10.1109/CVPR.2019.01112 | c |
| `li2021rle` | 10.1109/ICCV48922.2021.01084 | c |
| `li2021tokenpose` | 10.1109/ICCV48922.2021.01112 | d |
| `li2022simcc` | 10.1007/978-3-031-20068-7_6 | d |
| `lin2014coco` | 10.1007/978-3-319-10602-1_48 | d |
| `liu2023grouppose` | 10.1109/ICCV51070.2023.01380 | c |
| `lloyd1982kmeans` | 10.1109/TIT.1982.1056489 | c |
| `lu2024rtmo` | 10.1109/CVPR52733.2024.00148 | c |
| `maher2025gec` | 10.18178/joig.13.2.130-139 | c |
| `newell2016hourglass` | 10.1007/978-3-319-46484-8_29 | c |
| `newman2006modularity` | 10.1073/pnas.0601602103 | c |
| `ning2020lighttrack` | 10.1109/CVPRW50498.2020.00525 | c |
| `papandreou2018personlab` | 10.1007/978-3-030-01264-9_17 | c |
| `ranftl2020midas` | 10.1109/TPAMI.2020.3019967 | c |
| `sandler2018mobilenetv2` | 10.1109/CVPR.2018.00474 | d |
| `sarkar2011treeembed` | 10.1007/978-3-642-25878-7_34 | c |
| `schroff2015facenet` | 10.1109/CVPR.2015.7298682 | c |
| `shi2000ncut` | 10.1109/34.868688 | c |
| `shi2019twostreamadaptive` | 10.1109/CVPR.2019.01230 | c |
| `shi2022petr` | 10.1109/CVPR52688.2022.01079 | c |
| `sun2018integral` | 10.1007/978-3-030-01231-1_33 | c |
| `sun2019hrnet` | 10.1109/CVPR.2019.00584 | d |
| `toshev2014deeppose` | 10.1109/CVPR.2014.214 | d |
| `vonluxburg2007spectral` | 10.1007/s11222-007-9033-z | c |
| `wei2016cpm` | 10.1109/CVPR.2016.511 | c |
| `xiao2018simplebaselines` | 10.1007/978-3-030-01231-1_29 | c |
| `xu2023vitposepp` | 10.1109/TPAMI.2023.3330016 | c |
| `yan2018stgcn` | 10.1609/aaai.v32i1.12328 | c |
| `yang2019face` | 10.1109/CVPR.2019.00240 | d |
| `yang2020gcnve` | 10.1109/CVPR42600.2020.01338 | c |
| `yang2023instanceaware` | 10.1016/j.patcog.2022.109232 | c |
| `yang2024xpose` | 10.1007/978-3-031-72952-2_15 | d |
| `zhang2019pose2seg` | 10.1109/CVPR.2019.00098 | d |
| `zhao2019semanticgcn` | 10.1109/CVPR.2019.00354 | c |
| `zheng2023hpesurvey` | 10.1145/3603618 | c |

## C. Missing page ranges filled

| Key | Pages added |
|---|---|
| `cheng2020higherhrnet` | 5385--5394 |
| `geng2021dekr` | 14671--14681 |
| `khirodkar2024sapiens` | 206--228 |
| `lin2014coco` | 740--755 |
| `liu2023grouppose` | 14983--14992 |
| `sun2019hrnet` | 5693--5703 |
| `xie2016dec` | 478--487 |
| `yang2020gcnve` | 13366--13375 |
| `yang2024xpose` | 249--268 |

## D. DECISIONS NEEDED (nothing applied)

### D1. Existing page numbers that differ from the official record

These entries already have page numbers using the **CVF/CVPR open-access** numbering; IEEE Xplore / Springer assign different numbers for the same papers. The candidate bib keeps your current values. Pick one convention if you want consistency (IEEE style thesis → arguably the IEEE/Springer numbers).

| Key | Current (kept) | Official record |
|---|---|---|
| `braso2021centergroup` | 11853--11863 | IEEE Xplore 11833-11843 |
| `doosti2020hopenet` | 6608--6617 | IEEE Xplore 6607-6616 |
| `fang2017rmpe` | 2334--2343 | IEEE Xplore 2353-2362 |
| `he2017maskrcnn` | 2961--2969 | IEEE Xplore 2980-2988 |
| `li2019crowdpose` | 10863--10872 | IEEE Xplore 10855-10864 |
| `li2021rle` | 11025--11034 | IEEE Xplore 11005-11014 |
| `li2021tokenpose` | 11313--11322 | IEEE Xplore 11293-11302 |
| `papandreou2018personlab` | 269--286 | Springer LNCS 282-299 |
| `shi2019twostreamadaptive` | 12026--12035 | IEEE Xplore 12018-12027 |
| `shi2022petr` | 11069--11078 | IEEE Xplore 11059-11068 |
| `sun2018integral` | 529--545 | Springer LNCS 536-553 |
| `xiao2018simplebaselines` | 466--481 | Springer LNCS 472-487 |

### D2. Existing pages that look wrong / conflict

| Key | Current | Proposed | Note |
|---|---|---|---|
| `zhao2019semanticgcn` | 3425--3435 | 3420--3430 | IEEE DOI record differs; verify on IEEE Xplore |
| `ning2020lighttrack` | 1034--1035 | 4456--4465 | current 1034-1035 is a 2-page span — almost certainly wrong; recommend the change |

### D3. Optional DOIs (not added)

arXiv-only and NeurIPS papers: these DOIs exist but are non-standard to cite. Add if you want a DOI on every entry.

| Key | Optional DOI | Type |
|---|---|---|
| `jiang2023rtmpose` | 10.48550/arXiv.2303.07399 | arXiv |
| `jiang2024rtmw` | 10.48550/arXiv.2407.08634 | arXiv |
| `xiao2022querypose` | 10.52202/068431-0905 | NeurIPS proceedings.com |
| `xu2022vitpose` | 10.52202/068431-2795 | NeurIPS proceedings.com |

_Plus arXiv DOIs (`10.48550/arXiv.<id>`) are available for every arXiv-only entry (ba2016layernorm, oord2018infonce, kochurov2020geoopt, nibali2018dsnt, hendrycks2016gelu, jiang2023rtmpose, jiang2024rtmw) and for the ICLR/NeurIPS papers, if desired._

### D4. Other judgment calls

- **`liu2020gastnet`** — arXiv preprint (5 authors, matches PDF) vs published IEEE ICRA 2021 version (7 authors, DOI 10.1109/ICRA48506.2021.9561605, pp 3374-3380). Current entry = preprint. Upgrade to the ICRA version?
- **`vonluxburg2007spectral`** — optional: change `Von Luxburg, Ulrike` → `von Luxburg, Ulrike` (lowercase nobiliary particle; renders "U. von Luxburg", sorts under L). Not applied.
- **`sarkar2011treeembed`** — Conference year 2011 (GD 2011) kept; Springer LNCS proceedings list 2012. Both defensible.
- **`chen2022hpesurvey`** — key says 2022; field year 2020 (CVIU online Mar 2020) is correct — cosmetic key-name mismatch only
- **`xu2023vitposepp`**, **`zheng2023hpesurvey`** — year kept as the early-access/online year (2023); the bound-volume issue date is 2024. Both defensible; DOI added either way.

## E. Entries with no change (verified correct)

`arthur2007kmeanspp`, `ba2016layernorm`, `blender`, `brody2022gatv2`, `chen2020simclr`, `fey2019pyg`, `gilmer2017mpnn`, `hamilton2017graphsage`, `hendrycks2016gelu`, `jiang2023rtmpose`, `jiang2024rtmw`, `kipf2017gcn`, `kochurov2020geoopt`, `li2018hyperband`, `liu2020gastnet`, `locatello2020slotattention`, `loshchilov2017cosineannealing`, `loshchilov2019adamw`, `macqueen1967kmeans`, `newell2017associative`, `nibali2018dsnt`, `nickel2018lorentz`, `oord2018infonce`, `sala2018learningmrep`, `scikitlearn`, `tsitsulin2023dmon`, `ultralytics2024yolo11`, `vandermaaten2008tsne`, `velickovic2018gat`, `wandb`, `xiu2018poseflow`, `yuan2021hrformer`
