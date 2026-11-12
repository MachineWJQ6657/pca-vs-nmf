# NMF vs. PCA on Olivetti faces — STOR 565 group project (code)

This repository holds the code behind the group report. We compare **PCA** (from class) with
**non-negative matrix factorization (NMF)** on the Olivetti faces (400 images of 40 people, 64 × 64
pixels), using one shared train/test split and one error definition: per-pixel test MSE on
**unclipped** reconstructions.

All parts are in **one notebook**, `STOR565_Project_Combined.ipynb`. Run it top to bottom.

## Team and parts of the notebook

Replace "Person N" with names before submitting.

| Member | Part | Where in the notebook | Status |
|---|---|---|---|
| Person 1 | Data preprocessing + PCA implementation | "PART P1" | original cells, copied unchanged |
| Person 2 | PCA analysis | "PART P2" | original cells, copied unchanged |
| Person 3 | NMF implementation | "PART P3" | original code; text translated to English; one `print` line changed (marked) |
| Person 4 | NMF theory and analysis | "PART P4" | revised and extended (main part) |
| — | Results showcase for the talk | "RESULTS SHOWCASE" | new |

Every cell copied from a teammate's notebook has the cell tag `original-P1`, `original-P2` or
`original-P3`. P1 and P2 are identical to the original notebooks. P3 was written in Chinese; its text
(markdown, code comments, printed messages) is translated into English (cells tagged `translated-P3`), and
its code is unchanged except one approved edit: in P3's last code cell (also tagged `modified-P3`) the final
`print` shows the relative output folder instead of an absolute path on the author's computer; the line
carries a comment saying so. A comparison of the code syntax trees against the original confirms this. The
only cells that are not from P1–P4 are the overview at the top and the short "Setup" cell (integration glue),
which puts the data file where P1/P2 and P3 each expect it.

## Folder layout

```text
project/
├── README.md
├── requirements.txt
├── STOR565_Project_Combined.ipynb  (the code; saved with outputs from one full run)
├── data/olivetti_preprocessed.npz
├── make_data.py                    (rebuilds / checks the data file)
├── report/                         (LaTeX source main.tex, figs/, compiled STOR565_Report.pdf)
├── slides/STOR565_slides_preview.pdf   (static preview of the talk slides)
├── results/                        (figures and tables from the saved run; large .npz files omitted)
└── (created when you run the notebook)  pca_results/  outputs/team_nmf/  p4_results/  showcase/
                                         pca_eigenfaces.png  pca_reconstruction_k50.png
```

The `results/` folder is a copy of what the notebook writes, kept so the figures and tables can be read
without running anything; running the notebook writes fresh copies next to it. To rebuild the report,
run `pdflatex main.tex` twice in `report/` (the figures are in `report/figs/`).

The setup cell looks for the data in `.`, `data/`, `../data/` and `upload/`, and copies it to the two
places the original parts read from (the notebook's folder for P1/P2, and `data/` for P3).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

Open `STOR565_Project_Combined.ipynb` and choose *Run All*. Everything runs on CPU; no GPU is needed.

### Tested environment

Last run end to end with Python 3.13.18, NumPy 2.5.3, SciPy 1.18.1, pandas 3.0.5, Matplotlib 3.11.2,
scikit-learn 1.9.1, threadpoolctl 3.7.0. The team's saved outputs came from older versions (Person 2:
NumPy 2.3.2 / scikit-learn 1.7.1; original Person 4: scikit-learn 1.6.1). The reported numbers were
identical across all of them. The fits use `random_state` and, in Part P4, a one-thread BLAS limit
(`threadpool_limits(limits=1)`) so results do not depend on the number of cores.

## Data

`data/olivetti_preprocessed.npz` contains:

| Array | Shape | Meaning |
|---|---|---|
| `X_train`, `X_test` | 320 × 4096, 80 × 4096 | flattened 64 × 64 grayscale images, values in [0, 1] |
| `y_train`, `y_test` | 320, 80 | person id (40 people; 8 training and 2 test images each) |
| `train_idx`, `test_idx` | 320, 80 | positions of the images in the original 400-image order |

SHA-256: `85be22755fab6319247cbfc7174e5afaf43c9a30c2f40ef14160418f10be4eef`

Source images: Olivetti Faces, AT&T Laboratories Cambridge. Part P4 asserts the split structure
(disjoint indices covering all 400 images, 8 + 2 images per person, values in [0, 1]) before using the
data. The split can be reproduced with
`train_test_split(np.arange(400), test_size=0.2, stratify=np.repeat(np.arange(40), 10), random_state=42)`;
we checked that this gives exactly the stored `train_idx` and `test_idx`.

**Data link and regeneration.** The course requires a data link with the code. The file is small
(1.8 MB), so submit it inside the archive or repository (`data/olivetti_preprocessed.npz`), and also cite
the public source of the images: Olivetti faces, AT&T Laboratories Cambridge, as distributed by
scikit-learn (`sklearn.datasets.fetch_olivetti_faces`; documentation:
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_olivetti_faces.html).
`make_data.py` rebuilds the file from that public dataset and compares it with the stored one:
`python make_data.py --check`. The split recipe in it is verified (it reproduces the stored
`train_idx` / `test_idx` exactly). The download step itself could not be run where this was prepared,
so **Person 1 should run `python make_data.py --check` once** and note the result here. If it prints
`MATCH`, the data are fully reproducible; if the `X_` arrays differ, add one line on how the images were
loaded and scaled.

## Run times

Measured on a 2-core cloud container, one run of the whole notebook:

| Part | Time | Writes to |
|---|---|---|
| P1 | a few seconds | `pca_eigenfaces.png`, `pca_reconstruction_k50.png` |
| P2 | ≈ 10 s | `pca_results/` |
| P3 | ≈ 25 s | `outputs/team_nmf/` |
| P4 | ≈ 7 min | `p4_results/` |
| Showcase | a few seconds | `showcase/` |

In Part P4 the slow sections are the component-count grid (6.1) and the initialization study (6.2);
together they account for most of the run time.

## What a correct run prints

If your output matches these numbers, the environment is working.

**P1:** PCA with 50 components, test MSE 0.003337.

**P2 (PCA, fitted on the training images; cumulative variance is training variance):**

| k | Train MSE | Test MSE | Cumulative variance |
|---|---|---|---|
| 5 | 0.008804 | 0.008860 | 0.540825 |
| 10 | 0.006621 | 0.006848 | 0.654674 |
| 25 | 0.003937 | 0.004565 | 0.794667 |
| 50 | 0.002331 | 0.003337 | 0.878398 |
| 100 | 0.001110 | 0.002503 | 0.942131 |

90% of the variance needs 62 components, 95% needs 111.

**P3 (NMF, k = 40; `nndsvda`, coordinate descent, 3000 iterations max, tol 1e-4, seed 42):**
1472 iterations, train MSE 0.00307863, test MSE 0.00394635, no convergence warnings.

**P4 (k = 40 comparison, test MSE):**

| Method | Test MSE | vs PCA |
|---|---|---|
| PCA, 40 components + mean | 0.003658 | — |
| PCA, 39 components + mean | 0.003696 | +1.0% |
| Uncentered SVD, 40 components (no mean) | 0.003658 | −0.005% |
| NMF, 40 components (Person 3 settings) | 0.003946 | +7.9% |

Further P4 results: NMF's test MSE is above PCA's at every k on Person 2's grid, with the gap rising
from 1.9% (k = 5) to 17.5% (k = 100). Across initializations and tolerances at k = 40 the gap is 5.5%
to 7.9%. PCA has the lower error on 74 of 80 test images. A typical image uses about 30 of the 40 NMF
patterns. The showcase section checks that P2, P3 and P4 agree on the shared numbers and stops with an
assertion error if they do not.

## Notes

- **Shared variable names.** P1–P3 and P4 use some of the same variable names (`nmf`, `H`, `OUT`, …).
  Part P4 reloads the data and defines everything it needs, so it does not depend on earlier parts, and
  the showcase reads the files the parts wrote. If you re-run a single part after running P4, restart
  the kernel first.
- **ConvergenceWarnings are expected in Part P4, Section 6.** NMF fits at k = 50 and k = 100, the
  stricter-tolerance run, and the random-start runs reach the iteration limit. The notebook records this
  in a "Hit iteration cap" column and marks those points with hollow markers; their errors overstate
  what NMF could reach with more iterations.
- **The P4 text is checked against the numbers.** The cell titled "Consistency check between the text
  and the computed values" fails if a claim in the written analysis is no longer supported by the run.
  If it fails after you change the data or settings, revisit the text rather than deleting the check.
- **Figures.** `p4_results/`: `01_components`, `02_additive_reconstruction`, `03_matched_reconstructions`,
  `04_convergence`, `06_paired_errors`, `07_k_grid`; `showcase/`: `headline_results`. Tables are saved as
  CSV (`comparison_k40`, `k_grid`, `sparsity_k40`, `init_sensitivity_k40`); `run_metadata.json` records the
  data hash, library versions and settings.
- **References.** Titles, authors, venues and pages were checked against publisher or arXiv pages,
  except the page range of Lee & Seung (2001), which is omitted, and the journal version of Vavasis,
  which is cited through its arXiv report. Check both before citing them in the report.

## Common problems

- `FileNotFoundError` for the data file: put `olivetti_preprocessed.npz` in `data/` or next to the notebook.
- Tiny numerical differences on another machine are possible (different BLAS builds can change float
  rounding). Part P4 reports its cross-check against Persons 2 and 3 as `True`/`False` without stopping
  the run; a `False` with differences in the last digits is not a bug, but a large difference is. The
  showcase's cross-part check uses a tight relative tolerance (1e-8) and will stop the run if it is not met.
- The notebook takes several minutes (see "Run times"), almost all of it in Part P4; this is normal.

## AI assistance

Claude (Anthropic) was used to review the Person 4 notebook against the course requirements and the other
team notebooks, to add the analyses in P4 Sections 3.1, 4.1, 5 and 6, to merge the four parts into one
notebook, to translate the text of Part P3 into English, to write the results showcase, and to draft this
README, `requirements.txt`, the LaTeX report and the slides. All numbers
above were produced by running the code. Record what was delegated, what was kept, and your own checks in
the report's AI appendix.

## Before submitting

- [ ] Replace "Person N" with names in the notebook and this README.
- [ ] Run `python make_data.py --check` once (Person 1) and record the result; include the data file or its link in the archive.
- [ ] Decide whether to keep P1's duplicated loading cell (cell 0 repeats cell 1; harmless when the data
      file is next to the notebook, which the setup cell ensures).
