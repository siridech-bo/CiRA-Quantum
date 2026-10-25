# Manuscript compile: handoff

**Repository:** https://github.com/siridech-bo/CiRA-Quantum, branch **`qrc-simulation`**. The QRC work is not on `main`.

```
git clone -b qrc-simulation https://github.com/siridech-bo/CiRA-Quantum.git
cd CiRA-Quantum/docs/QRC
pdflatex QRC_Manuscript.tex
pdflatex QRC_Manuscript.tex
```

Run pdflatex twice so the references resolve. The bibliography is inline (`thebibliography`), so there is no BibTeX step.

## Requirements

- **Packages:** RevTeX 4.2 (`revtex4-2`), `graphicx`, `bm`, `physics`, `booktabs`, `hyperref`, plus the `cm-super` fonts. These are in TeX Live or MiKTeX full.
- **Overleaf:** upload `QRC_Manuscript.tex` and the five figures below into a `figures/` folder.

## Figures

The manuscript uses `\graphicspath{{figures/}}`. All five figures are committed in `docs/QRC/figures/`.

| file | Fig. | regenerate with |
|---|---|---|
| `figR0_pipeline.png` | 1, pipeline schematic | `python docs/QRC/figures/code/qrc_pipeline_diagram.py` |
| `figR2_memory_gain.png` | memory vs encoding gain | `python docs/QRC/figures/code/qrc_realworld_figures.py` |
| `figR3_horizon.png` | R² by horizon, Lagos | same script |
| `figR1_confirm_forest.png` | 9-city paired CIs | same script |
| `figR4_drift.png` | drift test | same script |

## Real RevTeX compile (2026-10-25)

- **Result:** TeX Live 2026 (TinyTeX) with `revtex4-2`, two `pdflatex` passes. 0 errors, 0 undefined references or citations, 0 overfull boxes, 28 pages.
- **PDF:** the compiled file is committed as `docs/QRC/QRC_Manuscript.pdf`.
- **Fixes the real compile needed:**
  - `siunitx` removed. In this RevTeX 4.2 setup it loads `array`, which breaks any `p{}` table column with "Extra \or" and then hangs pdflatex. The two unit macros used, `\SI` and `\SIrange`, are defined in the preamble instead. **Do not re-add `siunitx` or `array`.**
  - The class option is `preprint`, which is single-column, so `table*` spans only one column. Wide tables were fitted:
    - `tab:prereg`: wrapping ragged-right `p{}` columns via the local `\rr` macro.
    - `tab:bench`: `\scriptsize` and shorter labels.
    - `tab:confirmR2`: `\footnotesize`.
    - `tab:confirmDelta`: each CI stacked under its point estimate, all numbers unchanged.
  - Acknowledgments: file paths set as `\url{}` so they can break.
- **Layout checked by eye:** the title block, Fig. 1, all wide tables and the bibliography render. In preprint mode the abstract is on its own page, which is RevTeX's standard.

## Status of the text

- **Static check passes:** braces, environments, every `\ref` and `\cite` resolve, and table columns are consistent.
- **Compiled cleanly with RevTeX (see above).** Report any new LaTeX error back rather than rewriting content. Every number has a source comment next to it (see `docs/QRC/data/MANIFEST.md`).

## Open items: not compile errors, need author input

- `\author{[Author list to confirm]}`, line 36.
- Acknowledgments: `[Funding and computing acknowledgments to be added.]`, line ~987. Also put the Zenodo DOIs here once the archive is published (`docs/QRC/ZENODO_HANDOFF.md`).
- Four bibliography entries marked `[details to verify]`: Hou2026, Das2026, Kubota2023, Suzuki2022.
- Dataset citations are missing (comment at line ~365): ERA5 / Open-Meteo, ETDataset, UCI PRSA (Beijing PM2.5), Santa Fe A, exchange-rate and electricity (LTSF collection).
