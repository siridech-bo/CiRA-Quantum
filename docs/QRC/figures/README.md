# QRC figures — provenance

Which script writes each figure and where its numbers come from. "Data package" =
`docs/QRC/data/` (SHA-256 in `docs/QRC/data/MANIFEST.md`). All scripts are CPU-only.

| figure | script | numbers from | status |
|---|---|---|---|
| figP1–figP4 | `docs/QRC/figures/code/qrc_paper_figures.py` | data package + pinned run traces | **current** (manuscript) |
| fig4_weather | `backend/scripts/qrc_make_weather_fig.py` | `qrc_paper4_weather_v2.json` | current (Paper-4 reproduction) |
| fig5_narma_vs_experiment | `backend/scripts/qrc_make_narma_compare_fig.py` | `qrc_paper4_narma.json` (`qrc_narma.*.nmse_paper`, `paper_table_i.*.best`) | current; hardcoded values removed 2026-10-24 |
| fig6_weather_sim_vs_expt | `backend/scripts/qrc_make_weather_compare_fig.py` | `qrc_paper4_weather_v2.json`; Hou et al. Fig 4b digitized (literature, typed in) | current |
| fig7_fid_points | `backend/scripts/qrc_make_fid_compare_fig.py` | `qrc_paper4_weather.json`, `_v2.json`; Hou et al. digitized (typed in) | current |
| fig8_phase1_features | `backend/scripts/qrc_make_phase1_fig.py` | recomputed from the raw trace `backend/artifacts/traces/weather_full.npz` | current (no typed numbers) |
| fig1–fig3 | `backend/scripts/qrc_make_figures.py` | `backend/qrc_scaling_progress.jsonl` (not in the data package) | early scaling report only |
| fig15_6spin_multiseed | `backend/scripts/qrc_6spin_aggregate.py` | **typed in** from 2026-08 run logs (observable-readout era) | **SUPERSEDED** by figP2 + `qrc_paired_tests.md` — do not cite |
| fig20_classical_baseline | `backend/scripts/qrc_classical_baseline.py` | size-only ESN, data seed 1 only (`qrc_classical_baseline_v1_superseded.json`) | **SUPERSEDED** by the fair baselines (`qrc_classical_fair_{esn,lstm}.json`, figP3) — do not cite |
| fig9–fig14, fig16–fig19 | encoding-study / prototype scripts (`qrc_make_encoding_report.py`, `qrc_learnable_*.py`, …) | per-run outputs | exploratory; not in the manuscript |

Note: `qrc_learnable_9spin.py` rewrites `fig14_learnable_6spin.png` on every run, so that
file changes whenever a learnable run executes.

Paired statistics for the synthetic claims (learned vs arcsin, τ→0 ablation, tuned
LSTM/ESN): `docs/QRC/data/qrc_paired_tests.{json,md}` from
`backend/scripts/qrc_paired_tests.py`.
