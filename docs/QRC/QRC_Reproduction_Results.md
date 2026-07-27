# QRC Paper-4 Reproduction — Results

Results of the faithful Tier-A reproduction of **Hou et al. 2026 (PRL 136, 120602)**,
per [`QRC_Reproduction_Plan.md`](QRC_Reproduction_Plan.md). Raw numbers:
[`data/qrc_paper4_narma.json`](data/qrc_paper4_narma.json).

**Configuration.** 9-spin ¹³C crotonic-acid reservoir (`crotonic9_paper4`, exact SM
Table II parameters); Paper-4 sine-wave NARMA input; τ = 0.01 s; **FID readout** (π/2
proton pulse → 2048-point transverse acquisition → FFT → top-653 spectral peaks); ridge
readout, 400 train / 100 test; GPU (`evolution_mode="gpu"`), exact reduced-Liouvillian
FID. Single reservoir pass, 5 NARMA orders fit as separate readouts (multitasking).
Wall time ≈ 4.1 h (reservoir pass 4.0 h at ~24 s/step + readouts/ESN). repro_hash
`a0ed09f5b7c628a0`.

## 1. NARMA — reproduction succeeded

| NARMA | our NMSE (Σ/Σy²) | paper best | our R² |
|------:|:----------------:|:----------:|:------:|
| 2  | 5.19×10⁻⁶ | 1.74×10⁻⁷ | 0.9998 |
| 5  | 5.21×10⁻⁵ | 4.44×10⁻⁵ | 0.9997 |
| 10 | 2.46×10⁻⁵ | 5.84×10⁻⁵ | 0.9997 |
| 15 | 1.82×10⁻⁵ | 6.37×10⁻⁵ | 0.9995 |
| 20 | 3.24×10⁻⁶ | 4.34×10⁻⁵ | 0.9989 |

We land in the paper's **10⁻⁵–10⁻⁶ high-accuracy regime** across all orders (R² ≈ 0.999),
and on NARMA-10/15/20 we are **better** than the published experimental numbers — expected,
since ours is a noise-free simulation while the experiment carries systematic errors
(cross-correlated relaxation, RF inhomogeneity, drift; acknowledged in the paper).

**The decisive finding:** with observables-only readout the same system gives NMSE ≈ 0.25;
with the **FID-653 spectral readout** it drops to ≈ 10⁻⁶. The readout — turning ~9 static
observables into 653 independent time-multiplexed readout functions — was the missing
ingredient, exactly as the paper's thesis predicts ("information processing capacity is
fundamentally limited by the number of independent readout functions"). This supersedes the
earlier observables-only scaling study's pessimistic reading, which was a readout artifact.

## 2. Classical ESN baseline (NARMA-10) — the honest half

| ESN size | NMSE (Σ/Σy²) | R² |
|---------:|:------------:|:--:|
| 500   | 2.0×10⁻⁷ | 1.0000 |
| 1000  | 6.4×10⁻⁸ | 1.0000 |
| 5000  | 1.1×10⁻⁸ | 1.0000 |
| 10000 | 8.8×10⁻⁹ | 1.0000 |

On **NARMA**, a classical Echo State Network beats our QRC by ~100–1000×. This is *not* a
contradiction with the paper:

- NARMA is a canonical benchmark that ESNs are practically designed for; with the smooth
  sine-superposition input it is nearly trivial for a large ESN.
- Paper 4's NARMA comparison was only against a **weak** "classical spin-based counterpart"
  (avg R² < 0.75), **not** a strong ESN.
- Their **quantum-advantage claim is on weather forecasting** (QRC matching/beating ESNs of
  500–10000 nodes on real, chaotic Delhi climate data), **not** on NARMA.

**Conclusion so far.** We have faithfully **reproduced the paper's NARMA accuracy** and
confirmed the FID-readout mechanism. We have **not** reproduced a quantum *advantage* — and
on NARMA one is not expected. The advantage claim must be tested on the **weather task**,
which is the next step.

## 3. Independent validation (SLEEPY)

Separately, the underlying NMR physics is cross-checked against **SLEEPY** (an independent
Liouville-space NMR engine, Nature Comms 2025): a two-proton FID from our QuTiP `fid_signal`
matches SLEEPY to **0.3–0.5 Hz** on peak positions and **~0.3 % RMS** — so the reproduction
rests on physics validated three ways (internal cross-backend, SLEEPY, and the paper's
molecule/spectrum).

## 4. Reproduce

```bash
cd backend && pip install -e ".[qrc]"
python scripts/qrc_reproduce_paper4.py            # ~4 h on GPU; watch artifacts/paper4_full/progress.html
# faster reduced pass:
python scripts/qrc_reproduce_paper4.py --fid-points 1024 --orders 2 10
```

Cost note: the FID acquisition is a full quantum evolution over the ~0.6 s window **per
input step** (~24–42 s/step at N=9), so the run is GPU-hours; the runner uses the paper's
multitasking (one reservoir pass, N readouts) and checkpoints per order.

## 5. Next: weather forecasting

The real test of the paper's advantage claim. Multivariate encoding (temperature → protons,
humidity → carbons), single- and multi-step-ahead forecasting on the Delhi daily-climate
dataset (374 washout / 600 train / 600 test), compared against ESN(500–10000) and QRC+RBF-SVR
post-processing. Wired as `run_weather` in the runner; results will be appended here.
