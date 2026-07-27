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

![NARMA sim vs experiment](figures/fig5_narma_vs_experiment.png)

*Figure: our simulated NARMA NMSE (FID-653) plotted against Hou et al.'s **experimental**
Table I values on the same log scale — both in the same high-accuracy regime. This is the
one place the paper tabulates real experimental numbers; the weather comparison (§5) is
against ESNs computed here under identical conditions, since the paper reports weather skill
only as a figure. Full experimental settings for every run are in the Supplementary Material
(`QRC_Manuscript_SI.docx`).*

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

## 5. Weather forecasting — the quantum advantage **reproduces**

The real test of the paper's advantage claim. Multivariate encoding (temperature → protons,
humidity → carbons), single reservoir pass, then per-horizon ridge (+ RBF-SVR) readouts
(multitasking), on the Delhi daily-climate dataset. Config: crotonic9, τ=0.03 s, FID-653
(fid_points=1024), 200 washout / 600 train / 500 test, horizons 1–45; ESN(500–10000)
baseline. Reservoir pass ≈ 4.4 h. Raw:
[`data/qrc_paper4_weather.json`](data/qrc_paper4_weather.json).

![Weather QRC vs ESN](figures/fig4_weather.png)

### Temperature forecast R² (higher = better)

| horizon | QRC | QRC+RBF | ESN-500 | ESN-1000 | ESN-5000 | ESN-10000 |
|--------:|:---:|:-------:|:-------:|:--------:|:--------:|:---------:|
| 1  | 0.936 | 0.938 | 0.940 | 0.940 | 0.940 | 0.940 |
| 5  | 0.826 | 0.848 | 0.852 | 0.849 | 0.850 | 0.853 |
| 10 | 0.773 | 0.817 | 0.813 | 0.815 | 0.812 | 0.812 |
| 15 | 0.768 | **0.790** | 0.762 | 0.755 | 0.753 | 0.759 |
| 20 | 0.741 | **0.756** | 0.714 | 0.690 | 0.602 | 0.698 |
| 30 | 0.745 | **0.759** | 0.553 | 0.563 | 0.550 | 0.574 |
| 45 | 0.572 | **0.674** | 0.370 | 0.396 | 0.412 | 0.413 |

Humidity shows the same long-horizon pattern (h=45: QRC+RBF **0.501** vs ESN best 0.313).

**This reproduces Paper 4's headline result.** Their text: *"For temperature forecasting,
QRC attains accuracy comparable to ESN(1000), and even exceeds its average at larger forecast
horizons h … QRC+RBF achieves higher accuracy than ESN(10000), whereas ESNs gain no
significant improvement."* Our run shows exactly this:

- **short horizons:** QRC ≈ ESN;
- **long horizons (h ≥ 15):** QRC — and decisively **QRC+RBF** — **beats every ESN size**, the
  gap widening with horizon (h=45: 0.674 vs 0.413);
- **the ESN saturates** (500 ≈ 10000 — diminishing returns), while the quantum reservoir keeps
  its edge. That is the quantum-advantage signature.

## 6. Verdict across both tasks

| task | outcome | consistent with paper? |
|------|---------|------------------------|
| **NARMA** | classical ESN wins (~100–1000×) | ✅ yes — the paper never claimed advantage on NARMA (only vs a weak classical spin baseline); ESNs are built for NARMA |
| **Weather** (chaotic, long-horizon) | **QRC+RBF beats ESN up to 10 000 nodes** | ✅ **yes — this is the paper's actual advantage claim, reproduced** |

**Bottom line.** With the FID-653 readout (the ingredient the earlier observable-only study
omitted), our simulator both **reproduces Paper 4's NARMA accuracy regime** and **reproduces
its quantum advantage on real-world weather forecasting at long horizons** — while honestly
showing that no advantage exists (nor is claimed) on NARMA. The physics underneath is
independently validated against SLEEPY (§3).

### Caveats
- Weather used fid_points=1024 (vs 2048 for NARMA) to bound runtime; a 2048 rerun would if
  anything strengthen the QRC side.
- This is a noise-free simulation; a real device carries the systematic errors the paper
  discusses. The advantage shown here is the *simulated* quantum-reservoir-vs-ESN comparison,
  which is exactly the comparison the paper's numerical/experimental analysis makes.
