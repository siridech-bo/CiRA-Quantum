"""Generate the encoding-study manuscript addendum as a .docx (python-docx).

Mirrors docs/QRC/QRC_Encoding_Study.md: methodology, results (with the two
tables + Fig. 9 embedded), discussion, reproducibility. Run from ``backend``::

    python scripts/qrc_make_encoding_report.py
"""
from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[2] / "docs" / "QRC"
FIG9 = ROOT / "figures" / "fig9_encoding_judging.png"
FIG10 = ROOT / "figures" / "fig10_phaseamp.png"
FIG11 = ROOT / "figures" / "fig11_protons.png"
FIG15 = ROOT / "figures" / "fig15_6spin_multiseed.png"
FIG16 = ROOT / "figures" / "fig16_correlation_readout.png"
FIG6 = ROOT / "figures" / "fig6_weather_sim_vs_expt.png"
OUT = ROOT / "QRC_Encoding_Study.docx"

ENCODINGS = [
    ("arcsin_sqrt", "θ = arcsin(√s)", "[0, π/2]", "Hou 2026 (Paper 4)"),
    ("arccos", "θ = arccos(2s − 1)", "[0, π]", "Negoro 2018 (Paper 3)"),
    ("linear", "θ = s·π", "[0, π]", "baseline"),
    ("sinusoidal", "θ = π·sin²(s)", "[0, π·sin²1]", "emphasizes mid-range"),
    ("logarithmic", "θ = π·log(1+s)", "[0, π·log2]", "compresses range"),
    ("polynomial", "θ = π·s³", "[0, π]", "emphasizes extremes"),
    ("exponential", "θ = π·(1 − e^−s)", "[0, π(1−1/e)]", "smooth saturation"),
]
RESULTS = [
    ("arcsin_sqrt", "5.51", "3.66", "1.86", "0.595"),
    ("exponential", "3.66", "2.49", "1.17", "0.680"),
    ("logarithmic", "3.12", "2.15", "0.97", "0.720"),
    ("sinusoidal", "2.68", "1.83", "0.85", "0.744"),
    ("arccos", "2.67", "1.58", "1.09", "0.950"),
    ("linear", "1.83", "0.99", "0.84", "1.012"),
    ("polynomial", "1.55", "0.73", "0.83", "1.084"),
]


def _h(doc, text, level):
    doc.add_heading(text, level=level)


def _p(doc, text, *, italic=False, size=None):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.italic = italic
    if size:
        r.font.size = Pt(size)
    return p


def _table(doc, header, rows, bold_first_row_data=True):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Light Grid Accent 1"
    for i, h in enumerate(header):
        c = t.rows[0].cells[i].paragraphs[0].add_run(h)
        c.bold = True
    for ri, row in enumerate(rows):
        cells = t.add_row().cells
        for ci, val in enumerate(row):
            run = cells[ci].paragraphs[0].add_run(str(val))
            if bold_first_row_data and ri == 0:
                run.bold = True
    return t


def main() -> None:
    doc = Document()
    title = doc.add_heading("Input-Encoding Optimization for NMR Quantum Reservoir Computing", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _p(doc,
       "Manuscript addendum — systematic comparison of input-encoding functions on a "
       "9-spin ¹³C crotonic-acid reservoir, judged by intrinsic reservoir-capacity "
       "metrics. Companion to the main reproduction manuscript (Hou et al. 2026, "
       "PRL 136, 120602) and to §5.2 (FID feature representations, Fig. 8).",
       italic=True, size=9).alignment = WD_ALIGN_PARAGRAPH.CENTER

    _h(doc, "1. Motivation", 1)
    _p(doc,
       "Input encoding — the map from a classical scalar s ∈ [0,1] to an RF-pulse "
       "rotation angle θ(s) — is a foundational but under-studied design choice in QRC. "
       "Existing experiments each fix an encoding (arcsin√s in Hou 2026; arccos(2s−1) in "
       "Negoro 2018; s·π in linear schemes) and none compares them systematically. Yet "
       "the encoding is the only nonlinearity applied at the input stage: the reservoir "
       "Hamiltonian is fixed, so the curvature of θ(s) directly shapes how much of the "
       "input range is used and how much nonlinearity is injected. We compare seven "
       "encoding functions on the same reservoir and rank them by a panel of intrinsic "
       "reservoir-quality metrics rather than a single downstream forecast.")

    _h(doc, "2. Encoding functions tested", 1)
    _table(doc, ["name", "formula", "range", "origin"], ENCODINGS, bold_first_row_data=False)

    _h(doc, "3. Methodology", 1)
    _h(doc, "3.1 Reservoir and per-encoding evolution", 2)
    _p(doc,
       "The reservoir is the exact 9-spin ¹³C crotonic-acid system used in the main "
       "reproduction (rotating-frame H = Σ π νᵢ σᶻᵢ + Σ (π/2) Jᵢⱼ σᶻᵢσᶻⱼ, Lindblad "
       "T₁/T₂ dissipation providing fading memory), read out through the time-multiplexed "
       "FID signal. Because changing the encoding changes the reservoir dynamics, each "
       "encoding requires its own reservoir evolution (unlike the feature study of §5.2, "
       "which re-scores one cached trace). Fidelity ('quick'): fid_points=512, "
       "n_virtual=10, split (30,110,70)=210 steps, GPU evolution, seed 42. All settings "
       "other than the encoding are held identical, so metric differences are "
       "attributable to the encoding alone.")
    _h(doc, "3.2 Why not a downstream forecast (weather R²)?", 2)
    _p(doc,
       "Ranking encodings by weather-forecast R² failed: at any affordable fidelity the "
       "metric was noise-dominated (mostly negative cross-validated R², error bars larger "
       "than the differences). The cause is a fidelity wall — a temperature forecast is a "
       "demanding long-memory task that is only well-conditioned near full fidelity "
       "(~14–18 h GPU per encoding), infeasible across seven. This is itself a finding: "
       "downstream task R² is the wrong tool for screening encodings, conflating encoding "
       "quality with the task's sample/fidelity budget.")
    _h(doc, "3.3 The judging panel — intrinsic, fidelity-robust metrics", 2)
    _p(doc,
       "Each encoding is judged by intrinsic metrics well-defined at modest fidelity. The "
       "reservoir is driven by uniform i.i.d. random input u ∈ [0,1] and its FID readout "
       "matrix X is scored on: (i) linear memory capacity (MC) — held-out squared "
       "correlation reconstructing u[t−k] over delays k=1…30; (ii) nonlinear "
       "information-processing capacity (IPC) — as MC but with degree-2 and degree-3 "
       "Legendre-polynomial targets P_d(u[t−k]), the orthogonal basis; (iii) NARMA-10 "
       "NMSE — a standard nonlinear task computed on the same random drive (rescaled to "
       "0.5·u for recurrence stability); (iv) effective dimensionality — participation "
       "ratio of the state covariance. Rigor: train/test split, readout fit on train and "
       "scored on test, PCA-50 conditioning so p<n (in-sample capacity would be "
       "overfit-inflated). The MC estimator was validated on a synthetic 8-tap delay line "
       "(returns linear MC = 8.00 exactly, nonlinear ≈ 0).")
    _h(doc, "3.4 Data provenance and reproducibility", 2)
    _p(doc,
       "Every reservoir evolution is expensive; every metric is cheap. Each encoding's raw "
       "per-step FID waveform is persisted to disk (memcap_<fn>_<hash>.npz, including the "
       "exact random input) BEFORE any metric is computed. Thus all metrics are computed "
       "offline from the saved waveforms; any future metric or alternate-readout "
       "re-analysis is free with no re-evolution; and every FID/spectrum in the "
       "control-plane UI names the exact saved waveform it was drawn from, so figures are "
       "independently verifiable against on-disk data.")
    _h(doc, "3.5 Readout-robustness control", 2)
    _p(doc,
       "Because each metric is computed through a readout representation, the panel is "
       "computed under two readouts: the Paper-4 magnitude653 spectral-peak set (primary "
       "verdict) and the richer multimodal set (§5.2). Identical rankings under both imply "
       "readout-independence.")

    _h(doc, "4. Results", 1)
    _p(doc, "Multi-metric panel, magnitude653 readout (Fig. 9):")
    _table(doc, ["encoding", "total capacity", "linear MC", "nonlinear IPC", "NARMA-10 NMSE ↓"], RESULTS)
    _p(doc, "(Effective dimensionality was ≈ 1.1 for all encodings and did not discriminate — see §5.)",
       italic=True, size=9)
    if FIG9.exists():
        doc.add_picture(str(FIG9), width=Inches(6.2))
        cap = _p(doc,
                 "Figure 9: encoding functions judged on (A) total capacity (linear MC + "
                 "nonlinear IPC, stacked), (B) NARMA-10 NMSE (lower better), (C) effective "
                 "dimensionality, magnitude653 readout. arcsin_sqrt leads every panel.",
                 italic=True, size=9)
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for txt in [
        "arcsin_sqrt wins on every discriminating metric: the most linear memory (3.66), "
        "the most nonlinearity (IPC 1.86, ≈1.6× the runner-up), the highest total capacity "
        "(5.51 vs 3.66 for #2), and the best task performance (NARMA NMSE 0.595 — the only "
        "encoding clearly below 1; linear and polynomial are ≥1.0, no better than the mean).",
        "The ranking is well-separated, decreasing monotonically 5.51 → 1.55 (a factor "
        "~3.5), far larger than plausible estimator noise — in sharp contrast to the "
        "weather-R² attempt, where encodings were statistically indistinguishable.",
        "The ranking is readout-independent: recomputing total capacity under the "
        "multimodal readout yields the identical order.",
    ]:
        b = doc.add_paragraph(style="List Bullet")
        b.add_run(txt)

    _h(doc, "4.1 Phase-amplitude encoding does not help", 2)
    _p(doc,
       "The seven functions all inject the input through a single amplitude channel R_x(θ(s)). "
       "A natural extension adds a second degree of freedom per input — a phase-amplitude pulse "
       "R_z(2π·s)·R_x(θ(s)) that also rotates each spin about z by an input-proportional angle. "
       "We tested this on the winning encoding (arcsin_sqrt) with a dedicated GPU run, then "
       "scored it against the plain-amplitude baseline by recomputing memory capacity on both "
       "persisted waveforms under identical settings (kmax=30, washout=10, n_pca=50, degrees 1–3):")
    _table(doc, ["arcsin_sqrt variant", "linear MC", "nonlinear MC", "total MC"],
           [("amplitude only  R_x(θ)", "4.25", "6.55", "10.81"),
            ("+ phase-amplitude  R_z(2πs)·R_x(θ)", "1.01", "2.05", "3.06")])
    if FIG10.exists():
        doc.add_picture(str(FIG10), width=Inches(6.2))
        cap = _p(doc,
                 "Figure 10: (A) memory-capacity spectrum, identical settings on both persisted "
                 "waveforms; (B) representative FID at step 105 — the phase-amp trace is compressed "
                 "relative to amplitude-only.",
                 italic=True, size=9)
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _p(doc,
       "Adding the phase channel reduces total capacity by ≈72% — and it hurts both the linear "
       "(−76%) and nonlinear (−69%) components, so it is not a memory-for-nonlinearity trade. "
       "The extra R_z(2π·s) rotation collapses the FID dynamic range: it drives the encoded "
       "states toward a smaller, more scrambled region of the readout manifold rather than "
       "spreading them into new independent directions. For this NMR reservoir and its "
       "FID-magnitude/quadrature readout, the phase degree of freedom is not observable in a way "
       "that adds reservoir information — it only dilutes the amplitude signal arcsin√ had already "
       "placed optimally. Plain amplitude encoding remains the best choice; the second channel is "
       "counter-productive here. (Single-seed, quick fidelity, as in §5 limitations; the ≈3.5× "
       "gap makes the direction robust to noise.)")

    _h(doc, "4.2 Protons-only injection loses the nonlinearity the carbons feed", 2)
    _p(doc,
       "The baseline pulses the input into all nine spins. The FID is read out from the five "
       "protons (H1–H5, indices 4–8) while the four carbons (C1–C4) act as a spectator bath "
       "(§3.1), so a natural question is whether pulsing the bath carbons contributes anything, "
       "or whether encoding only into the readout protons (target_qubits=[4,5,6,7,8]) is as good "
       "or better. We ran the proton-only variant of the winning encoding and scored it against "
       "the all-spins baseline under identical settings:")
    _table(doc, ["arcsin_sqrt injection", "linear MC", "nonlinear MC", "total MC"],
           [("all 9 spins (baseline)", "4.25", "6.55", "10.81"),
            ("protons only [H1–H5]", "4.26", "4.02", "8.28")])
    if FIG11.exists():
        doc.add_picture(str(FIG11), width=Inches(6.2))
        cap = _p(doc,
                 "Figure 11: (A) memory-capacity spectrum, identical settings; (B) representative "
                 "FID — the single-trace envelopes are similar; the difference is in the cross-input "
                 "nonlinear structure, not the waveform shape.",
                 italic=True, size=9)
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _p(doc,
       "The result is a clean dissociation: linear memory is unchanged (4.25 → 4.26), but "
       "nonlinear capacity drops ≈39% (6.55 → 4.02), for a −23% total. The two spin groups play "
       "distinct roles. The linear memory lives on the protons (the readout nuclei), so injecting "
       "the input directly into them or into the carbons-then-coupling-in are equally good for "
       "remembering past inputs. But the carbons supply nonlinearity: driving the bath carbons "
       "pushes the input through the inter-nuclear J-couplings before it reaches the detected "
       "protons — an extra layer of coupled evolution that mixes and multiplies past inputs, "
       "exactly what a reservoir needs. Removing carbon injection removes that layer. All-spins "
       "injection is therefore better, and the 'spectator' carbons are not merely a bath term to "
       "optimize away at the encoding stage — they are an active nonlinear resource.")
    _p(doc,
       "Phase-2 summary. Across the three sub-questions — which function (§4), amplitude vs. "
       "phase-amplitude (§4.1), and all-spins vs. protons-only (§4.2) — the answer is consistent: "
       "arcsin_sqrt amplitude encoding into all nine spins is the best configuration. The two ways "
       "of adding structure beyond it both reduce capacity, and both do so by losing nonlinearity "
       "rather than memory.")

    _h(doc, "5. Discussion", 1)
    for head, body in [
        ("Mechanism — why arcsin_sqrt wins",
         "With the Hamiltonian fixed, θ(s) is the sole input nonlinearity, and two "
         "properties govern quality: how uniformly it spreads inputs across the rotation "
         "range (memory), and how much curvature it injects (nonlinearity). arcsin(√s) is "
         "the angle whose excitation probability sin²θ = s is linear in the input — it "
         "spreads inputs evenly over the Bloch polar angle, using the full readout range. "
         "The losers fail here: θ = π·s³ (polynomial) compresses almost all inputs toward "
         "θ ≈ 0 (near-identity pulses), giving the lowest memory AND nonlinearity; linear "
         "similarly underuses the range. That the losers are worst on both memory and "
         "nonlinearity indicates the dominant effect is input-range utilization, with "
         "arcsin√'s curvature adding a nonlinearity bonus (its IPC lead exceeds its MC lead)."),
        ("Validation of the literature choice",
         "Paper 4's θ = arcsin(√s) is confirmed as the best of the seven — for the first "
         "time with a mechanistic and quantitative justification rather than as an "
         "unexplained convention."),
        ("Decoupling of encoding and readout",
         "Readout-independence, together with the §5.2 finding that feature representations "
         "are low-headroom, indicates encoding and readout are largely separable design "
         "axes: the encoding governs intrinsic reservoir quality, the readout contributes "
         "little once the reservoir is fixed. Optimize the encoding first, the readout "
         "second."),
        ("Methodological contribution",
         "The fidelity-wall failure of weather-R² and the success of the intrinsic-capacity "
         "panel argue a general point: use intrinsic capacity metrics (MC/IPC), not a "
         "demanding downstream task, to screen QRC design choices."),
        ("Limitations",
         "(i) Single seed, quick fidelity — the large monotonic gaps make the ranking "
         "robust to noise, but multiple seeds and higher fidelity would attach formal error "
         "bars. (ii) Effective dimensionality did not discriminate (≈1.1 for all), expected "
         "to become informative only at larger n_virtual/system size. (iii) Absolute "
         "capacities are modest and should be read as relative comparisons."),
        ("Future work",
         "(i) Phase-amplitude encoding R_z(2πs)·R_x(θ) — resolved (§4.1): it lowers capacity "
         "by ≈72%, so a second (phase) channel is not the way to beat arcsin√ on this system. "
         "(ii) Learned encoding (gradient optimization) — resolved in §6: a gradient-learned "
         "per-spin encoding beats arcsin√ by ~40% on an encoding-sensitive task at 6 spins. "
         "(iii) Encoding×feature interaction. (iv) Higher-fidelity, multi-seed confirmation."),
    ]:
        p = doc.add_paragraph()
        p.add_run(head + ". ").bold = True
        p.add_run(body)

    _h(doc, "6. Beyond fixed encodings: a gradient-learned per-spin encoding", 1)
    _p(doc,
       "The comparison so far ranks hand-designed encodings and finds arcsin√ best. The next "
       "question is whether a gradient-learned encoding can beat it. We make the reservoir step "
       "differentiable and train an encoding network by gradient descent through the quantum "
       "evolution (the 'molecule as a Quantum Neural ODE'; full methodology + the real-hardware "
       "gradient are in the companion QRC_Learnable_Encoding_Concept document).")
    _h(doc, "6.1 Method", 2)
    _p(doc,
       "What is trained: only the encoding network W (a small MLP s → pulse angles) is "
       "gradient-trained (Adam). The Hamiltonian and dissipation are fixed by the molecule (the "
       "'hidden layers'); the readout is a closed-form ridge (differentiable). Autograd through "
       "the Lindblad evolution was verified against finite differences to ~1e-9 (complex128) and "
       "independently reproduced by the hardware parameter-shift rule in simulation to machine "
       "precision. Two encodings are learned: a global angle θ(s) (arcsin√'s structure) and a "
       "per-spin angle vector θᵢ(s) (frequency-selective — each spin its own learned map, which "
       "a global pulse cannot express).")
    _p(doc,
       "Task and cost: NARMA-2, deliberately not NARMA-10. NARMA-10 is memory-dominated (its "
       "difficulty is a 10-step autoregression), and memory lives in the fixed reservoir, so the "
       "encoding has almost no leverage there; NARMA-2 has short memory but a strong cubic input "
       "nonlinearity, so the encoding is the bottleneck — the fair test. Cost = NMSE via ridge on "
       "a leakage-free 3-way split (readout fit on train, encoder optimized on validation, test "
       "NMSE the held-out verdict). A dense exact-propagator formulation of the differentiable "
       "step makes the 6-spin runs ~1000× faster (minutes).")
    _h(doc, "6.2 Result — per-spin learning beats arcsin√ (6 spins, 5 seeds)", 2)
    _table(doc, ["encoding", "NARMA-2 test NMSE (5 seeds)", "vs arcsin√"],
           [("arcsin√ (baseline)", "0.42 ± 0.06", "—"),
            ("learned global", "0.78 ± 0.59", "unreliable"),
            ("learned per-spin", "0.25 ± 0.11", "−40%, beats 4/5 seeds")])
    if FIG15.exists():
        doc.add_picture(str(FIG15), width=Inches(6.2))
        cap = _p(doc,
                 "Figure 15: NARMA-2 test NMSE over 5 seeds. Per-spin learning beats arcsin√ in "
                 "4/5 seeds (~40% mean, paired t≈−3.0, p≈0.04), robust to the optimizer recipe; "
                 "learned global is unreliable.", italic=True, size=9)
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for txt in [
        "Per-spin learnable encoding beats arcsin√ — ~40% lower NARMA-2 error, 4/5 seeds, paired "
        "p≈0.04, and optimizer-robust (same verdict under loose lr 0.04 and a stabilized lr 0.02 "
        "+ grad-clip + best-val optimizer). First case in this study where any encoding beats arcsin√.",
        "The win is the per-spin (frequency-selective) degree of freedom, not 'learning' per se: a "
        "learned global angle map only marginally beats arcsin√ and its training is unreliable "
        "(diverges to ≈4× a mean-predictor on some seeds, regardless of LR/clipping). Only giving "
        "each spin its own learned input map — which arcsin√'s global pulse cannot — reliably helps.",
        "It is conditional, not universal: the advantage appears only for an encoding-sensitive "
        "task (NARMA-2) AND a rich enough reservoir. At 3 spins, with a global learned encoding, "
        "or on a memory-bound task, learning merely ties arcsin√. One 6-spin seed also only tied.",
    ]:
        doc.add_paragraph(style="List Bullet").add_run(txt)
    _h(doc, "6.3 Interpretation and limits", 2)
    _p(doc,
       "The result matches the §5 mechanism: arcsin√ is the optimal global amplitude map, so "
       "learning a global map cannot beat it by much (and is unstable). A per-spin encoding "
       "injects the input into different spins with different nonlinear maps, building a richer, "
       "higher-dimensional input embedding than any single global rotation — and on a task whose "
       "difficulty is the input nonlinearity (NARMA-2), that converts directly into lower error. "
       "It does nothing for memory-bound tasks, where the fixed reservoir is the bottleneck.")
    _p(doc,
       "Limits: (i) 6 spins — the full 9-spin system is out of reach for reverse-mode backprop "
       "(stiff dynamics need ~5300 sub-steps → ~200 GB autograd graph per input-step, measured); "
       "a 9-spin test needs the adjoint method, gradient checkpointing, or the memory-free hardware "
       "parameter-shift rule. (ii) Seed variance — one of five seeds tied. (iii) One task family "
       "(NARMA-2). (iv) The hardware realization (parameter-shift rule on a real spectrometer) is "
       "designed and validated in simulation but not yet run.")

    _h(doc, "7. The output side: correlation readout and the decoherence limit", 1)
    _p(doc,
       "§6 improved the input (encoding); the complementary lever is the readout. The standard QRC "
       "readout takes single-qubit observables <sigma_i>, discarding almost all of the 2^n-dim joint "
       "state n coupled qubits can hold. Reading multi-qubit correlations <sigma_i sigma_j> accesses "
       "the joint (entangled) state. On the 6-spin reservoir (fixed arcsin√, NARMA-2) we measured "
       "effective dimensionality (participation ratio of the feature covariance) and task NMSE:")
    _table(doc, ["readout / drive", "effective dim", "NARMA-2 test NMSE"],
           [("single-qubit, baseline τ", "1.4", "0.373"),
            ("+2-body correlations, baseline τ", "1.8", "0.304"),
            ("single-qubit, long τ", "1.1", "0.334"),
            ("+2-body correlations, long τ", "1.1", "0.207")])
    if FIG16.exists():
        doc.add_picture(str(FIG16), width=Inches(6.2))
        cap = _p(doc, "Figure 16: correlation readout lowers NARMA-2 error (~44% best case) but "
                 "effective dimensionality stays ~1-2; longer coherent evolution reduces it further.",
                 italic=True, size=9)
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for txt in [
        "Correlation readout genuinely helps: NARMA-2 NMSE 0.373 -> 0.207 (~44%). Reading the joint "
        "state extracts information single-qubit readout throws away -- a real, cheap performance "
        "lever, comparable to the learned encoding.",
        "But effective dimensionality stays ~1-2: even with 126 correlation features the reservoir "
        "lives in ~1-2 directions. The 2^n space is NOT used; the parallel qubits are not an "
        "exponential resource here.",
    ]:
        doc.add_paragraph(style="List Bullet").add_run(txt)
    _p(doc,
       "Mechanism: more coherent evolution LOWERED effective dim (1.4 -> 1.1). Decoherence (T2) "
       "collapses the state faster than the moderate Ising dynamics spread it across the 2^n space. "
       "The two resources -- many qubits (width) and coherence time (depth) -- are in TENSION, not "
       "additive: on crotonic-acid NMR the reservoir loses the entangling-vs-decoherence race, which "
       "is the mechanistic reason effective dimensionality is ~1. An exponential-width advantage "
       "needs T2 x coupling large enough to populate the space before it decoheres -- longer "
       "coherence and/or stronger, faster couplings than this molecule provides.")

    _h(doc, "8. Where quantum stands: the classical comparison", 1)
    _p(doc,
       "An encoding/readout improvement matters only if the substrate is worth using. We compared "
       "the quantum reservoir against two different classes of classical model.")

    _h(doc, "8.1 The Quantum RNN (closed-loop quantum-memory RNN)", 2)
    _p(doc,
       "A quantum reservoir is already a recurrent network: the density matrix rho_t is the hidden "
       "state and the Lindblad evolution rho_t = exp(tau L) U(theta_t) rho_{t-1} U(theta_t)^dag is "
       "the recurrence -- one fixed cell (encode -> evolve -> read) reused every timestep, the "
       "physics playing the role of shared, untrained recurrent weights. The learnable-encoding work "
       "of section 6 is thus a vanilla quantum RNN with a trained input map.")
    _p(doc,
       "The Quantum RNN adds a closed-loop feedback controller -- the one thing an LSTM has that a "
       "reservoir lacks: control that adapts to the current state. Each step a small MEMORYLESS MLP "
       "maps the input AND the previous reservoir readout to the per-spin drive angles:")
    for line in [
        "theta_t = controller(s_t, f_{t-1})     (memoryless MLP; f_{-1}=0)",
        "U_t     = tensor_i R_x(theta_{t,i})    (per-spin encoding pulse)",
        "rho_t   = exp(tau L) U_t rho_{t-1} U_t^dag   (quantum evolution = recurrence; memory in rho)",
        "f_t     = <O>(rho_t)                   (readout, fed back next step)",
        "y_hat   = ridge(f_t)                   (closed-form linear readout)",
    ]:
        mono = doc.add_paragraph()
        r = mono.add_run(line); r.font.name = "Consolas"; r.font.size = Pt(9)
    _p(doc,
       "The controller has NO recurrent state of its own -- the intended memory lives entirely in "
       "the quantum state rho. It is trained by backpropagation-through-time through the "
       "differentiable reservoir step; the readout is closed-form ridge. So it is a hybrid: a "
       "classical memoryless controller steering a quantum memory, with a feedback loop closed "
       "through the reservoir readout.")
    _p(doc,
       "Honest architectural caveat: feeding f_{t-1} back into the controller ALSO creates a "
       "classical recurrence -- the feature vector itself forms a hidden state f_t = G(f_{t-1}, s_t), "
       "i.e. a classical RNN whose hidden units are the measured observables. Memory can flow through "
       "two channels: rho (quantum) and f (classical feedback). The tau->0 ablation disentangles them.")

    _h(doc, "8.2 NARMA-2 scoreboard", 2)
    _p(doc, "All configurations on NARMA-2 (6-spin, leakage-free test NMSE, lower is better):")
    _table(doc, ["Configuration", "quantum memory", "test NMSE", "vs arcsin"],
           [("arcsin (fixed encoding) -- QRC baseline", "ON", "0.395", "--"),
            ("learned per-spin (open-loop, sec 6)", "ON", "0.216", "-45%"),
            ("Quantum RNN (closed-loop, feedback)", "ON", "0.137", "-65%"),
            ("  classical feedback RNN (closed-loop, tau->0)", "OFF", "0.011", "-97%"),
            ("per-spin, tau->0 (sanity)", "OFF", "1.010", "collapses")])
    for txt in [
        "The Quantum RNN (0.137) is the best quantum-involving configuration -- the feedback "
        "controller improves the open-loop reservoir (0.216 -> 0.137, -37%). If one is committed to "
        "using the quantum reservoir, this is the best way to run it.",
        "But the tau->0 row is decisive: with the quantum memory disabled the closed loop still "
        "reaches 0.011 -- near-perfect, and BETTER than with quantum memory on (0.137). The classical "
        "feedback channel alone (the f-recurrence) is a classical RNN that models NARMA-2's "
        "deterministic recurrence essentially exactly. So the Quantum RNN's gain is CLASSICAL: the "
        "quantum memory is redundant and mildly harmful.",
        "The clean quantum control is the last row: strip out the feedback (per-spin, tau->0) and the "
        "model collapses to a mean-predictor (1.010) -- a scalar encoder has no classical memory, so "
        "THAT result (the section-6 open-loop encoding) provably used the quantum reservoir; the "
        "Quantum RNN does not.",
    ]:
        doc.add_paragraph(style="List Bullet").add_run(txt)
    _p(doc,
       "Takeaway: the Quantum RNN is a real improvement to the quantum reservoir (best "
       "quantum-involving number), but the improvement is NOT quantum -- it is the classical feedback "
       "loop, which on its own beats the full quantum system ~12x. This is why NARMA-2 is a "
       "classical-friendly task and why the fair test of quantum value is reservoir-vs-reservoir "
       "(below) and, ultimately, tasks classical memory cannot handle (section 7).")

    _h(doc, "8.3 Reservoir vs. reservoir, and reservoir vs. trained RNN", 2)
    _p(doc,
       "vs. a classical RESERVOIR (ESN) -- the fair, same-paradigm comparison. Both use fixed "
       "dynamics + a trained linear readout. On real weather forecasting (Delhi climate; our sim "
       "reproduced against Hou et al. 2026, ESN baselines 500-10000 nodes) the quantum reservoir "
       "matches or beats the ESN, most clearly at long horizons for temperature (R2 ~0.8 at 45 days "
       "vs ESN ~0.4-0.7). As a reservoir, the quantum system is competitive-to-better on a real "
       "task. (Caveats: the core curve is partly digitized from the paper; ESN tuning and matched "
       "conditions warrant independent verification.)")
    if FIG6.exists():
        doc.add_picture(str(FIG6), width=Inches(6.2))
        cap = _p(doc, "Figure 6: weather forecasting R2 vs horizon -- quantum reservoir (QRC) vs "
                 "classical ESN, simulation and experiment (Hou et al. 2026).", italic=True, size=9)
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _p(doc,
       "vs. a TRAINED-recurrence model (RNN/LSTM) -- a harder, different bar. As the scoreboard "
       "(8.2) shows, on deterministic NARMA-2 a small trained feedback RNN reaches NMSE 0.011 (~20x "
       "better than the quantum reservoir's 0.216); a classical ESN would lose to it too. This is a "
       "limitation of reservoir computing in general, not of quantum specifically. A second control "
       "reinforces it: windowed encoding beats arcsin√ on NARMA-10 but a classical linear ridge on "
       "the same window matches it -- again the win is the classical window, not the quantum "
       "reservoir. Together with the Quantum RNN's tau->0 result, the tau->0 ablation separates "
       "genuine quantum-mediated results (the section-6 open-loop encoding, which collapses to a "
       "mean-predictor without the reservoir) from classically-explainable ones.")
    _p(doc,
       "The honest bound. (i) Learned per-spin encoding and correlation readout are real, "
       "quantum-mediated improvements to the quantum reservoir (§6, §7). (ii) As a reservoir, the "
       "quantum system beats a classical reservoir (ESN) on real forecasting (§8). (iii) It is NOT "
       "competitive with a trained RNN/LSTM on classical-friendly tasks -- and cannot be, because it "
       "is decoherence-limited to effective dimension ~1 (§7). A quantum ADVANTAGE (not merely a "
       "quantum improvement) must be sought where classical models cannot cheaply reach: tasks "
       "needing the exponential state space, quantum-native/quantum-sensor data, or "
       "hardware/energy-efficiency arguments -- on a platform whose coherence outlasts its "
       "entangling dynamics.")

    _h(doc, "9. Reproducibility", 1)
    for item in [
        "Per-encoding evolution + waveform persistence: scripts/qrc_memcap.py "
        "(--experiment all --fidelity quick).",
        "Memory-capacity / IPC estimator + delay-line self-test: scripts/qrc_memcap.py "
        "(memory_capacity), --selftest.",
        "Multi-metric judging + readout-robustness + Fig. 9: scripts/qrc_judge.py "
        "--glob 'artifacts/traces/memcap_*.npz'.",
        "Saved waveforms: artifacts/traces/memcap_<fn>_<hash>.npz (7 files), each with its "
        "exact random input, split, and configuration metadata.",
    ]:
        doc.add_paragraph(style="List Bullet").add_run(item)

    doc.save(str(OUT))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
