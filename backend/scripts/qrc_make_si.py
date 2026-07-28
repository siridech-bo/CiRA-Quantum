"""Build the Supplementary Material (.docx): full experimental settings."""
from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

FIG = Path("../docs/QRC/figures")
OUT = Path("../docs/QRC/QRC_Manuscript_SI.docx")

doc = Document()
normal = doc.styles["Normal"]
normal.font.name = "Times New Roman"
normal.font.size = Pt(11)
for i in range(1, 4):
    doc.styles[f"Heading {i}"].font.color.rgb = RGBColor(0, 0, 0)


def para(text="", *, italic=False, size=None, space=6, bold=False):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.italic = italic
    r.bold = bold
    if size:
        r.font.size = Pt(size)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(space)
    return p


def table(header, rows, caption, widths=None):
    c = doc.add_paragraph()
    r = c.add_run(caption)
    r.font.size = Pt(9)
    r.italic = True
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, h in enumerate(header):
        run = t.rows[0].cells[j].paragraphs[0].add_run(h)
        run.bold = True
        run.font.size = Pt(9)
    for row in rows:
        cells = t.add_row().cells
        for j, v in enumerate(row):
            run = cells[j].paragraphs[0].add_run(str(v))
            run.font.size = Pt(9)
    doc.add_paragraph().paragraph_format.space_after = Pt(6)


def figure(name, caption):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(FIG / name), width=Inches(4.8))
    c = doc.add_paragraph()
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = c.add_run(caption)
    r.font.size = Pt(9)
    r.italic = True


# --------------------------------------------------------------------------
title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
tr = title.add_run("Supplementary Material — Reproducing Experimental Quantum "
                   "Reservoir Computing in Simulation")
tr.bold = True
tr.font.size = Pt(14)
para("This document lists the complete experimental settings for every run reported "
     "in the main text, so that all results are reproducible from the released code.",
     size=10, space=12)

doc.add_heading("S1. Reservoir system (nine-spin crotonic acid)", level=1)
para("Rotating-frame Hamiltonian H = Σ_i πν_i σz_i + Σ_{i<j} (π/2) J_ij σz_i σz_j, with "
     "Lindblad relaxation (T1 amplitude damping, T2 pure dephasing, γφ = 1/T2 − 1/2T1). "
     "Chemical shifts are rotating-frame offsets referenced to 100.6273 MHz (13C) and "
     "400.2118 MHz (1H). Readout is on the five protons; the four carbons are an "
     "inaccessible coupled bath. Methyl protons H3–H5 are magnetically equivalent.")
table(["Spin", "ν (Hz)", "T1 (s)", "T2* (ms)"],
      [["C1", "−7749.7", "5.9", "212"], ["C2", "5430.1", "4.9", "231"],
       ["C3", "2699.9", "5.6", "208"], ["C4", "7673.7", "27.5", "241"],
       ["H1", "985.9", "3.2", "203"], ["H2", "520.3", "3.4", "332"],
       ["H3–H5 (methyl)", "−1081.5", "2.2", "320"]],
      "Table S1. Chemical shifts and relaxation times (Hou et al. SM, Table II).")
table(["Pair", "J (Hz)", "Pair", "J (Hz)", "Pair", "J (Hz)"],
      [["C1–C2", "40.8", "C2–C3", "69.5", "C1–C3", "1.6"],
       ["C1–C4", "8.5", "C2–C4", "1.4", "C3–C4", "71.0"],
       ["C1–H1", "4.0", "C2–H1", "155.6", "C3–H1", "−1.8"],
       ["C4–H1", "6.5", "C1–H2", "6.6", "C2–H2", "−0.7"],
       ["C3–H2", "162.9", "C4–H2", "3.3", "H1–H2", "15.8"],
       ["C1–Me", "128.0", "C2–Me", "−7.1", "C3–Me", "6.6"],
       ["C4–Me", "−0.9", "H1–Me", "6.9", "H2–Me", "−1.7"]],
      "Table S2. Scalar J-couplings (Hz); ‘Me’ = each methyl proton; intra-methyl J = 0.")

doc.add_heading("S2. Input encoding and FID readout", level=1)
para("Scalar inputs are normalised to [0,1] and encoded as a global rotation "
     "R_x(θ), θ = arcsin(√s). Multivariate inputs use frequency-selective encoding on "
     "distinct species (see per-task settings). Readout: a π/2 pulse on the protons, then "
     "the free-induction-decay S(t) = Tr[e^{tL}(UρU†) O_FID], O_FID = Σ_protons(σy+iσx), "
     "sampled on a uniform grid, Fourier-transformed; the largest-magnitude spectral bins "
     "(fixed across time steps) are the readout features. The non-readout (carbon) "
     "chemical-shift terms commute with the proton transverse operators and are dropped "
     "from the readout Liouvillian (exact; verified to ≈10⁻¹⁴).")

doc.add_heading("S3. Per-task settings", level=1)
table(["Setting", "NARMA", "Weather"],
      [["Reservoir", "crotonic9 (Table S1/S2)", "crotonic9 (Table S1/S2)"],
       ["Input", "superposition of sine waves", "Delhi daily temp + humidity"],
       ["Encoding", "arcsin(√s) on all spins", "temp→protons, humidity→carbons"],
       ["Evolution time τ", "0.01 s", "0.03 s"],
       ["FID samples (fid_points)", "2048", "2048 (ref.); 1024 (readout study)"],
       ["FID dwell", "0.3 ms", "0.3 ms"],
       ["Spectral peaks (features)", "653", "653"],
       ["Washout / train / test", "100 / 400 / 100", "374 / 600 / 500"],
       ["Targets", "NARMA-n, n=2,5,10,15,20", "horizon h=1,5,10,15,20,30,45 days"],
       ["Readout training", "ridge, 10-fold CV", "ridge, 10-fold CV"],
       ["Nonlinear variant", "—", "CV-tuned RBF-SVR on QRC feats (QRC+RBF)"],
       ["Metric", "NMSE = Σ(y−ŷ)²/Σy²", "R² per variable per horizon"],
       ["Multitasking", "one pass, readout per order", "one pass, readout per horizon"],
       ["Evolution backend", "GPU (torch CUDA)", "GPU (torch CUDA)"],
       ["Reservoir wall time", "≈4.0 h", "≈14.6 h (fid 2048)"],
       ["Seed", "42", "42"]],
      "Table S3. Complete NARMA and weather experiment settings.")
para("Dataset: Delhi Daily Climate (Kaggle), train+test concatenated (≈1576 days); "
     "temperature and humidity each min-max normalised to [0,1]. Weather series "
     "columns: meantemp, humidity. Classical baseline: leaky-integrator Echo State "
     "Networks (tanh, spectral radius 0.9, ridge readout) of 500/1000/5000/10000 nodes, "
     "driven by the same 2-D input with identical splits and horizons.", size=10)

doc.add_heading("S4. Evolution backends", level=1)
table(["Backend", "Method", "Memory", "N=9 cost / limit"],
      [["propagator", "dense Liouvillian superoperator", "4^N dense", "≈550 GB — infeasible"],
       ["action (CPU)", "sparse Krylov exp(t·L) (expm_multiply)", "≈0.001% dense", "≈6.5 s / matrix action"],
       ["gpu", "Taylor+substep exp(t·L), sparse CUDA matvecs", "sparse on GPU", "≈0.23 s (≈28× vs CPU)"],
       ["mesolve", "QuTiP adaptive ODE", "2^N×2^N", "reference; stiff past ~7 qubits"]],
      "Table S4. The four cross-validated evolution backends (agree to ≤10⁻³).")

doc.add_heading("S5. Independent validation (SLEEPY)", level=1)
para("A two-proton subsystem (H1, H2; ν = 985.9, 520.3 Hz; J = 15.8 Hz; "
     "T1 = 3.2, 3.4 s; T2 = 203, 332 ms) built identically in our QuTiP engine and in "
     "SLEEPY (sleepy-nmr 1.1.2). Complex FIDs (fid_points 1024/2048, dwell 0.3 ms) are "
     "compared: spectral peak positions agree to 0.27–0.49 Hz and the normalised FID "
     "traces to ≈0.3 % RMS — confirming the Lindblad implementation reproduces genuine "
     "NMR dynamics.")

doc.add_heading("S6. Software and hardware", level=1)
para("Python 3.12; QuTiP 5.3.0; PyTorch 2.10 (CUDA 12.8); SciPy 1.17; NumPy 2.2; "
     "scikit-learn; SLEEPY (sleepy-nmr) 1.1.2. GPU: NVIDIA GeForce RTX 5070 Ti (17 GB). "
     "All runs use fixed random seed 42; the GPU backend uses complex64 (validated to "
     "≤2×10⁻³ against the exact CPU path).", size=10)

doc.add_heading("S7. NARMA: simulation vs experiment", level=1)
para("For completeness, our simulated NARMA NMSE is compared directly with the "
     "experimental values of Hou et al. (Table I). The simulation reaches the same "
     "10⁻⁵–10⁻⁶ regime and, on higher orders, undercuts the experiment — expected, as the "
     "simulation is noise-free whereas the experiment carries cross-correlated relaxation "
     "and systematic errors.")
table(["NARMA n", "This work (sim)", "Hou et al. (experiment)"],
      [["2", "5.19×10⁻⁶", "1.74×10⁻⁷"], ["5", "5.21×10⁻⁵", "4.44×10⁻⁵"],
       ["10", "2.46×10⁻⁵", "5.84×10⁻⁵"], ["15", "1.82×10⁻⁵", "6.37×10⁻⁵"],
       ["20", "3.24×10⁻⁶", "4.34×10⁻⁵"]],
      "Table S5. NARMA NMSE: this work (simulation) vs Hou et al. Table I (experiment).")
figure("fig5_narma_vs_experiment.png",
       "Figure S1. NARMA NMSE versus order: this work (simulation, FID-653) against the "
       "experimental values of Hou et al. 2026 (Table I). Both occupy the same "
       "high-accuracy regime.")
para("Weather comparison with the experiment: Hou et al. report weather skill only as a "
     "figure (their Fig. 4b), without a numerical table. For the main-text sim-vs-experiment "
     "overlay (Figure 3) we digitised their QRC and QRC+RBF curves from Fig. 4b at our "
     "forecast horizons; these values are approximate (±~0.03) and are labelled as such. "
     "Digitised Paper-4 temperature R² (QRC+RBF): h=1,15,30,45 ≈ 0.92, 0.85, 0.83, 0.82; "
     "QRC ≈ 0.92, 0.80, 0.72, 0.67. Humidity (QRC+RBF) ≈ 0.72, 0.52, 0.48, 0.47.", size=10)

OUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(str(OUT))
print(f"saved {OUT.resolve()} ({OUT.stat().st_size} bytes)")
