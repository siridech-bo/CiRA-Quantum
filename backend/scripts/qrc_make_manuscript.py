"""Build the journal-ready QRC manuscript as a .docx (python-docx).

Assembles the methods + results from the reproduction runs into a single
Word manuscript with numbered sections, captioned tables, and the figures
in docs/QRC/figures/. Output: docs/QRC/QRC_Manuscript.docx.
"""
from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

FIG = Path("../docs/QRC/figures")
OUT = Path("../docs/QRC/QRC_Manuscript.docx")

doc = Document()

# -- base style -------------------------------------------------------------
normal = doc.styles["Normal"]
normal.font.name = "Times New Roman"
normal.font.size = Pt(11)
for i in range(1, 4):
    h = doc.styles[f"Heading {i}"]
    h.font.name = "Times New Roman"
    h.font.color.rgb = RGBColor(0, 0, 0)


def para(text="", *, italic=False, align=None, size=None, bold=False, space=6):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.italic = italic
    r.bold = bold
    if size:
        r.font.size = Pt(size)
    if align == "center":
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    else:
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(space)
    return p


def heading(text, level=1):
    doc.add_heading(text, level=level)


def figure(name, caption):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(FIG / name), width=Inches(6.0))
    c = doc.add_paragraph()
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = c.add_run(caption)
    r.font.size = Pt(9)
    r.italic = True
    c.paragraph_format.space_after = Pt(10)


def table(header, rows, caption):
    c = doc.add_paragraph()
    r = c.add_run(caption)
    r.font.size = Pt(9)
    r.italic = True
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, htext in enumerate(header):
        cell = t.rows[0].cells[j]
        cell.text = ""
        run = cell.paragraphs[0].add_run(htext)
        run.bold = True
        run.font.size = Pt(9)
    for row in rows:
        cells = t.add_row().cells
        for j, val in enumerate(row):
            cells[j].text = ""
            run = cells[j].paragraphs[0].add_run(str(val))
            run.font.size = Pt(9)
    doc.add_paragraph().paragraph_format.space_after = Pt(6)


# ===========================================================================
# Front matter
# ===========================================================================
title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
tr = title.add_run(
    "Reproducing Experimental Quantum Reservoir Computing in Simulation: "
    "A GPU-Accelerated N-Qubit NMR Reservoir with Free-Induction-Decay "
    "Readout, Cross-Validated Against SLEEPY"
)
tr.bold = True
tr.font.size = Pt(15)

para("Siridech Boonsang", align="center", bold=True, space=0)
para("Faculty of Information Technology & CiRA CORE AI Center, "
     "King Mongkut's Institute of Technology Ladkrabang (KMITL), Bangkok, Thailand",
     align="center", italic=True, size=10, space=2)
para("Corresponding author: siridech.bo@kmitl.ac.th", align="center", size=9, space=12)

heading("Abstract", level=1)
para(
    "Quantum reservoir computing (QRC) exploits the natural dynamics of a quantum "
    "system as a computational substrate for temporal information processing. A recent "
    "experiment (Hou et al., Phys. Rev. Lett. 136, 120602, 2026) demonstrated "
    "state-of-the-art QRC on a nine-spin nuclear-magnetic-resonance (NMR) platform, "
    "reporting a practical quantum advantage over large classical echo state networks "
    "(ESNs) on real-world weather forecasting. We present an open, GPU-accelerated "
    "simulator that reproduces this work in silico and, critically, isolates the "
    "ingredient responsible for its performance: a time-multiplexed free-induction-decay "
    "(FID) spectral readout. Our simulator integrates the Lindblad master equation for an "
    "arbitrary N-spin weak-coupling NMR network, with four interchangeable evolution "
    "backends cross-validated to within 10^-3; a sparse Krylov (“action”) method and a "
    "GPU implementation remove the 4^N-dimensional dense-superoperator memory wall "
    "(≈550 GB at N=9) and accelerate the nine-qubit regime ≈28×. On the NARMA benchmark "
    "the simulator attains the paper’s 10^-5–10^-6 accuracy regime (R²≈0.999); on Delhi "
    "weather forecasting the quantum reservoir—particularly with radial-basis-function "
    "post-processing—matches classical ESNs at short horizons and outperforms ESNs of up "
    "to 10,000 nodes at long horizons (temperature R² 0.67 vs 0.41 at a 45-day horizon), "
    "reproducing the reported advantage, while ESN accuracy saturates with size. We show "
    "that a naive single-time observable readout gives NMSE ≈0.25, two-to-three orders of "
    "magnitude worse, confirming that the readout—not merely the qubit count—is decisive. "
    "The underlying NMR physics is independently validated against SLEEPY, a dedicated "
    "Liouville-space NMR engine, with FID spectral lines agreeing to sub-hertz precision. "
    "All code, data, and reproduction scripts are released."
)
para("Keywords: quantum reservoir computing; quantum machine learning; nuclear magnetic "
     "resonance; Lindblad master equation; echo state network; time-series forecasting.",
     italic=True, size=10)

# ===========================================================================
heading("1. Introduction", level=1)
para(
    "Reservoir computing processes temporal signals by driving them into a fixed, "
    "high-dimensional nonlinear dynamical system—the reservoir—and training only a linear "
    "readout on the reservoir state. Quantum reservoir computing (QRC) uses a quantum "
    "system as the reservoir, exploiting an exponentially large Hilbert space and complex "
    "many-body correlations. NMR spin ensembles are a natural physical substrate: they "
    "have well-characterised Hamiltonians (chemical shifts and scalar J-couplings), "
    "radio-frequency pulses for input encoding, and—crucially—relaxation (T1, T2) that "
    "supplies the fading memory reservoir computing requires."
)
para(
    "Hou et al. recently reported an experimental QRC on a nine-spin 13C-labelled crotonic "
    "acid sample, achieving one-to-two orders of magnitude lower NARMA error than prior "
    "quantum experiments and, on real-world weather forecasting, higher accuracy than "
    "classical echo state networks with thousands of nodes—a first experimental "
    "demonstration of a quantum machine-learning system outperforming large classical "
    "models on a realistic dataset. Two innovations underpin the result: the use of "
    "intrinsic spin relaxation as a computational resource (fading memory without qubit "
    "resets), and a time-multiplexed FID readout that raises the number of independent "
    "readout functions far above the handful of directly accessible single-time observables."
)
para(
    "This paper contributes an open, reproducible simulator for NMR-based QRC and uses it "
    "to (i) reproduce the NARMA and weather-forecasting results of Hou et al.; (ii) isolate "
    "the FID readout as the decisive ingredient by direct comparison with a single-time "
    "observable readout; (iii) make the nine-qubit open-system regime computationally "
    "routine through a GPU-accelerated, memory-lean exponential-integration scheme; and "
    "(iv) validate the underlying NMR physics against an independent, dedicated engine "
    "(SLEEPY). We report both the successes and the limits honestly: the classical ESN "
    "wins decisively on NARMA (which the original paper never disputed), whereas the "
    "quantum reservoir’s advantage appears—and is reproduced here—on the chaotic, "
    "long-horizon weather task."
)

# ===========================================================================
heading("2. Methods", level=1)

heading("2.1 NMR reservoir model", level=2)
para(
    "We model N spin-½ nuclei in the weak-coupling (secular) NMR regime. In the rotating "
    "frame the Hamiltonian is H = Σ_i πν_i σz_i + Σ_{i<j} (π/2) J_ij σz_i σz_j "
    "(angular frequency units), with ν_i the chemical-shift offsets (Hz) and J_ij the "
    "scalar couplings (Hz). Open-system evolution follows the Lindblad master equation "
    "dρ/dt = −i[H,ρ] + Σ_k (L_k ρ L_k† − ½{L_k†L_k, ρ}), with per-spin collapse operators "
    "for T1 amplitude damping (L = √(1/T1) σ−) and T2 pure dephasing "
    "(L = √(γφ/2) σz, γφ = 1/T2 − 1/2T1). Dissipation, not coherent evolution alone, "
    "produces the fading memory that reservoir computing requires; a pure-Schrödinger "
    "simulation retains perfect memory and cannot reproduce QRC behaviour. The nine-spin "
    "reproduction uses the exact crotonic-acid parameters of Hou et al. (Table 1); the "
    "simulator is generic in N and ships additional presets."
)
table(
    ["Spin", "ν (Hz)", "T1 (s)", "T2* (ms)"],
    [["C1", "−7749.7", "5.9", "212"], ["C2", "5430.1", "4.9", "231"],
     ["C3", "2699.9", "5.6", "208"], ["C4", "7673.7", "27.5", "241"],
     ["H1", "985.9", "3.2", "203"], ["H2", "520.3", "3.4", "332"],
     ["H3–5 (methyl)", "−1081.5", "2.2", "320"]],
    "Table 1. Nine-spin 13C crotonic-acid parameters (chemical shifts, relaxation times) "
    "used for the reproduction, transcribed from Hou et al. Supplemental Material. "
    "J-couplings (not shown) range 0.7–163 Hz. Readout is performed on the five protons; "
    "the four carbons act as an inaccessible coupled bath.",
)

heading("2.2 Input encoding and FID readout", level=2)
para(
    "A scalar input s ∈ [0,1] is encoded as a global rotation R_x(θ) with θ = arcsin(√s), "
    "applied to the designated nuclei; multivariate inputs are encoded on distinct nuclear "
    "species (e.g. temperature on protons, humidity on carbons). Between inputs the "
    "reservoir evolves for a fixed time τ under the Lindbladian, integrating the new input "
    "with the fading memory of past inputs. Readout follows Hou et al.: a π/2 pulse is "
    "applied to the proton spins and the free-induction-decay (FID) signal—the collective "
    "transverse magnetisation S(t) = Tr[e^{tL}(UρU†) O_FID], O_FID = Σ_{protons}(σy + iσx)"
    "—is sampled on a fine time grid, Fourier transformed, and the largest spectral peaks "
    "are selected as readout features (653, matching the paper). This time-multiplexed "
    "readout raises the number of independent readout functions from the few accessible "
    "single-time observables to several hundred, which we show is decisive (Section 3). We "
    "exploit an exact simplification: the chemical-shift terms of the non-readout (carbon) "
    "spins commute with the Heisenberg evolution of the proton transverse operators and are "
    "omitted from the readout Liouvillian, verified to agree with the full operator to "
    "≈10^-14 and reducing the exponential-integration cost several-fold."
)

heading("2.3 Evolution backends and GPU acceleration", level=2)
para(
    "The reservoir must evolve ρ and sample it over hundreds to thousands of input steps. "
    "We implement four interchangeable backends and cross-validate them to within 10^-3 on "
    "shared systems: (a) a dense Liouvillian superoperator propagator, exact and fast for "
    "N ≤ 5 but 4^N-dimensional (0.27 GB at N=6, 4.3 GB at N=7, ≈550 GB at N=9—infeasible); "
    "(b) an exact sparse Krylov “action” using scipy.expm_multiply on the "
    "≈ 0.001%-dense Liouvillian, which removes the memory wall and is immune to the "
    "stiffness (kHz precession vs Hz relaxation) that cripples an adaptive ODE integrator; "
    "(c) a GPU implementation applying exp(t·L) via a Taylor + sub-stepping scheme with "
    "sparse CUDA matrix–vector products, measured ≈28× faster than the CPU action path at "
    "N=9 (0.23 vs 6.5 s per matrix action); and (d) QuTiP’s adaptive ODE for reference. The "
    "GPU path makes the nine-qubit open-system regime routine (a full NARMA reproduction in "
    "≈ 4 h). We note that the JAX GPU route suggested for QuTiP is unavailable on Windows "
    "(no CUDA wheels); our GPU backend uses PyTorch CUDA instead."
)

heading("2.4 Tasks, metrics, and baselines", level=2)
para(
    "The linear readout is trained by ridge regression with k-fold cross-validated "
    "regularisation. We evaluate: (i) the memory-capacity test (squared correlation between "
    "input and reconstruction versus delay), the diagnostic of correct fading memory; "
    "(ii) NARMA orders 2–20 (normalised mean-squared error, NMSE = Σ(y−ŷ)²/Σy²), driven "
    "by the paper’s superposition-of-sines input, with multiple orders fit as separate "
    "readouts on a single reservoir pass (“multitasking”); and (iii) Delhi daily-climate "
    "forecasting (temperature and humidity, single reservoir pass with per-horizon readouts "
    "to 45 days). Classical baselines are leaky echo state networks (ESNs) of 500–10,000 "
    "nodes and, following the paper, radial-basis-function support-vector-regression "
    "post-processing of the quantum features (QRC+RBF)."
)

heading("2.5 Independent physics validation (SLEEPY)", level=2)
para(
    "To confirm that our Lindblad implementation reproduces genuine NMR dynamics—rather "
    "than merely being internally consistent—we cross-check against SLEEPY, an independent "
    "Liouville-space NMR simulation library (Nature Communications, 2025). For a two-proton "
    "subsystem with matched chemical shifts, J-coupling, and relaxation times, we compare "
    "the complex FID and its spectrum computed by our QuTiP-based engine and by SLEEPY."
)

# ===========================================================================
heading("3. Results", level=1)

heading("3.1 Fading memory and cross-validation", level=2)
para(
    "The memory-capacity test yields the required fading-memory signature: squared "
    "correlation ≈0.96 at zero delay decaying smoothly to zero within ≈12–15 steps "
    "(Figure 1). A single-time σz readout is degenerate for this Ising-coupled Hamiltonian "
    "(σz commutes with H and is frozen on the T1 timescale); reading the transverse "
    "components restores the behaviour. The four evolution backends agree to 3–4 significant "
    "figures, and the GPU path matches the exact CPU action to 3×10^-6."
)
figure("fig2_fading_memory.png",
       "Figure 1. Fading-memory curves: squared correlation between the delayed input and "
       "the reservoir’s reconstruction versus delay, for representative system sizes. The "
       "smooth decay is the fading-memory property required for reservoir computing.")

heading("3.2 NARMA reproduction", level=2)
para(
    "With the FID-653 readout the simulator reaches the paper’s high-accuracy regime across "
    "all NARMA orders (Table 2), with R² ≈ 0.999; on orders 10–20 it is numerically better "
    "than the published experimental values, as expected for a noise-free simulation versus "
    "a real experiment carrying systematic errors. Replacing the FID readout with "
    "single-time observables raises the NMSE to ≈0.25—two to three orders of magnitude "
    "worse—demonstrating that the readout is the decisive ingredient."
)
table(
    ["NARMA order", "This work (NMSE)", "Hou et al. (best)", "This work (R²)"],
    [["2", "5.19×10⁻⁶", "1.74×10⁻⁷", "0.9998"],
     ["5", "5.21×10⁻⁵", "4.44×10⁻⁵", "0.9997"],
     ["10", "2.46×10⁻⁵", "5.84×10⁻⁵", "0.9997"],
     ["15", "1.82×10⁻⁵", "6.37×10⁻⁵", "0.9995"],
     ["20", "3.24×10⁻⁶", "4.34×10⁻⁵", "0.9989"]],
    "Table 2. NARMA reproduction (normalised mean-squared error) versus Hou et al. Table I "
    "best cases. The simulator reaches the same 10⁻⁵–10⁻⁶ regime.",
)
para(
    "On NARMA, however, a classical ESN is far stronger: NMSE falls from 2×10^-7 (500 "
    "nodes) to 9×10^-9 (10,000 nodes), beating the quantum reservoir by two to three orders "
    "of magnitude. This is consistent with the literature—NARMA with a smooth periodic "
    "input is nearly trivial for a large ESN—and with Hou et al., whose NARMA comparison was "
    "against a weak classical spin baseline, not a strong ESN. The quantum advantage is "
    "claimed, and found, elsewhere."
)

heading("3.3 Weather forecasting and the quantum advantage", level=2)
para(
    "On Delhi daily-climate forecasting the picture inverts (Table 3, Figure 2). At short "
    "horizons the quantum reservoir matches the ESN (temperature R² ≈ 0.94 at one day). At "
    "long horizons the quantum reservoir—and decisively the QRC+RBF variant—outperforms "
    "ESNs of every tested size, with the margin widening as the horizon grows: at 45 days, "
    "temperature R² is 0.67 (QRC+RBF) versus 0.41 for the best ESN. Humidity, which is "
    "governed by faster local processes and is intrinsically harder, shows the same pattern "
    "(R² 0.50 vs 0.31 at 45 days). Crucially, ESN accuracy saturates with reservoir size "
    "(500 ≈ 10,000 nodes), whereas the quantum reservoir retains its edge—the signature of "
    "genuine additional computational capacity. This reproduces the central advantage claim "
    "of Hou et al."
)
table(
    ["Horizon (days)", "QRC", "QRC+RBF", "ESN-500", "ESN-1000", "ESN-5000", "ESN-10000"],
    [["1", "0.936", "0.938", "0.940", "0.940", "0.940", "0.940"],
     ["5", "0.826", "0.848", "0.852", "0.849", "0.850", "0.853"],
     ["10", "0.773", "0.817", "0.813", "0.815", "0.812", "0.812"],
     ["15", "0.768", "0.790", "0.762", "0.755", "0.753", "0.759"],
     ["20", "0.741", "0.756", "0.714", "0.690", "0.602", "0.698"],
     ["30", "0.745", "0.759", "0.553", "0.563", "0.550", "0.574"],
     ["45", "0.572", "0.674", "0.370", "0.396", "0.412", "0.413"]],
    "Table 3. Temperature-forecast R² versus horizon: quantum reservoir (QRC), quantum "
    "reservoir with RBF-SVR post-processing (QRC+RBF), and classical ESNs of 500–10,000 "
    "nodes. Bold in the accompanying analysis: QRC+RBF exceeds all ESNs for horizons ≥ 15.",
)
figure("fig4_weather.png",
       "Figure 2. Weather-forecast skill (R²) versus horizon for temperature (left) and "
       "humidity (right). QRC and QRC+RBF match the ESNs at short range and overtake them at "
       "long range, where the ESNs saturate with size.")

heading("3.4 Compute performance", level=2)
para(
    "The GPU backend makes the study practical: a nine-qubit FID reservoir pass over several "
    "hundred input steps completes in hours rather than the days a CPU action path would "
    "require, and the dense-superoperator method cannot reach N ≥ 8 at all. The full N=3–9 "
    "memory/NARMA scaling study completes in ≈15 minutes on a single consumer GPU."
)

# ===========================================================================
heading("4. Discussion", level=1)
para(
    "Two conclusions follow. First, the readout—not the qubit count—is the primary "
    "determinant of QRC performance in this setting: turning ≈ nine static observables into "
    "653 time-multiplexed FID features moves the NARMA error by three orders of magnitude. "
    "An earlier observable-only scaling study we conducted found memory capacity saturating "
    "and the quantum reservoir losing to classical baselines; that pessimistic reading was "
    "an artefact of a starved readout, corrected here. Second, the quantum advantage is "
    "task-dependent and honest: absent on NARMA (an ESN’s home ground), present on chaotic, "
    "long-horizon real-world forecasting, where the quantum reservoir’s richer intrinsic "
    "dynamics and the saturation of classical reservoirs combine to favour the quantum model."
)
para(
    "Because the reproduction is a noise-free simulation, it does not capture the "
    "cross-correlated relaxation and systematic errors of a physical device (which is why "
    "our NARMA errors can undercut the experiment). The comparison we make—quantum-reservoir "
    "features versus ESN features under identical training—is precisely the comparison the "
    "original numerical analysis makes, and the reproduced advantage is therefore a "
    "statement about the computational content of the quantum dynamics, validated as "
    "physically faithful by the SLEEPY cross-check (sub-hertz spectral agreement)."
)

# ===========================================================================
heading("5. Conclusion", level=1)
para(
    "We have built and released an open, GPU-accelerated simulator for NMR-based quantum "
    "reservoir computing and used it to reproduce Hou et al. (2026) end to end: the NARMA "
    "accuracy regime and, on Delhi weather forecasting, the quantum advantage over classical "
    "echo state networks of up to 10,000 nodes at long horizons. We isolate the "
    "time-multiplexed FID readout as the decisive mechanism, make the nine-qubit open-system "
    "regime computationally routine, and independently validate the physics against SLEEPY. "
    "The simulator and its honest, task-dependent verdict provide a foundation for designing "
    "and de-risking future NMR-QRC experiments before hardware time is committed."
)

heading("Data and Code Availability", level=1)
para(
    "All source code, the exact reservoir parameters, reproduction runners with durable "
    "progress logging, the Delhi climate dataset, and the raw result files are released in "
    "the project repository (app/qrc and docs/QRC). Every figure and table is regenerated "
    "by the included scripts from fixed random seeds.",
    size=10,
)

heading("References", level=1)
refs = [
    "Y. Hou, J. Hua, Z. Wu, W. Xia, Y. Chen, X. Li, Z. Li, X. Peng, J. Du. "
    "High-Accuracy Temporal Prediction via Experimental Quantum Reservoir Computing in "
    "Correlated Spins. Phys. Rev. Lett. 136, 120602 (2026).",
    "K. Nakajima, K. Fujii, M. Negoro, K. Mitarai, M. Kitagawa. Boosting Computational "
    "Power through Spatial Multiplexing in Quantum Reservoir Computing. Phys. Rev. Applied "
    "11, 034021 (2019); arXiv:1803.04574.",
    "M. Negoro, K. Mitarai, K. Fujii, K. Nakajima, M. Kitagawa. Machine learning with "
    "controllable quantum dynamics of a nuclear spin ensemble in a solid. arXiv:1806.10910 "
    "(2018).",
    "J. Dambre, D. Verstraeten, B. Schrauwen, S. Massar. Information Processing Capacity of "
    "Dynamical Systems. Sci. Rep. 2, 514 (2012).",
    "H. Jaeger, H. Haas. Harnessing Nonlinearity: Predicting Chaotic Systems and Saving "
    "Energy in Wireless Communication. Science 304, 78–80 (2004).",
    "A. F. Atiya, A. G. Parlos. New results on recurrent network training. IEEE Trans. "
    "Neural Netw. 11, 697 (2000).",
    "N. Lambert et al. QuTiP 5: The Quantum Toolbox in Python. arXiv:2412.04705 (2024).",
    "A. A. Smith-Penzel et al. SLEEPY: a comprehensive Python module for simulating "
    "relaxation and dynamics in nuclear magnetic resonance. Nat. Commun. (2025); "
    "doi:10.1038/s41467-025-65091-6.",
]
for i, r in enumerate(refs, 1):
    p = para(f"[{i}] {r}", size=9, space=3)

OUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(str(OUT))
print(f"saved {OUT.resolve()}  ({OUT.stat().st_size} bytes)")
