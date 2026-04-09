"""
Particle in a Box — Quantum Mechanics Simulator
Computational Chemistry Project
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch

# ── Constants ──────────────────────────────────────────────────────────────────
HBAR  = 1.0545718e-34   # J·s
ME    = 9.10938e-31     # kg  (electron mass)
EV    = 1.60218e-19     # J per eV
NM    = 1e-9            # metres per nm

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Particle in a Box Simulator",
    page_icon="⚛️",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title  { font-size: 2rem; font-weight: 600; color: #1a1a2e; margin-bottom: 0; }
    .sub-title   { font-size: 1rem; color: #555; margin-top: 0; margin-bottom: 1.5rem; }
    .metric-box  { background:#f0f4ff; border-radius:10px; padding:14px 18px; text-align:center; }
    .metric-val  { font-size:1.6rem; font-weight:700; color:#3a5bd9; }
    .metric-lbl  { font-size:0.8rem; color:#666; margin-top:2px; }
    .formula-box { background:#fff8e7; border-left:4px solid #f4c430; border-radius:6px;
                   padding:12px 16px; font-family:monospace; font-size:0.95rem; }
    .info-box    { background:#e8f4fd; border-left:4px solid #4da6e8; border-radius:6px;
                   padding:10px 16px; font-size:0.88rem; color:#1a4a6b; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Physics functions ──────────────────────────────────────────────────────────

def energy_eV(n: int, L_nm: float, mass_factor: float) -> float:
    """Return energy of level n in eV."""
    L = L_nm * NM
    m = mass_factor * ME
    E = (n**2 * np.pi**2 * HBAR**2) / (2 * m * L**2)
    return E / EV

def wavefunction(n: int, x: np.ndarray, L_nm: float) -> np.ndarray:
    """Normalised wavefunction ψ_n(x)."""
    L = L_nm * NM
    x_m = x * NM
    return np.sqrt(2 / L) * np.sin(n * np.pi * x_m / L)

def probability_density(psi: np.ndarray) -> np.ndarray:
    """Return |ψ|²."""
    return psi**2

def expectation_x(n: int, L_nm: float) -> float:
    """⟨x⟩ = L/2 for all n (analytical result)."""
    return L_nm / 2

def expectation_x2(n: int, L_nm: float) -> float:
    """⟨x²⟩ analytical."""
    return L_nm**2 * (1/3 - 1 / (2 * n**2 * np.pi**2))

def uncertainty_x(n: int, L_nm: float) -> float:
    """Δx = sqrt(⟨x²⟩ - ⟨x⟩²)."""
    return np.sqrt(expectation_x2(n, L_nm) - expectation_x(n, L_nm)**2)

def momentum_uncertainty(n: int, L_nm: float, mass_factor: float) -> float:
    """Δp = nπℏ/L."""
    return n * np.pi * HBAR / (L_nm * NM)

# ── Plot helpers ───────────────────────────────────────────────────────────────

PALETTE = ["#3a5bd9", "#e05252", "#27ae60", "#e67e22", "#8e44ad", "#16a085"]

def plot_wavefunction(ax, n_levels, L_nm, mass_factor, selected_n, show_prob):
    ax.set_facecolor("#fafafa")
    x = np.linspace(0, L_nm, 500)

    for n in n_levels:
        psi = wavefunction(n, x, L_nm)
        col = PALETTE[(n - 1) % len(PALETTE)]
        alpha = 1.0 if n == selected_n else 0.28
        lw    = 2.5 if n == selected_n else 1.2

        if show_prob:
            y = probability_density(psi) * (L_nm / 6)   # scale for display
            ax.fill_between(x, y, alpha=alpha * 0.25, color=col)
            ax.plot(x, y, color=col, lw=lw, label=f"n={n}  |ψ|²")
        else:
            ax.fill_between(x, psi, alpha=alpha * 0.15, color=col)
            ax.plot(x, psi, color=col, lw=lw, label=f"n={n}  ψ(x)")

    ax.axhline(0, color="#999", lw=0.8, ls="--")
    ax.axvline(0,   color="#333", lw=2)
    ax.axvline(L_nm, color="#333", lw=2)

    ax.set_xlabel("Position x (nm)", fontsize=11)
    ylabel = "|ψ(x)|²  (scaled)" if show_prob else "ψ(x)  (nm⁻¹/²)"
    ax.set_ylabel(ylabel, fontsize=11)
    title  = "Probability Density  |ψₙ(x)|²" if show_prob else "Wavefunction  ψₙ(x)"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlim(-0.05 * L_nm, 1.05 * L_nm)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)


def plot_energy_diagram(ax, n_max, L_nm, mass_factor, selected_n):
    ax.set_facecolor("#fafafa")
    energies = [energy_eV(n, L_nm, mass_factor) for n in range(1, n_max + 1)]
    E1 = energies[0]

    for n, E in enumerate(energies, 1):
        col   = PALETTE[(n - 1) % len(PALETTE)]
        alpha = 1.0 if n == selected_n else 0.45
        lw    = 2.5 if n == selected_n else 1.5

        ax.hlines(E, 0.2, 0.8, colors=col, lw=lw, alpha=alpha)
        ax.text(0.83, E, f"n={n}   {E:.3f} eV   ({E/E1:.0f}E₁)",
                va="center", fontsize=8.5, color=col, alpha=max(alpha, 0.6))

        if n == selected_n:
            ax.hlines(E, 0.2, 0.8, colors=col, lw=4, alpha=0.2)

    ax.set_xlim(0, 1.6)
    ax.set_ylim(0, energies[-1] * 1.18)
    ax.set_ylabel("Energy (eV)", fontsize=11)
    ax.set_title("Energy Levels", fontsize=13, fontweight="bold")
    ax.set_xticks([])
    ax.spines[["top","right","bottom"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)


def plot_3d_surface(ax3d, selected_n, L_nm):
    x = np.linspace(0, L_nm, 120)
    t = np.linspace(0, 2 * np.pi, 120)
    X, T = np.meshgrid(x, t)
    psi  = wavefunction(selected_n, X, L_nm)
    Z    = psi * np.cos(T)

    col = PALETTE[(selected_n - 1) % len(PALETTE)]
    ax3d.plot_surface(X, T, Z, cmap="coolwarm", alpha=0.8, linewidth=0)
    ax3d.set_xlabel("x (nm)", fontsize=8)
    ax3d.set_ylabel("Time phase", fontsize=8)
    ax3d.set_zlabel("ψ", fontsize=8)
    ax3d.set_title(f"Time evolution  n={selected_n}", fontsize=11, fontweight="bold")
    ax3d.tick_params(labelsize=7)

# ── Streamlit UI ───────────────────────────────────────────────────────────────

st.markdown('<p class="main-title">⚛️ Particle in a Box — Quantum Simulator</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Computational Chemistry Project &nbsp;|&nbsp; 1D Infinite Square Well</p>', unsafe_allow_html=True)

# ── Sidebar controls ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🎛️ Parameters")

    selected_n = st.slider("Quantum number  n", 1, 8, 1)
    L_nm       = st.slider("Box length  L (nm)", 0.1, 3.0, 1.0, 0.05)
    mass_fac   = st.slider("Particle mass  (× mₑ)", 0.1, 5.0, 1.0, 0.1)
    n_show     = st.slider("Show levels  1 → n", 1, 8, selected_n)

    st.markdown("---")
    st.subheader("📊 Display")
    show_prob  = st.checkbox("Show |ψ|² (probability density)", False)
    show_3d    = st.checkbox("Show time evolution (3D)", False)

    st.markdown("---")
    st.subheader("ℹ️ About")
    st.markdown("""
    The **infinite square well** (particle in a box) is one of the simplest
    exactly-solvable quantum mechanics problems.  
    - Wavefunction: `ψₙ(x) = √(2/L) sin(nπx/L)`  
    - Energy: `Eₙ = n²π²ℏ²/(2mL²)`
    """)

# ── Metrics row ────────────────────────────────────────────────────────────────
E   = energy_eV(selected_n, L_nm, mass_fac)
E1  = energy_eV(1,          L_nm, mass_fac)
dx  = uncertainty_x(selected_n, L_nm)
dp  = momentum_uncertainty(selected_n, L_nm, mass_fac)
hup = dx * dp / HBAR   # should be ≥ 0.5

c1, c2, c3, c4, c5 = st.columns(5)

def metric_card(col, value, label):
    col.markdown(
        f'<div class="metric-box"><div class="metric-val">{value}</div>'
        f'<div class="metric-lbl">{label}</div></div>',
        unsafe_allow_html=True,
    )

metric_card(c1, f"{E:.4f} eV",      f"Energy  E_{selected_n}")
metric_card(c2, f"{E/E1:.0f} × E₁", "Energy ratio")
metric_card(c3, f"{selected_n-1}",   "Nodes  (n − 1)")
metric_card(c4, f"{dx:.3f} nm",      "Δx  (position uncert.)")
metric_card(c5, f"{hup:.3f} ℏ",     "ΔxΔp / ℏ  (≥ 0.5)")

st.markdown("<br>", unsafe_allow_html=True)

# ── Plots ──────────────────────────────────────────────────────────────────────
n_levels = list(range(1, max(n_show, selected_n) + 1))

if show_3d:
    fig = plt.figure(figsize=(14, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2], projection="3d")
    plot_wavefunction(ax1, n_levels, L_nm, mass_fac, selected_n, show_prob)
    plot_energy_diagram(ax2, max(n_levels), L_nm, mass_fac, selected_n)
    plot_3d_surface(ax3, selected_n, L_nm)
else:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    plt.subplots_adjust(wspace=0.35)
    plot_wavefunction(ax1, n_levels, L_nm, mass_fac, selected_n, show_prob)
    plot_energy_diagram(ax2, max(n_levels), L_nm, mass_fac, selected_n)

st.pyplot(fig, use_container_width=True)
plt.close(fig)

# ── Formula & explanation ──────────────────────────────────────────────────────
st.markdown("---")
col_f, col_i = st.columns([1, 1])

with col_f:
    st.markdown("**Key equations**")
    st.markdown(f"""
<div class="formula-box">
<b>Wavefunction:</b><br>
  ψₙ(x) = √(2/L) · sin(nπx/L)<br><br>
<b>Energy levels:</b><br>
  Eₙ = n²π²ℏ² / (2mL²) = n² · E₁<br><br>
<b>Ground state energy  E₁:</b><br>
  E₁ = {E1:.6f} eV  (L={L_nm} nm, m={mass_fac}mₑ)<br><br>
<b>Selected level  E_{selected_n}:</b><br>
  E_{selected_n} = {selected_n}² × E₁ = {E:.6f} eV
</div>
""", unsafe_allow_html=True)

with col_i:
    st.markdown("**Physical interpretation**")
    st.markdown(f"""
<div class="info-box">
<b>n = {selected_n}</b> &nbsp;→&nbsp; {selected_n - 1} node{"s" if selected_n > 2 else ""} in the wavefunction.<br><br>
The particle is <b>never</b> at the walls (ψ = 0 at x = 0, L).<br><br>
Heisenberg uncertainty:  ΔxΔp = <b>{hup:.3f} ℏ</b>  
{"✅ satisfies" if hup >= 0.5 else "⚠️"} the ≥ ℏ/2 bound.<br><br>
Increasing <b>L</b> lowers all energy levels (∝ 1/L²).<br>
Increasing <b>n</b> raises energy as n².
</div>
""", unsafe_allow_html=True)

# ── Data table ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("**Energy level table**")

import pandas as pd
rows = []
for n in range(1, 9):
    rows.append({
        "n": n,
        "Energy (eV)": round(energy_eV(n, L_nm, mass_fac), 6),
        "Ratio (Eₙ/E₁)": n**2,
        "Nodes": n - 1,
        "Δx (nm)": round(uncertainty_x(n, L_nm), 4),
        "ΔxΔp / ℏ": round(uncertainty_x(n, L_nm) * momentum_uncertainty(n, L_nm, mass_fac) / HBAR, 4),
    })

df = pd.DataFrame(rows)
df["Selected"] = df["n"] == selected_n
st.dataframe(
    df.style.apply(
        lambda row: ["background-color: #dce8ff; font-weight:bold"] * len(row)
                    if row["Selected"] else [""] * len(row),
        axis=1,
    ),
    use_container_width=True,
    hide_index=True,
)

st.caption("Particle in a Box Simulator · Computational Chemistry · Built with Python, Streamlit & Matplotlib")
