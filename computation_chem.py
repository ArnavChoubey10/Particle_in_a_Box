"""
physics.py — Particle in a Box: core quantum mechanics calculations.

All functions are pure (no side-effects) and return SI or eV values as noted.
"""

import numpy as np

# ── Physical constants ─────────────────────────────────────────────────────────
HBAR = 1.0545718e-34   # Reduced Planck constant  (J·s)
ME   = 9.10938e-31     # Electron rest mass        (kg)
EV   = 1.60218e-19     # Electronvolt to joules    (J/eV)
NM   = 1e-9            # Nanometre to metres       (m/nm)


# ── Energy ─────────────────────────────────────────────────────────────────────

def energy_joules(n: int, L_nm: float, mass_me: float = 1.0) -> float:
    """
    Energy of quantum level n in joules.

    Parameters
    ----------
    n        : principal quantum number (1, 2, 3, ...)
    L_nm     : box length in nanometres
    mass_me  : particle mass as a multiple of the electron mass (default 1)
    """
    L = L_nm * NM
    m = mass_me * ME
    return (n**2 * np.pi**2 * HBAR**2) / (2 * m * L**2)


def energy_eV(n: int, L_nm: float, mass_me: float = 1.0) -> float:
    """Energy of quantum level n in electronvolts."""
    return energy_joules(n, L_nm, mass_me) / EV


def energy_levels(n_max: int, L_nm: float, mass_me: float = 1.0) -> np.ndarray:
    """Return array of energies (eV) for n = 1 … n_max."""
    return np.array([energy_eV(n, L_nm, mass_me) for n in range(1, n_max + 1)])


# ── Wavefunctions ──────────────────────────────────────────────────────────────

def wavefunction(n: int, x_nm: np.ndarray, L_nm: float) -> np.ndarray:
    """
    Normalised wavefunction  ψ_n(x) = √(2/L) · sin(nπx/L).

    Parameters
    ----------
    n     : quantum number
    x_nm  : position array in nanometres (0 ≤ x ≤ L_nm)
    L_nm  : box length in nanometres

    Returns
    -------
    ψ_n(x) in nm^{-1/2}
    """
    L_m = L_nm * NM
    x_m = x_nm * NM
    norm = np.sqrt(2.0 / L_m)
    return norm * np.sin(n * np.pi * x_m / L_m)


def probability_density(n: int, x_nm: np.ndarray, L_nm: float) -> np.ndarray:
    """Return |ψ_n(x)|² in nm⁻¹."""
    psi = wavefunction(n, x_nm, L_nm)
    return psi**2


# ── Expectation values ─────────────────────────────────────────────────────────

def expectation_x(L_nm: float) -> float:
    """⟨x⟩ = L/2  (independent of n)."""
    return L_nm / 2.0


def expectation_x2(n: int, L_nm: float) -> float:
    """
    ⟨x²⟩ = L²(1/3 − 1/(2n²π²))  — analytical result.
    """
    return L_nm**2 * (1.0 / 3.0 - 1.0 / (2.0 * n**2 * np.pi**2))


def uncertainty_x(n: int, L_nm: float) -> float:
    """Position uncertainty  Δx = √(⟨x²⟩ − ⟨x⟩²)  in nm."""
    return np.sqrt(expectation_x2(n, L_nm) - expectation_x(L_nm)**2)


def expectation_p2(n: int, L_nm: float, mass_me: float = 1.0) -> float:
    """
    ⟨p²⟩ = 2mEₙ  (from the Hamiltonian).  Returns value in kg²·m²/s².
    """
    return 2.0 * mass_me * ME * energy_joules(n, L_nm, mass_me)


def uncertainty_p(n: int, L_nm: float) -> float:
    """
    Momentum uncertainty  Δp = nπℏ/L  in kg·m/s.
    """
    return n * np.pi * HBAR / (L_nm * NM)


def heisenberg_product(n: int, L_nm: float) -> float:
    """
    ΔxΔp / ℏ — should always be ≥ 0.5 (Heisenberg uncertainty principle).
    """
    return uncertainty_x(n, L_nm) * NM * uncertainty_p(n, L_nm) / HBAR
    # Note: uncertainty_x is in nm, convert to m first


def heisenberg_product_v2(n: int, L_nm: float) -> float:
    """Correct ΔxΔp/ℏ using SI units throughout."""
    dx = uncertainty_x(n, L_nm) * NM          # metres
    dp = uncertainty_p(n, L_nm)                # kg·m/s
    return dx * dp / HBAR


# ── Time evolution ─────────────────────────────────────────────────────────────

def time_evolved_wavefunction(n: int, x_nm: np.ndarray, L_nm: float,
                               t: float, mass_me: float = 1.0) -> np.ndarray:
    """
    Real part of the time-dependent wavefunction  ψ(x,t) = ψ_n(x)·e^{-iEt/ℏ}.

    Parameters
    ----------
    t : time in femtoseconds (1 fs = 1e-15 s)
    """
    t_s  = t * 1e-15
    E    = energy_joules(n, L_nm, mass_me)
    psi  = wavefunction(n, x_nm, L_nm)
    phase = np.exp(-1j * E * t_s / HBAR)
    return np.real(psi * phase)


# ── Superposition state ────────────────────────────────────────────────────────

def superposition(coeffs: dict, x_nm: np.ndarray, L_nm: float) -> np.ndarray:
    """
    Build a superposition  Ψ = Σ cₙ ψₙ  and normalise.

    Parameters
    ----------
    coeffs : {n: weight}  e.g. {1: 1, 2: 1} for equal mix of n=1 and n=2
    """
    psi_total = np.zeros_like(x_nm, dtype=float)
    for n, c in coeffs.items():
        psi_total += c * wavefunction(n, x_nm, L_nm)
    # Normalise numerically
    norm = np.trapz(psi_total**2, x_nm * NM)
    return psi_total / np.sqrt(norm)
