"""Physical diagnostics used to validate simulation correctness.

These are the quantities that MUST be (approximately) conserved by a correct
symplectic integrator on an isolated system. Any solver or integrator change
should be checked against these before being trusted.
"""
from __future__ import annotations

import numpy as np

from physics.forces import compute_potential_energy


def kinetic_energy(masses: np.ndarray, velocities: np.ndarray) -> float:
    """KE = sum_i 0.5 * m_i * |v_i|^2"""
    speed2 = np.sum(velocities * velocities, axis=-1)
    return float(0.5 * np.sum(masses * speed2))


def potential_energy(
    positions: np.ndarray, masses: np.ndarray, softening: float, G: float = 1.0
) -> float:
    """Total gravitational potential energy (see physics.forces)."""
    return compute_potential_energy(positions, masses, softening, G)


def total_energy(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    softening: float,
    G: float = 1.0,
) -> float:
    """E = KE + U. Should stay ~constant for a correct symplectic run."""
    return kinetic_energy(masses, velocities) + potential_energy(
        positions, masses, softening, G
    )


def total_momentum(masses: np.ndarray, velocities: np.ndarray) -> np.ndarray:
    """p = sum_i m_i * v_i. Exactly conserved (no external forces)."""
    return np.sum(masses[:, np.newaxis] * velocities, axis=0)


def total_angular_momentum(
    positions: np.ndarray, masses: np.ndarray, velocities: np.ndarray
) -> np.ndarray:
    """L = sum_i m_i * (r_i x v_i). Exactly conserved (central forces only)."""
    momenta = masses[:, np.newaxis] * velocities
    return np.sum(np.cross(positions, momenta), axis=0)


def center_of_mass(positions: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """R_com = sum_i m_i r_i / sum_i m_i"""
    total_mass = np.sum(masses)
    return np.sum(masses[:, np.newaxis] * positions, axis=0) / total_mass


def center_of_mass_velocity(velocities: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """V_com = sum_i m_i v_i / sum_i m_i. Should be constant in time."""
    total_mass = np.sum(masses)
    return np.sum(masses[:, np.newaxis] * velocities, axis=0) / total_mass


def relative_energy_drift(energies: np.ndarray) -> float:
    """(E(t) - E(0)) / |E(0)|, evaluated at the final recorded step.

    The standard scalar figure-of-merit for "did my integrator misbehave".
    """
    e0 = energies[0]
    if e0 == 0:
        return float(energies[-1])
    return float((energies[-1] - e0) / abs(e0))
