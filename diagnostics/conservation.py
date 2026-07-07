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
    Note this captures secular (monotonic) drift at the end of the run, but
    can mask oscillatory error -- see `max_relative_energy_drift` for that.
    """
    e0 = energies[0]
    if e0 == 0:
        return float(energies[-1])
    return float((energies[-1] - e0) / abs(e0))


def max_relative_energy_drift(energies: np.ndarray) -> float:
    """max_t |E(t) - E(0)| / |E(0)|, over the whole recorded run.

    A symplectic integrator like leapfrog does NOT conserve energy exactly;
    it conserves a nearby "shadow" Hamiltonian, so the true energy oscillates
    within a bounded envelope rather than drifting to zero or diverging. This
    metric captures that envelope, which `relative_energy_drift` (a single
    final-step value) can miss if the run happens to sample near a trough or
    peak of the oscillation. Use both: a small envelope with a small final
    value is the good/expected outcome for leapfrog; a small final value
    riding on top of a large envelope would still be a red flag.
    """
    e0 = energies[0]
    if e0 == 0:
        return float(np.max(np.abs(energies)))
    return float(np.max(np.abs(energies - e0)) / abs(e0))


def relative_vector_drift(vectors: np.ndarray) -> float:
    """Relative drift of a conserved vector quantity (e.g. angular momentum).

    Defined as |v(t_final) - v(0)| / |v(0)|. Falls back to the absolute
    displacement when the initial vector is ~0 (e.g. a system with zero net
    angular momentum by construction), since a relative measure is undefined
    there.
    """
    v0 = vectors[0]
    v_final = vectors[-1]
    norm0 = float(np.linalg.norm(v0))
    diff = float(np.linalg.norm(v_final - v0))
    if norm0 < 1e-12:
        return diff
    return diff / norm0


def center_of_mass_drift(com_history: np.ndarray) -> float:
    """Maximum displacement of the center of mass from its initial position.

    For the symmetric benchmark systems used in this project, initial COM
    velocity is constructed to be ~0, so a well-behaved run should keep the
    COM essentially pinned at its starting point. This is a diagnostic of
    accumulated numerical asymmetry, not a "conserved quantity" in the usual
    sense (a system with genuine nonzero COM velocity would legitimately
    show linear COM drift, which is physical, not an error).
    """
    displacement = com_history - com_history[0]
    return float(np.max(np.linalg.norm(displacement, axis=1)))

        return float(energies[-1])
    return float((energies[-1] - e0) / abs(e0))


def max_relative_energy_drift(energies: np.ndarray) -> float:
    """max_t |E(t) - E(0)| / |E(0)|, over the whole recorded run.

    A symplectic integrator like leapfrog does NOT conserve energy exactly;
    it conserves a nearby "shadow" Hamiltonian, so the true energy oscillates
    within a bounded envelope rather than drifting to zero or diverging. This
    metric captures that envelope, which `relative_energy_drift` (a single
    final-step value) can miss if the run happens to sample near a trough or
    peak of the oscillation. Use both: a small envelope with a small final
    value is the good/expected outcome for leapfrog; a small final value
    riding on top of a large envelope would still be a red flag.
    """
    e0 = energies[0]
    if e0 == 0:
        return float(np.max(np.abs(energies)))
    return float(np.max(np.abs(energies - e0)) / abs(e0))


def relative_vector_drift(vectors: np.ndarray) -> float:
    """Relative drift of a conserved vector quantity (e.g. angular momentum).

    Defined as |v(t_final) - v(0)| / |v(0)|. Falls back to the absolute
    displacement when the initial vector is ~0 (e.g. a system with zero net
    angular momentum by construction), since a relative measure is undefined
    there.
    """
    v0 = vectors[0]
    v_final = vectors[-1]
    norm0 = float(np.linalg.norm(v0))
    diff = float(np.linalg.norm(v_final - v0))
    if norm0 < 1e-12:
        return diff
    return diff / norm0


def center_of_mass_drift(com_history: np.ndarray) -> float:
    """Maximum displacement of the center of mass from its initial position.

    For the symmetric benchmark systems used in this project, initial COM
    velocity is constructed to be ~0, so a well-behaved run should keep the
    COM essentially pinned at its starting point. This is a diagnostic of
    accumulated numerical asymmetry, not a "conserved quantity" in the usual
    sense (a system with genuine nonzero COM velocity would legitimately
    show linear COM drift, which is physical, not an error).
    """
    displacement = com_history - com_history[0]
    return float(np.max(np.linalg.norm(displacement, axis=1)))

