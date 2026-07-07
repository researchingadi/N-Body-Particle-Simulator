"""Direct O(N^2) Newtonian gravitational force computation.

Uses Plummer softening to avoid singular forces at small separations:

    a_i = G * sum_{j != i} m_j * (r_j - r_i) / (|r_j - r_i|^2 + eps^2)^(3/2)

This is the reference / correctness-baseline solver. It is intentionally
simple: no tree, no GPU, no shortcuts. All other solvers (Barnes-Hut, GPU)
must agree with this one on small test systems before they are trusted.
"""
from __future__ import annotations

import numpy as np

G_DEFAULT: float = 1.0  # simulation units unless overridden by caller


def pairwise_separations(positions: np.ndarray) -> np.ndarray:
    """Return r_j - r_i for all pairs, shape (N, N, 3).

    diff[i, j] = positions[j] - positions[i]
    """
    return positions[np.newaxis, :, :] - positions[:, np.newaxis, :]


def compute_accelerations(
    positions: np.ndarray,
    masses: np.ndarray,
    softening: float,
    G: float = G_DEFAULT,
) -> np.ndarray:
    """Compute gravitational acceleration on every body via direct summation.

    Args:
        positions: (N, 3) array of positions.
        masses: (N,) array of masses.
        softening: Plummer softening length epsilon (> 0 recommended).
        G: gravitational constant in simulation units.

    Returns:
        (N, 3) array of accelerations, one per body.

    Complexity: O(N^2). This is the ground-truth solver used to validate
    Barnes-Hut and GPU implementations.
    """
    n = positions.shape[0]
    if n == 0:
        return np.zeros((0, 3))

    diff = pairwise_separations(positions)  # (N, N, 3), diff[i,j] = r_j - r_i
    dist2 = np.sum(diff * diff, axis=-1) + softening**2  # (N, N)
    np.fill_diagonal(dist2, 1.0)  # avoid div-by-zero on self term, masked out below

    inv_dist3 = dist2 ** (-1.5)
    # zero out self-interaction explicitly (no self-force)
    np.fill_diagonal(inv_dist3, 0.0)

    # a_i = G * sum_j m_j * diff[i,j] * inv_dist3[i,j]
    weighted = inv_dist3[:, :, np.newaxis] * diff  # (N, N, 3)
    accelerations = G * np.einsum("ijk,j->ik", weighted, masses)
    return accelerations


def compute_potential_energy(
    positions: np.ndarray,
    masses: np.ndarray,
    softening: float,
    G: float = G_DEFAULT,
) -> float:
    """Total gravitational potential energy of the system.

    U = -G * sum_{i < j} m_i * m_j / sqrt(|r_i - r_j|^2 + eps^2)
    """
    n = positions.shape[0]
    if n < 2:
        return 0.0
    diff = pairwise_separations(positions)
    dist = np.sqrt(np.sum(diff * diff, axis=-1) + softening**2)
    iu = np.triu_indices(n, k=1)
    pair_terms = masses[iu[0]] * masses[iu[1]] / dist[iu]
    return float(-G * np.sum(pair_terms))
    n = positions.shape[0]
    if n < 2:
        return 0.0
    diff = pairwise_separations(positions)
    dist = np.sqrt(np.sum(diff * diff, axis=-1) + softening**2)
    iu = np.triu_indices(n, k=1)
    pair_terms = masses[iu[0]] * masses[iu[1]] / dist[iu]
    return float(-G * np.sum(pair_terms))
    n = positions.shape[0]
    if n < 2:
        return 0.0
    diff = pairwise_separations(positions)
    dist = np.sqrt(np.sum(diff * diff, axis=-1) + softening**2)
    iu = np.triu_indices(n, k=1)
    pair_terms = masses[iu[0]] * masses[iu[1]] / dist[iu]
    return float(-G * np.sum(pair_terms))
