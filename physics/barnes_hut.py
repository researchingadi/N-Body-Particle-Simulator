"""Barnes-Hut tree approximation for gravitational force computation.

Direct summation (physics/forces.py) is O(N^2): every particle interacts
with every other particle, every step. That's fine for a few hundred bodies
but becomes the bottleneck long before you reach the particle counts a
"star cluster" or "galaxy" preset implies (thousands to tens of thousands).

Barnes-Hut (Barnes & Hut 1986) trades a small, controllable amount of
accuracy for much better scaling: particles are organized into an octree,
and a distant *group* of particles is approximated as a single point mass
at the group's center of mass, rather than summed individually. Whether a
group counts as "distant enough" is controlled by the opening angle
`theta`: a node of physical size `s` at distance `d` from the particle
being evaluated is approximated (not opened) when `s / d < theta`. Smaller
theta means more groups get opened (more accurate, closer to direct
summation, slower); larger theta approximates more aggressively (faster,
less accurate). theta -> 0 recovers direct summation; theta is typically
used in the range ~0.3-1.0 in practice.

This module is intentionally a clear, readable, pure-Python recursive
octree -- not yet a performance-optimized implementation (no vectorized
tree traversal, no Cython/GPU). That optimization is deferred until this
version is validated against the direct solver and profiled; per the
project's correctness-first ordering, an accurate slow tree is more
valuable right now than a fast one of unknown accuracy.

IMPORTANT: the direct O(N^2) solver in physics/forces.py remains the
ground-truth reference. Barnes-Hut is an approximation, not a more
accurate alternative -- see the validation script and README section for
the honest accuracy/performance tradeoff.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# If more than this many particles land in one leaf due to exhausting the
# subdivision depth (only happens for near-coincident positions), stop
# subdividing and treat the leaf as a small direct-summation bucket instead
# of recursing forever. This is a pragmatic fallback for degenerate inputs,
# not a normal code path for realistic particle distributions.
MAX_TREE_DEPTH = 32


@dataclass
class BHNode:
    """A cubic region of space in the octree.

    A node is either:
    - an internal node (`children` is a list of 8 slots, some possibly
      None) whose `indices` list is always empty (every particle has been
      pushed down into a child), or
    - a leaf (`children` is None) holding zero, one, or -- only in the
      degenerate near-coincident-position case -- more than one particle
      index directly in `indices`.

    `mass` and `com` (center of mass) are aggregate properties of every
    particle contained anywhere in this node's subtree; they are filled in
    by `_aggregate_mass_and_com` after the tree shape is built, not during
    insertion.
    """

    center: tuple[float, float, float]
    half_size: float  # half the side length of this node's cubic region
    depth: int
    children: list["BHNode | None"] | None = None
    indices: list[int] = field(default_factory=list)
    mass: float = 0.0
    com: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def is_leaf(self) -> bool:
        return self.children is None


def _bounding_cube(positions: np.ndarray) -> tuple[tuple[float, float, float], float]:
    """Smallest cube (center, half_size) containing all positions, with padding.

    Barnes-Hut requires a cubic (not just rectangular) root region so that
    "node size" in the opening-angle test is unambiguous. A small relative
    padding keeps particles strictly inside the cube even at floating-point
    boundary cases (e.g. a particle exactly at the computed max extent).
    """
    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    center = (mins + maxs) / 2.0
    extent = np.max(maxs - mins)
    if extent <= 0.0:
        extent = 1.0  # degenerate case: all particles coincident
    half_size = float(extent) * 0.5 * 1.001 + 1e-12  # small padding
    return (float(center[0]), float(center[1]), float(center[2])), half_size


def _octant_index(center: tuple[float, float, float], point: tuple[float, float, float]) -> int:
    """Which of the 8 octants `point` falls into, relative to `center`.

    Bit 0 = x >= cx, bit 1 = y >= cy, bit 2 = z >= cz.
    """
    cx, cy, cz = center
    px, py, pz = point
    return (px >= cx) | ((py >= cy) << 1) | ((pz >= cz) << 2)


def _child_center(
    parent_center: tuple[float, float, float], parent_half_size: float, octant: int
) -> tuple[float, float, float]:
    """Center of the child cube occupying the given octant of the parent."""
    offset = parent_half_size / 2.0
    cx, cy, cz = parent_center
    dx = offset if (octant & 1) else -offset
    dy = offset if (octant & 2) else -offset
    dz = offset if (octant & 4) else -offset
    return (cx + dx, cy + dy, cz + dz)


def _insert(node: BHNode, index: int, points: list[tuple[float, float, float]]) -> None:
    """Insert particle `index` into the subtree rooted at `node`."""
    if node.children is not None:
        # Internal node: descend into (creating if needed) the right child.
        octant = _octant_index(node.center, points[index])
        child = node.children[octant]
        if child is None:
            child_center = _child_center(node.center, node.half_size, octant)
            child = BHNode(center=child_center, half_size=node.half_size / 2.0, depth=node.depth + 1)
            node.children[octant] = child
        _insert(child, index, points)
        return

    # Leaf node.
    if len(node.indices) == 0:
        node.indices.append(index)
        return

    if node.depth >= MAX_TREE_DEPTH:
        # Degenerate case (near-coincident positions): give up subdividing
        # and let this leaf hold multiple particles directly.
        node.indices.append(index)
        return

    # Leaf currently holds exactly one particle and a new one has arrived:
    # subdivide into 8 children and push both particles down.
    existing_indices = node.indices
    node.indices = []
    node.children = [None] * 8
    for idx in (*existing_indices, index):
        _insert(node, idx, points)


def _aggregate_mass_and_com(
    node: BHNode, masses: np.ndarray, points: list[tuple[float, float, float]]
) -> tuple[float, tuple[float, float, float]]:
    """Post-order pass: fill in node.mass and node.com from the leaves up."""
    if node.children is None:
        if not node.indices:
            node.mass = 0.0
            node.com = node.center
            return node.mass, node.com

        total_mass = float(sum(masses[i] for i in node.indices))
        cx = sum(masses[i] * points[i][0] for i in node.indices) / total_mass
        cy = sum(masses[i] * points[i][1] for i in node.indices) / total_mass
        cz = sum(masses[i] * points[i][2] for i in node.indices) / total_mass
        node.mass = total_mass
        node.com = (cx, cy, cz)
        return node.mass, node.com

    total_mass = 0.0
    wx = wy = wz = 0.0
    for child in node.children:
        if child is None:
            continue
        m, (cx, cy, cz) = _aggregate_mass_and_com(child, masses, points)
        total_mass += m
        wx += m * cx
        wy += m * cy
        wz += m * cz

    node.mass = total_mass
    node.com = (wx / total_mass, wy / total_mass, wz / total_mass) if total_mass > 0 else node.center
    return node.mass, node.com


def build_octree(positions: np.ndarray, masses: np.ndarray) -> BHNode:
    """Build a Barnes-Hut octree over `positions`/`masses`.

    Exposed as a public function (rather than only an internal step of
    `barnes_hut_accelerations`) so tree structure, mass, and center-of-mass
    values can be checked directly in tests, independent of the force
    evaluation / opening-angle logic.
    """
    n = positions.shape[0]
    center, half_size = _bounding_cube(positions)
    root = BHNode(center=center, half_size=half_size, depth=0)

    points = [tuple(positions[i]) for i in range(n)]
    for i in range(n):
        _insert(root, i, points)

    _aggregate_mass_and_com(root, masses, points)
    return root


def _accel_on_particle(
    node: BHNode,
    i: int,
    point: tuple[float, float, float],
    masses: np.ndarray,
    points: list[tuple[float, float, float]],
    G: float,
    softening2: float,
    theta: float,
) -> tuple[float, float, float]:
    """Recursively accumulate acceleration on particle `i` from `node`'s subtree."""
    if node.mass == 0.0:
        return (0.0, 0.0, 0.0)

    if node.children is None:
        # Leaf: direct sum over whatever particles are actually here
        # (normally just one; see MAX_TREE_DEPTH fallback for the rare
        # multi-particle case). Explicit self-exclusion, matching the
        # direct solver's "no self-force" behavior.
        ax = ay = az = 0.0
        px, py, pz = point
        for j in node.indices:
            if j == i:
                continue
            jx, jy, jz = points[j]
            dx, dy, dz = jx - px, jy - py, jz - pz
            dist2 = dx * dx + dy * dy + dz * dz + softening2
            inv_dist3 = dist2 ** -1.5
            factor = G * masses[j] * inv_dist3
            ax += factor * dx
            ay += factor * dy
            az += factor * dz
        return (ax, ay, az)

    # Internal node: opening-angle test against this node's aggregate mass/COM.
    px, py, pz = point
    cx, cy, cz = node.com
    dx, dy, dz = cx - px, cy - py, cz - pz
    dist2 = dx * dx + dy * dy + dz * dz + softening2
    node_size = 2.0 * node.half_size

    # dist2 > softening2 guards the ratio test against a near-zero distance
    # (e.g. the evaluation point sitting almost exactly at this subtree's
    # COM); in that regime the node is always opened rather than risking a
    # division by a tiny number.
    if dist2 > softening2 and (node_size * node_size) < (theta * theta) * dist2:
        inv_dist3 = dist2 ** -1.5
        factor = G * node.mass * inv_dist3
        return (factor * dx, factor * dy, factor * dz)

    ax = ay = az = 0.0
    for child in node.children:
        if child is not None:
            cax, cay, caz = _accel_on_particle(child, i, point, masses, points, G, softening2, theta)
            ax += cax
            ay += cay
            az += caz
    return (ax, ay, az)


def barnes_hut_accelerations(
    positions: np.ndarray,
    masses: np.ndarray,
    G: float = 1.0,
    softening: float = 0.01,
    theta: float = 0.5,
) -> np.ndarray:
    """Approximate gravitational acceleration via a Barnes-Hut octree.

    Same contract as physics.forces.compute_accelerations: pass positions
    (N, 3) and masses (N,), get back accelerations (N, 3). This function is
    a drop-in replacement anywhere an `accel_fn(positions, masses)` closure
    is used (see simulation.engine.Simulation._accel_fn).

    Args:
        positions: (N, 3) array of positions.
        masses: (N,) array of masses.
        G: gravitational constant.
        softening: Plummer softening length epsilon.
        theta: opening angle. Smaller = more accurate & slower (theta -> 0
            approaches direct summation); larger = faster & less accurate.

    Returns:
        (N, 3) array of accelerations, one per body.

    Complexity: O(N log N) typical case (tree build + tree walk per
    particle), vs. O(N^2) for direct summation -- the whole point of this
    module. This is an approximation; use physics.forces.compute_accelerations
    as ground truth when accuracy matters more than speed.
    """
    n = positions.shape[0]
    if n == 0:
        return np.zeros((0, 3))
    if n == 1:
        return np.zeros((1, 3))

    root = build_octree(positions, masses)
    points = [tuple(positions[i]) for i in range(n)]
    softening2 = softening**2

    accelerations = np.zeros((n, 3))
    for i in range(n):
        ax, ay, az = _accel_on_particle(root, i, points[i], masses, points, G, softening2, theta)
        accelerations[i, 0] = ax
        accelerations[i, 1] = ay
        accelerations[i, 2] = az
    return accelerations
