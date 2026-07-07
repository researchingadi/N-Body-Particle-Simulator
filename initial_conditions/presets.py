"""Standard test-system generators.

Every function returns (positions, velocities, masses) as
(N,3), (N,3), (N,) numpy arrays, using simulation units where G=1 unless
noted otherwise. All stochastic presets take an explicit `seed` for
reproducibility.
"""
from __future__ import annotations

import numpy as np


def binary_orbit(
    mass1: float = 1.0,
    mass2: float = 1.0,
    separation: float = 1.0,
    G: float = 1.0,
    eccentricity: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two-body circular (or eccentric) orbit in the xy-plane, COM at origin.

    Circular orbit speed derived from v_rel = sqrt(G*(m1+m2)/r) split
    between the two bodies inversely to mass so the COM stays fixed.
    """
    total_mass = mass1 + mass2
    r1 = separation * mass2 / total_mass
    r2 = separation * mass1 / total_mass

    positions = np.array([[-r1, 0.0, 0.0], [r2, 0.0, 0.0]])

    v_rel = np.sqrt(G * total_mass / separation) * (1.0 + eccentricity) ** 0.5
    v1 = v_rel * mass2 / total_mass
    v2 = v_rel * mass1 / total_mass
    velocities = np.array([[0.0, -v1, 0.0], [0.0, v2, 0.0]])

    masses = np.array([mass1, mass2])
    return positions, velocities, masses


def three_body_figure_eight(G: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The classic Chenciner-Montgomery figure-eight three-body solution.

    Equal masses, periodic, chaotic-adjacent -- a great stress test for
    integrator accuracy since the orbit is famously sensitive to error.
    """
    masses = np.array([1.0, 1.0, 1.0])
    positions = np.array(
        [
            [0.97000436, -0.24308753, 0.0],
            [-0.97000436, 0.24308753, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    v3 = np.array([-0.93240737, -0.86473146, 0.0])
    velocities = np.array(
        [
            [-0.5 * v3[0], -0.5 * v3[1], 0.0],
            [-0.5 * v3[0], -0.5 * v3[1], 0.0],
            [v3[0], v3[1], 0.0],
        ]
    )
    return positions, velocities, masses


def solar_system_like(G: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A toy Sun + 4 planets system with circular orbits at increasing radii.

    Not real astronomical data -- illustrative, hierarchical mass ratios and
    orbital spacing meant to look/behave "solar-system-like".
    """
    sun_mass = 1000.0
    planet_masses = np.array([1.0, 3.0, 2.0, 5.0])
    radii = np.array([2.0, 3.5, 5.0, 7.5])

    positions = [np.zeros(3)]
    velocities = [np.zeros(3)]
    masses = [sun_mass]

    rng_angle = np.linspace(0, 2 * np.pi, len(radii), endpoint=False)
    for m, r, theta in zip(planet_masses, radii, rng_angle):
        pos = np.array([r * np.cos(theta), r * np.sin(theta), 0.0])
        speed = np.sqrt(G * sun_mass / r)
        vel_dir = np.array([-np.sin(theta), np.cos(theta), 0.0])
        positions.append(pos)
        velocities.append(speed * vel_dir)
        masses.append(m)

    return np.array(positions), np.array(velocities), np.array(masses)


def plummer_sphere(
    n: int = 200,
    total_mass: float = 100.0,
    scale_radius: float = 1.0,
    G: float = 1.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Plummer (1911) model: a common self-gravitating star-cluster profile.

    Density: rho(r) = (3M / 4*pi*a^3) * (1 + r^2/a^2)^(-5/2)
    Positions sampled via inverse-CDF of the enclosed mass fraction.
    Velocities sampled from the isotropic Plummer distribution function
    using rejection sampling on the local escape-velocity fraction.
    """
    rng = np.random.default_rng(seed)
    masses = np.full(n, total_mass / n)

    # --- positions: inverse-CDF sampling of Plummer radius ---
    mass_frac = rng.uniform(0.0, 1.0, size=n)
    radii = scale_radius * (mass_frac ** (-2.0 / 3.0) - 1.0) ** -0.5

    costheta = rng.uniform(-1.0, 1.0, size=n)
    theta = np.arccos(costheta)
    phi = rng.uniform(0.0, 2 * np.pi, size=n)

    x = radii * np.sin(theta) * np.cos(phi)
    y = radii * np.sin(theta) * np.sin(phi)
    z = radii * np.cos(theta)
    positions = np.stack([x, y, z], axis=1)

    # --- velocities: rejection sampling from Plummer DF ---
    escape_speed = np.sqrt(2.0 * G * total_mass / np.sqrt(radii**2 + scale_radius**2))
    speeds = np.zeros(n)
    for i in range(n):
        while True:
            x1, x2 = rng.uniform(0.0, 1.0, size=2)
            # g(x) = x^2 (1-x^2)^3.5 ; peak g_max ~= 0.1 at x ~= 0.4
            if 0.1 * x2 <= x1**2 * (1.0 - x1**2) ** 3.5:
                speeds[i] = x1 * escape_speed[i]
                break

    directions = rng.normal(size=(n, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    velocities = speeds[:, np.newaxis] * directions

    # recenter to remove COM drift from finite-N sampling noise
    positions -= np.average(positions, axis=0, weights=masses)
    velocities -= np.average(velocities, axis=0, weights=masses)

    return positions, velocities, masses


def star_cluster(
    n: int = 500, total_mass: float = 300.0, scale_radius: float = 2.0, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Alias for a larger Plummer sphere, framed as a globular-cluster preset."""
    return plummer_sphere(n=n, total_mass=total_mass, scale_radius=scale_radius, seed=seed)


def disk_galaxy(
    n: int = 500,
    central_mass: float = 1000.0,
    disk_mass: float = 200.0,
    disk_radius: float = 10.0,
    thickness: float = 0.2,
    G: float = 1.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A rotating disk of test/light particles around a dominant central mass.

    Approximate rotation curve uses enclosed mass under the central point
    mass only (i.e. self-gravity of the disk is neglected for the initial
    velocity assignment). This is a common simplifying approximation for
    generating a "cold disk" initial condition; the disk's own gravity is
    still included in the N-body force evaluation once the run starts, so
    expect some early transient settling (spiral structure, mild heating).
    """
    rng = np.random.default_rng(seed)
    star_masses = np.full(n, disk_mass / n)

    radii = disk_radius * np.sqrt(rng.uniform(0.05, 1.0, size=n))
    theta = rng.uniform(0.0, 2 * np.pi, size=n)
    z = rng.normal(0.0, thickness, size=n)

    x = radii * np.cos(theta)
    y = radii * np.sin(theta)
    positions = np.stack([x, y, z], axis=1)

    speed = np.sqrt(G * central_mass / radii)
    vx = -speed * np.sin(theta)
    vy = speed * np.cos(theta)
    vz = np.zeros(n)
    velocities = np.stack([vx, vy, vz], axis=1)

    positions = np.vstack([[0.0, 0.0, 0.0], positions])
    velocities = np.vstack([[0.0, 0.0, 0.0], velocities])
    masses = np.concatenate([[central_mass], star_masses])

    return positions, velocities, masses


def ring_system(
    n: int = 300,
    central_mass: float = 500.0,
    ring_radius: float = 5.0,
    ring_width: float = 0.3,
    ring_mass: float = 5.0,
    G: float = 1.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A thin Saturn-ring-like collection of test particles on circular orbits."""
    rng = np.random.default_rng(seed)
    radii = rng.uniform(ring_radius - ring_width, ring_radius + ring_width, size=n)
    theta = rng.uniform(0.0, 2 * np.pi, size=n)
    z = rng.normal(0.0, ring_width * 0.05, size=n)

    x = radii * np.cos(theta)
    y = radii * np.sin(theta)
    positions = np.stack([x, y, z], axis=1)

    speed = np.sqrt(G * central_mass / radii)
    vx = -speed * np.sin(theta)
    vy = speed * np.cos(theta)
    velocities = np.stack([vx, vy, np.zeros(n)], axis=1)

    particle_masses = np.full(n, ring_mass / n)
    positions = np.vstack([[0.0, 0.0, 0.0], positions])
    velocities = np.vstack([[0.0, 0.0, 0.0], velocities])
    masses = np.concatenate([[central_mass], particle_masses])

    return positions, velocities, masses


def random_cloud(
    n: int = 200, extent: float = 5.0, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Uniformly random positions/masses in a cube, zero initial velocity.

    Not a physically motivated system -- this exists as a neutral,
    structure-free stress test for force-solver accuracy/performance
    comparisons (e.g. Barnes-Hut vs. direct summation), where we don't want
    any particular clustering or symmetry to make one solver look better or
    worse than it would on a generic distribution.
    """
    rng = np.random.default_rng(seed)
    positions = rng.uniform(-extent, extent, size=(n, 3))
    masses = rng.uniform(0.5, 2.0, size=n)
    velocities = np.zeros((n, 3))
    return positions, velocities, masses


def galaxy_merger(
    n_per_galaxy: int = 250,
    separation: float = 30.0,
    relative_speed: float = 0.5,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two disk galaxies on a collision course.

    Built by generating two independent disk_galaxy systems, offsetting one
    in position and giving both galaxies bulk velocities toward each other
    (small impact parameter along y so the merger is not perfectly head-on,
    which tends to look more visually interesting and avoids a degenerate
    coplanar collision).
    """
    pos1, vel1, mass1 = disk_galaxy(n=n_per_galaxy, seed=seed)
    pos2, vel2, mass2 = disk_galaxy(n=n_per_galaxy, seed=seed + 1)

    offset = np.array([separation, separation * 0.15, 0.0])
    approach_speed = np.array([-relative_speed, 0.0, 0.0])

    pos1 = pos1 - offset / 2.0
    pos2 = pos2 + offset / 2.0
    vel1 = vel1 + approach_speed
    vel2 = vel2 - approach_speed

    positions = np.vstack([pos1, pos2])
    velocities = np.vstack([vel1, vel2])
    masses = np.concatenate([mass1, mass2])
    return positions, velocities, masses

    positions = [np.zeros(3)]
    velocities = [np.zeros(3)]
    masses = [sun_mass]

    rng_angle = np.linspace(0, 2 * np.pi, len(radii), endpoint=False)
    for m, r, theta in zip(planet_masses, radii, rng_angle):
        pos = np.array([r * np.cos(theta), r * np.sin(theta), 0.0])
        speed = np.sqrt(G * sun_mass / r)
        vel_dir = np.array([-np.sin(theta), np.cos(theta), 0.0])
        positions.append(pos)
        velocities.append(speed * vel_dir)
        masses.append(m)

    return np.array(positions), np.array(velocities), np.array(masses)


def plummer_sphere(
    n: int = 200,
    total_mass: float = 100.0,
    scale_radius: float = 1.0,
    G: float = 1.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Plummer (1911) model: a common self-gravitating star-cluster profile.

    Density: rho(r) = (3M / 4*pi*a^3) * (1 + r^2/a^2)^(-5/2)
    Positions sampled via inverse-CDF of the enclosed mass fraction.
    Velocities sampled from the isotropic Plummer distribution function
    using rejection sampling on the local escape-velocity fraction.
    """
    rng = np.random.default_rng(seed)
    masses = np.full(n, total_mass / n)

    # --- positions: inverse-CDF sampling of Plummer radius ---
    mass_frac = rng.uniform(0.0, 1.0, size=n)
    radii = scale_radius * (mass_frac ** (-2.0 / 3.0) - 1.0) ** -0.5

    costheta = rng.uniform(-1.0, 1.0, size=n)
    theta = np.arccos(costheta)
    phi = rng.uniform(0.0, 2 * np.pi, size=n)

    x = radii * np.sin(theta) * np.cos(phi)
    y = radii * np.sin(theta) * np.sin(phi)
    z = radii * np.cos(theta)
    positions = np.stack([x, y, z], axis=1)

    # --- velocities: rejection sampling from Plummer DF ---
    escape_speed = np.sqrt(2.0 * G * total_mass / np.sqrt(radii**2 + scale_radius**2))
    speeds = np.zeros(n)
    for i in range(n):
        while True:
            x1, x2 = rng.uniform(0.0, 1.0, size=2)
            # g(x) = x^2 (1-x^2)^3.5 ; peak g_max ~= 0.1 at x ~= 0.4
            if 0.1 * x2 <= x1**2 * (1.0 - x1**2) ** 3.5:
                speeds[i] = x1 * escape_speed[i]
                break

    directions = rng.normal(size=(n, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    velocities = speeds[:, np.newaxis] * directions

    # recenter to remove COM drift from finite-N sampling noise
    positions -= np.average(positions, axis=0, weights=masses)
    velocities -= np.average(velocities, axis=0, weights=masses)

    return positions, velocities, masses


def star_cluster(
    n: int = 500, total_mass: float = 300.0, scale_radius: float = 2.0, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Alias for a larger Plummer sphere, framed as a globular-cluster preset."""
    return plummer_sphere(n=n, total_mass=total_mass, scale_radius=scale_radius, seed=seed)


def disk_galaxy(
    n: int = 500,
    central_mass: float = 1000.0,
    disk_mass: float = 200.0,
    disk_radius: float = 10.0,
    thickness: float = 0.2,
    G: float = 1.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A rotating disk of test/light particles around a dominant central mass.

    Approximate rotation curve uses enclosed mass under the central point
    mass only (i.e. self-gravity of the disk is neglected for the initial
    velocity assignment). This is a common simplifying approximation for
    generating a "cold disk" initial condition; the disk's own gravity is
    still included in the N-body force evaluation once the run starts, so
    expect some early transient settling (spiral structure, mild heating).
    """
    rng = np.random.default_rng(seed)
    star_masses = np.full(n, disk_mass / n)

    radii = disk_radius * np.sqrt(rng.uniform(0.05, 1.0, size=n))
    theta = rng.uniform(0.0, 2 * np.pi, size=n)
    z = rng.normal(0.0, thickness, size=n)

    x = radii * np.cos(theta)
    y = radii * np.sin(theta)
    positions = np.stack([x, y, z], axis=1)

    speed = np.sqrt(G * central_mass / radii)
    vx = -speed * np.sin(theta)
    vy = speed * np.cos(theta)
    vz = np.zeros(n)
    velocities = np.stack([vx, vy, vz], axis=1)

    positions = np.vstack([[0.0, 0.0, 0.0], positions])
    velocities = np.vstack([[0.0, 0.0, 0.0], velocities])
    masses = np.concatenate([[central_mass], star_masses])

    return positions, velocities, masses


def ring_system(
    n: int = 300,
    central_mass: float = 500.0,
    ring_radius: float = 5.0,
    ring_width: float = 0.3,
    ring_mass: float = 5.0,
    G: float = 1.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A thin Saturn-ring-like collection of test particles on circular orbits."""
    rng = np.random.default_rng(seed)
    radii = rng.uniform(ring_radius - ring_width, ring_radius + ring_width, size=n)
    theta = rng.uniform(0.0, 2 * np.pi, size=n)
    z = rng.normal(0.0, ring_width * 0.05, size=n)

    x = radii * np.cos(theta)
    y = radii * np.sin(theta)
    positions = np.stack([x, y, z], axis=1)

    speed = np.sqrt(G * central_mass / radii)
    vx = -speed * np.sin(theta)
    vy = speed * np.cos(theta)
    velocities = np.stack([vx, vy, np.zeros(n)], axis=1)

    particle_masses = np.full(n, ring_mass / n)
    positions = np.vstack([[0.0, 0.0, 0.0], positions])
    velocities = np.vstack([[0.0, 0.0, 0.0], velocities])
    masses = np.concatenate([[central_mass], particle_masses])

    return positions, velocities, masses


def galaxy_merger(
    n_per_galaxy: int = 250,
    separation: float = 30.0,
    relative_speed: float = 0.5,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two disk galaxies on a collision course.

    Built by generating two independent disk_galaxy systems, offsetting one
    in position and giving both galaxies bulk velocities toward each other
    (small impact parameter along y so the merger is not perfectly head-on,
    which tends to look more visually interesting and avoids a degenerate
    coplanar collision).
    """
    pos1, vel1, mass1 = disk_galaxy(n=n_per_galaxy, seed=seed)
    pos2, vel2, mass2 = disk_galaxy(n=n_per_galaxy, seed=seed + 1)

    offset = np.array([separation, separation * 0.15, 0.0])
    approach_speed = np.array([-relative_speed, 0.0, 0.0])

    pos1 = pos1 - offset / 2.0
    pos2 = pos2 + offset / 2.0
    vel1 = vel1 + approach_speed
    vel2 = vel2 - approach_speed

    positions = np.vstack([pos1, pos2])
    velocities = np.vstack([vel1, vel2])
    masses = np.concatenate([mass1, mass2])
    return positions, velocities, masses

    positions = [np.zeros(3)]
    velocities = [np.zeros(3)]
    masses = [sun_mass]

    rng_angle = np.linspace(0, 2 * np.pi, len(radii), endpoint=False)
    for m, r, theta in zip(planet_masses, radii, rng_angle):
        pos = np.array([r * np.cos(theta), r * np.sin(theta), 0.0])
        speed = np.sqrt(G * sun_mass / r)
        vel_dir = np.array([-np.sin(theta), np.cos(theta), 0.0])
        positions.append(pos)
        velocities.append(speed * vel_dir)
        masses.append(m)

    return np.array(positions), np.array(velocities), np.array(masses)


def plummer_sphere(
    n: int = 200,
    total_mass: float = 100.0,
    scale_radius: float = 1.0,
    G: float = 1.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Plummer (1911) model: a common self-gravitating star-cluster profile.

    Density: rho(r) = (3M / 4*pi*a^3) * (1 + r^2/a^2)^(-5/2)
    Positions sampled via inverse-CDF of the enclosed mass fraction.
    Velocities sampled from the isotropic Plummer distribution function
    using rejection sampling on the local escape-velocity fraction.
    """
    rng = np.random.default_rng(seed)
    masses = np.full(n, total_mass / n)

    # --- positions: inverse-CDF sampling of Plummer radius ---
    mass_frac = rng.uniform(0.0, 1.0, size=n)
    radii = scale_radius * (mass_frac ** (-2.0 / 3.0) - 1.0) ** -0.5

    costheta = rng.uniform(-1.0, 1.0, size=n)
    theta = np.arccos(costheta)
    phi = rng.uniform(0.0, 2 * np.pi, size=n)

    x = radii * np.sin(theta) * np.cos(phi)
    y = radii * np.sin(theta) * np.sin(phi)
    z = radii * np.cos(theta)
    positions = np.stack([x, y, z], axis=1)

    # --- velocities: rejection sampling from Plummer DF ---
    escape_speed = np.sqrt(2.0 * G * total_mass / np.sqrt(radii**2 + scale_radius**2))
    speeds = np.zeros(n)
    for i in range(n):
        while True:
            x1, x2 = rng.uniform(0.0, 1.0, size=2)
            # g(x) = x^2 (1-x^2)^3.5 ; peak g_max ~= 0.1 at x ~= 0.4
            if 0.1 * x2 <= x1**2 * (1.0 - x1**2) ** 3.5:
                speeds[i] = x1 * escape_speed[i]
                break

    directions = rng.normal(size=(n, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    velocities = speeds[:, np.newaxis] * directions

    # recenter to remove COM drift from finite-N sampling noise
    positions -= np.average(positions, axis=0, weights=masses)
    velocities -= np.average(velocities, axis=0, weights=masses)

    return positions, velocities, masses


def star_cluster(
    n: int = 500, total_mass: float = 300.0, scale_radius: float = 2.0, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Alias for a larger Plummer sphere, framed as a globular-cluster preset."""
    return plummer_sphere(n=n, total_mass=total_mass, scale_radius=scale_radius, seed=seed)


def disk_galaxy(
    n: int = 500,
    central_mass: float = 1000.0,
    disk_mass: float = 200.0,
    disk_radius: float = 10.0,
    thickness: float = 0.2,
    G: float = 1.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A rotating disk of test/light particles around a dominant central mass.

    Approximate rotation curve uses enclosed mass under the central point
    mass only (i.e. self-gravity of the disk is neglected for the initial
    velocity assignment). This is a common simplifying approximation for
    generating a "cold disk" initial condition; the disk's own gravity is
    still included in the N-body force evaluation once the run starts, so
    expect some early transient settling (spiral structure, mild heating).
    """
    rng = np.random.default_rng(seed)
    star_masses = np.full(n, disk_mass / n)

    radii = disk_radius * np.sqrt(rng.uniform(0.05, 1.0, size=n))
    theta = rng.uniform(0.0, 2 * np.pi, size=n)
    z = rng.normal(0.0, thickness, size=n)

    x = radii * np.cos(theta)
    y = radii * np.sin(theta)
    positions = np.stack([x, y, z], axis=1)

    speed = np.sqrt(G * central_mass / radii)
    vx = -speed * np.sin(theta)
    vy = speed * np.cos(theta)
    vz = np.zeros(n)
    velocities = np.stack([vx, vy, vz], axis=1)

    positions = np.vstack([[0.0, 0.0, 0.0], positions])
    velocities = np.vstack([[0.0, 0.0, 0.0], velocities])
    masses = np.concatenate([[central_mass], star_masses])

    return positions, velocities, masses


def ring_system(
    n: int = 300,
    central_mass: float = 500.0,
    ring_radius: float = 5.0,
    ring_width: float = 0.3,
    ring_mass: float = 5.0,
    G: float = 1.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A thin Saturn-ring-like collection of test particles on circular orbits."""
    rng = np.random.default_rng(seed)
    radii = rng.uniform(ring_radius - ring_width, ring_radius + ring_width, size=n)
    theta = rng.uniform(0.0, 2 * np.pi, size=n)
    z = rng.normal(0.0, ring_width * 0.05, size=n)

    x = radii * np.cos(theta)
    y = radii * np.sin(theta)
    positions = np.stack([x, y, z], axis=1)

    speed = np.sqrt(G * central_mass / radii)
    vx = -speed * np.sin(theta)
    vy = speed * np.cos(theta)
    velocities = np.stack([vx, vy, np.zeros(n)], axis=1)

    particle_masses = np.full(n, ring_mass / n)
    positions = np.vstack([[0.0, 0.0, 0.0], positions])
    velocities = np.vstack([[0.0, 0.0, 0.0], velocities])
    masses = np.concatenate([[central_mass], particle_masses])

    return positions, velocities, masses


def galaxy_merger(
    n_per_galaxy: int = 250,
    separation: float = 30.0,
    relative_speed: float = 0.5,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two disk galaxies on a collision course.

    Built by generating two independent disk_galaxy systems, offsetting one
    in position and giving both galaxies bulk velocities toward each other
    (small impact parameter along y so the merger is not perfectly head-on,
    which tends to look more visually interesting and avoids a degenerate
    coplanar collision).
    """
    pos1, vel1, mass1 = disk_galaxy(n=n_per_galaxy, seed=seed)
    pos2, vel2, mass2 = disk_galaxy(n=n_per_galaxy, seed=seed + 1)

    offset = np.array([separation, separation * 0.15, 0.0])
    approach_speed = np.array([-relative_speed, 0.0, 0.0])

    pos1 = pos1 - offset / 2.0
    pos2 = pos2 + offset / 2.0
    vel1 = vel1 + approach_speed
    vel2 = vel2 - approach_speed

    positions = np.vstack([pos1, pos2])
    velocities = np.vstack([vel1, vel2])
    masses = np.concatenate([mass1, mass2])
    return positions, velocities, masses
