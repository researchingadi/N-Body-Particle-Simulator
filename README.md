# Neural Gravity Lab: Advanced N-Body Particle Simulator

A research-grade gravitational N-body simulation platform combining
validated numerical physics, high-performance computation, cinematic
interactive visualization, and (later) scientific ML / uncertainty
quantification.

**Status: Stage 1 — CPU direct solver core.** Correctness-first. No GPU,
Barnes-Hut, frontend, or ML yet — those come after this core is validated.

## What's implemented right now

- Direct O(N²) Newtonian gravity with Plummer softening (`physics/forces.py`)
- Integrators: leapfrog KDK, velocity Verlet, RK4, and a deliberately-bad
  Euler baseline for comparison (`physics/integrators.py`)
- Conservation diagnostics: energy, linear momentum, angular momentum,
  center of mass (`diagnostics/conservation.py`)
- Inelastic collision/merger handling, mass- and momentum-conserving
  (`simulation/engine.py`)
- Initial condition presets: binary orbit, figure-eight three-body,
  toy solar system, Plummer sphere / star cluster, rotating disk galaxy,
  ring system, galaxy merger (`initial_conditions/presets.py`)
- CSV trajectory export + JSON config export for reproducibility
  (`io_utils/export.py`)
- Static matplotlib visualization: trajectories, energy plots, momentum
  plots (`visualization/plot.py`)
- Unit tests covering force symmetry, no self-force, momentum conservation,
  two-body orbit stability, energy drift bounds, and merger conservation
  (`tests/`)

## What's explicitly NOT here yet (by design)

- Barnes-Hut tree approximation
- GPU acceleration (Taichi/CUDA/WebGPU)
- FastAPI backend
- React/Three.js frontend
- PyTorch ML surrogate / uncertainty quantification
- Chaos / Lyapunov diagnostics

These are staged deliberately — see the roadmap below. Building them before
the core solver is validated would mean debugging physics and infrastructure
at the same time, which is how these projects rot.

## Quickstart

```bash
pip install -r requirements.txt
pytest                          # run the physics/conservation test suite
python scripts/run_demo.py      # produces plots + CSV in outputs/
```

## Architecture

```
neural_gravity_lab/
  physics/            Force law + integrators (pure functions, no state)
  diagnostics/         Conservation quantity calculations
  initial_conditions/  Reproducible test-system generators
  simulation/          Simulation class: owns state, stepping loop, collisions
  io_utils/            CSV/JSON export for reproducibility
  visualization/       Matplotlib plots (pre-frontend validation tool)
  tests/               Physics correctness test suite
  scripts/             Runnable demos
```

Design principle: `physics/` functions are pure (positions, masses in ->
accelerations out) so Barnes-Hut and GPU implementations can be dropped in
behind the same signature and validated against this direct solver.

## Roadmap

1. ✅ Minimal correct CPU direct solver
2. ✅ Leapfrog/Verlet integrator + diagnostics
3. ✅ Initial condition presets
4. ✅ Tests and validation
5. ✅ Simple visualization/export
6. ⬜ Interactive frontend (React/Next.js + Three.js/WebGPU)
7. ⬜ GPU acceleration (Taichi or CUDA), validated against the direct solver
8. ⬜ Barnes-Hut O(N log N) approximation, validated against direct solver
9. ⬜ Advanced visuals: trails, glow, camera modes, BH tree overlay, benchmarking
10. ⬜ ML surrogate + uncertainty quantification + chaos/Lyapunov diagnostics

## Why this project

Demonstrates: numerical methods for physical simulation (symplectic
integration, softening, conservation-law validation), scientific software
engineering (modular architecture, reproducibility, test-driven physics),
and a credible path to HPC (GPU) and scientific ML extensions — without
overclaiming features that aren't built yet.
