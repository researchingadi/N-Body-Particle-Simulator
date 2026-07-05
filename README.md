# Advanced N-Body Particle Simulator

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
  center of mass, plus drift metrics used by the validation harness
  (`diagnostics/conservation.py`)
- Inelastic collision/merger handling, mass- and momentum-conserving
  (`simulation/engine.py`)
- Initial condition presets: binary orbit, figure-eight three-body,
  toy solar system, Plummer sphere / star cluster, rotating disk galaxy,
  ring system, galaxy merger (`initial_conditions/presets.py`)
- CSV trajectory export + JSON config export for reproducibility
  (`io_utils/export.py`)
- Static matplotlib visualization: trajectories, energy plots, momentum
  plots (`visualization/plot.py`)
- **Convergence/validation harness**: dt and softening sweeps, integrator
  comparison, CSV + DataFrame export (`validation/convergence.py`)
- **Validation plots**: drift-vs-dt and log-log convergence-order plots
  (`visualization/validation_plots.py`)
- Unit tests covering force symmetry, no self-force, momentum conservation,
  two-body orbit stability, energy drift bounds, merger conservation, and
  convergence-harness correctness (`tests/`)

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

## Validation and Convergence Testing

Before adding Barnes-Hut, GPU acceleration, or a frontend, we need evidence
that the solver behaves the way the numerics say it should — not just that
it runs without crashing. `validation/convergence.py` + `scripts/run_convergence_study.py`
answer that.

**Why this matters.** A simulator can look fine in a rendered animation
while quietly integrating garbage — a bounded circular orbit that's actually
losing energy at 1% per orbit will still *look* like a stable orbit for a
while. Convergence testing catches this by checking a falsifiable numerical
prediction: for a given method, error should shrink at a known rate as the
timestep shrinks. If it doesn't, something in the force law, integrator, or
diagnostics is wrong, and that needs to be found now — before Barnes-Hut or
GPU code is built on top of it and the bug becomes much harder to isolate.

**What "relative energy drift" means.** `(E(t) - E(0)) / |E(0)|` — how much
total energy has changed, as a fraction of the starting energy. Two versions
are tracked:
- *final relative drift* — the value at the last recorded step (can look
  artificially small if the run happens to sample near a trough of an
  oscillation).
- *max relative drift (envelope)* — the largest deviation from `E(0)` at
  any point in the run. This is the more honest number for a symplectic
  integrator, whose energy error oscillates rather than trending in one
  direction.

**Why leapfrog should show bounded, oscillatory error — not perfect
conservation.** Leapfrog and velocity Verlet are symplectic integrators: they
exactly conserve a nearby "shadow" Hamiltonian rather than the true one, so
the true energy oscillates within a bounded envelope instead of drifting to
zero or diverging. What *should* improve as `dt` shrinks is the **size** of
that envelope — empirically close to second order (envelope ∝ dt²) for
leapfrog, confirmed below. This project does not claim exact energy
conservation anywhere, and treats a claim like that as a bug report, not a
feature.

**Why Euler is included.** Euler is not symplectic. Its energy error is
expected to grow steadily over time at any fixed `dt`, not oscillate — it's
included solely as a known-bad control to demonstrate what happens *without*
a symplectic integrator, never as a candidate for production use.

**Measured results (binary orbit, `dt` swept 0.02 → 0.0025, softening=0.01,
total_time=20):**

| dt | leapfrog envelope drift | Euler envelope drift |
|---|---|---|
| 0.02 | 6.2e-9 | 0.147 |
| 0.01 | 1.1e-9 | 0.084 |
| 0.005 | 2.4e-10 | 0.046 |
| 0.0025 | 5.9e-11 | 0.024 |

Leapfrog's envelope shrinks by roughly 4x per halving of `dt` (empirical
convergence order ≈ 2, matching theory) and stays 6–7 orders of magnitude
below Euler's at every matched `dt`. The figure-eight three-body system
(a much more demanding integrator stress test) shows the same clean
second-order convergence. Full numbers are in `outputs/convergence/*.csv`.

**Limitations / caveats:**
- These sweeps use small-to-moderate systems (2–3 bodies); convergence order
  for larger, chaotic N-body systems (Plummer sphere, disk galaxy) is
  noisier because chaotic trajectory divergence dominates over integrator
  truncation error at longer times — this is expected and is itself a
  motivating reason for the future chaos/Lyapunov diagnostics module, not
  a flaw in the harness.
- Angular momentum and center-of-mass drift are reported but are already
  near machine precision for these symmetric test systems (both are exactly
  conserved quantities with no dt-dependent truncation error in this force
  law), so they serve mainly as a sanity check / regression guard rather
  than a convergence-order signal.

**Running it:**

```bash
python scripts/run_convergence_study.py
```

Produces, in `outputs/convergence/`:
- `dt_sweep_binary_orbit.csv`, `dt_sweep_figure_eight.csv`
- `integrator_comparison_binary_orbit.csv`
- energy/angular-momentum/COM drift vs. dt plots, a log-log convergence-order
  plot with dt¹/dt² reference lines, and a leapfrog-vs-Euler comparison plot

## Quickstart

```bash
pip install -r requirements.txt
pytest                                    # run the full test suite
python scripts/run_demo.py                # produces plots + CSV in outputs/
python scripts/run_convergence_study.py   # convergence/validation study
```

## Architecture

```
neural_gravity_lab/
  physics/             Force law + integrators (pure functions, no state)
  diagnostics/         Conservation quantity calculations + drift metrics
  initial_conditions/  Reproducible test-system generators
  simulation/          Simulation class: owns state, stepping loop, collisions
  io_utils/            CSV/JSON export for reproducibility
  visualization/       Matplotlib plots + validation/convergence plots
  validation/          Convergence/validation sweep harness (dt, softening)
  tests/               Physics correctness + convergence-harness test suite
  scripts/             Runnable demos and the convergence study
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
5.5. ✅ Convergence/validation harness (dt & softening sweeps, integrator comparison)
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
