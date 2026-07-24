# Neural Gravity Lab: Advanced N-Body Particle Simulator

Research grade gravitational N-body simulation platform combining
validated numerical physics, high-performance computation, cinematic
interactive visualization, and (later) scientific ML / uncertainty
quantification.

**Status: Stage 4B — React/Three.js cinematic frontend added.** Correctness-first.
No ML/uncertainty features yet — those come after this core is validated.

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
- **Barnes-Hut octree approximation**: O(N log N) alternative to direct
  summation, same `accel_fn(positions, masses) -> accelerations` contract,
  swappable via `SimulationConfig.solver` (`physics/barnes_hut.py`)
- **Barnes-Hut validation harness**: theta sweeps, particle-count scaling,
  accuracy/runtime tradeoff, CSV + plots
  (`validation/barnes_hut_validation.py`, `visualization/barnes_hut_plots.py`)
- **Taichi-accelerated direct solver**: same O(N²) sum as the NumPy
  reference (not an approximation), evaluated on parallel GPU/CPU threads;
  crash-safe backend detection that never assumes GPU hardware is present
  (`physics/taichi_forces.py`)
- **GPU validation harness**: NumPy direct vs. Taichi direct vs. Barnes-Hut
  CPU compared on accuracy, runtime, speedup, and peak memory across
  particle counts, CSV + plots
  (`validation/gpu_validation.py`, `visualization/gpu_validation_plots.py`)
- **FastAPI backend**: thin HTTP layer over the same `Simulation` class
  every script uses -- create/step/query/delete simulations by preset,
  solver, and integrator, over a REST API the frontend calls
  (`api/main.py`, `api/models.py`, `api/simulation_manager.py`); CORS
  enabled for local frontend dev servers
- **React/Three.js cinematic frontend**: full-bleed WebGL scene with
  instanced glowing particles (mass/velocity-encoded color and size),
  shader-based fading orbital trails, bloom/vignette post-processing, four
  camera modes, and a live scientific HUD (diagnostics, controls, preset/
  solver selection) -- calls the FastAPI backend exclusively, computes no
  physics itself (`frontend/`)
- Unit tests covering force symmetry, no self-force, momentum conservation,
  two-body orbit stability, energy drift bounds, merger conservation,
  convergence-harness correctness, Barnes-Hut accuracy/tree-aggregation
  correctness, Taichi backend correctness/fallback safety, and the FastAPI
  layer's request handling/status codes/CORS (`tests/`)

## What's explicitly NOT here yet (by design)

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

## Barnes-Hut Approximation

**Why direct O(N²) doesn't scale.** Direct summation computes every
pairwise interaction, every step: double the particle count and the work
per step quadruples. That's fine for the tens-to-low-hundreds of bodies
used in the validation systems above, but a "star cluster" or "galaxy"
preset with thousands of particles makes O(N²) the bottleneck long before
anything else does — and, as measured below, its memory footprint (several
N×N×3 temporary arrays) can become a problem before its runtime does.

**How Barnes-Hut approximates distant groups.** Particles are organized
into an octree: recursively subdivide space into 8 sub-cubes until each
leaf holds one particle (or, in the degenerate case of near-coincident
positions, a handful). Each node also stores the total mass and center of
mass of everything in its subtree. When computing the force on particle
*i*, the tree is walked from the root; a node representing a *distant
enough* group of particles is treated as a single point mass at that
group's center of mass, instead of visiting every particle inside it
individually.

**What theta means.** "Distant enough" is decided by the opening angle
`theta`: a node of physical size `s` at distance `d` from the particle
being evaluated is approximated (not opened into its children) when
`s / d < theta`. Smaller `theta` opens more nodes — closer to exact direct
summation, slower. Larger `theta` approximates more aggressively — faster,
less accurate. `theta → 0` recovers direct summation exactly; typical
practical values are ~0.3–1.0.

**The accuracy/performance tradeoff (measured, `random_cloud`, N=500,
softening=0.05):**

| theta | mean rel. error | median rel. error | max rel. error |
|---|---|---|---|
| 0.2 | 3.5e-4 | 2.5e-4 | 3.0e-3 |
| 0.4 | 3.1e-3 | 2.3e-3 | 2.8e-2 |
| 0.6 | 1.1e-2 | 7.8e-3 | 8.7e-2 |
| 0.8 | 2.5e-2 | 1.8e-2 | 3.0e-1 |
| 1.0 | 4.7e-2 | 3.3e-2 | 5.0e-1 |

Error grows monotonically with theta, as it should. The same monotonic
trend holds for the Plummer sphere and disk galaxy test systems (see
`outputs/barnes_hut/theta_sweep_*.csv`); max relative error is always
noticeably worse than mean/median, because it reflects worst-case
individual particles (typically ones near a large, close, coarsely-opened
node), not the typical-case behavior most particles see.

**Runtime/speedup vs. N (measured, `random_cloud`, theta=0.5, median of 3
runs):**

| N | direct (s) | Barnes-Hut (s) | speedup |
|---|---|---|---|
| 100 | 0.0007 | 0.012 | 0.06x |
| 300 | 0.006 | 0.062 | 0.10x |
| 1,000 | 0.071 | 0.338 | 0.21x |
| 3,000 | 0.68 | 1.43 | 0.47x |
| 6,000 | 2.69 | 3.59 | 0.75x |

**Barnes-Hut does not beat direct summation anywhere in this table** — and
that's reported honestly, not hidden. Two real, distinct reasons, measured
on this machine: (1) `physics/barnes_hut.py` is currently a straightforward
recursive Python octree, not yet a performance-optimized implementation, so
it carries real per-node Python overhead that a vectorized or compiled tree
walk wouldn't; (2) direct summation's O(N²) *memory* footprint got the
process OOM-killed at N=8,000 on this machine's available RAM, before a
runtime crossover was reached — the speedup trend was still climbing
(~0.85x at N=7,000) when memory became the limit, not Barnes-Hut's speed.
Both are legitimate findings about *this implementation on this machine*,
not evidence that Barnes-Hut's algorithmic advantage is fake — the
trend is real and rising; optimizing the tree walk (vectorization, or
eventually a compiled/GPU implementation) is future work, not this stage's
job. Per the project's correctness-first ordering, an accurate slow tree
beats a fast unvalidated one.

**Why the direct solver remains the validation reference.** Barnes-Hut is
an approximation of direct summation, not a more accurate alternative to
it — it can never be "more correct" than the O(N²) sum it's approximating.
Every accuracy number above is measured as Barnes-Hut's deviation *from*
direct summation, never the reverse, and that won't change even after
Barnes-Hut is fast: the direct solver stays the ground truth this project
validates every other force backend against (Barnes-Hut now, GPU later).

**Using it:** set `solver="barnes_hut"` (and optionally `theta=...`) on
`SimulationConfig` — everything else about `Simulation` is unchanged:

```python
config = SimulationConfig(dt=0.005, softening=0.05, solver="barnes_hut", theta=0.5)
sim = Simulation(positions, velocities, masses, config)
sim.run(n_steps=1000)
```

**Running the validation script:**

```bash
python scripts/run_barnes_hut_validation.py   # accuracy (theta sweeps) + N-scaling plots
python benchmarks/benchmark_barnes_hut.py     # focused runtime/speedup regression benchmark
```

Produces, in `outputs/barnes_hut/`:
- `theta_sweep_<system>.csv` for random_cloud, plummer_sphere, disk_galaxy
- `particle_scaling_random_cloud.csv`, `benchmark_particle_scaling.csv`
- error-vs-theta, accuracy/runtime tradeoff, runtime-vs-N, and
  speedup-vs-N plots

**Limitations/caveats:**
- Pure-Python recursive tree: correct and readable, not yet fast. A
  vectorized or compiled (Cython/GPU) tree walk is future work once this
  version is validated and profiled — exactly the project's stated order
  (correctness before performance).
- No crossover N was reached within this machine's memory budget (see
  above); the crossover exists in principle (better asymptotic complexity)
  but wasn't directly observed here.
- Energy/momentum/angular-momentum diagnostics in `Simulation.record()`
  always use the exact direct-solver potential energy calculation
  regardless of which solver (`direct` or `barnes_hut`) is driving the
  dynamics — this is intentional: it keeps the diagnostics honest and
  comparable across solvers rather than compounding approximation error
  into the very numbers meant to detect problems.
- The `MAX_TREE_DEPTH` fallback (bucket multiple particles into one leaf
  when positions are so close that subdivision can't separate them) is a
  pragmatic degenerate-case guard, not a normal code path — it's exercised
  by `test_coincident_positions_do_not_crash_or_produce_nan` but shouldn't
  trigger on realistic particle distributions.

## GPU Acceleration with Taichi

**Why direct O(N²) force computation is expensive.** Every particle needs
a contribution from every other particle: double N and the arithmetic
quadruples. Stage 2's Barnes-Hut tree cuts the *algorithmic* cost to
O(N log N) by approximating distant groups — but the direct sum itself
never gets cheaper; the only way to make direct summation faster without
approximating it is to make each unit of that O(N²) arithmetic finish
faster, by doing many of them at once.

**Why it's highly parallelizable.** The acceleration on particle *i*
depends only on particle *i*'s position and every other particle's
position/mass — it does not depend on any other particle's *acceleration*.
That means all N per-particle force sums are completely independent of
each other and can be computed simultaneously, with no coordination needed
between them. This is about as parallelism-friendly as physics gets: it's
the textbook "embarrassingly parallel" case.

**How the GPU maps particles to parallel work.** `physics/taichi_forces.py`
expresses this directly: one Taichi kernel thread is launched per particle
*i*; each thread runs the same sequential inner loop over all *j*, summing
that particle's force contributions, exactly mirroring
`physics.forces.compute_accelerations`'s math
(`a_i = G * sum_j m_j (r_j - r_i) / (|r_j-r_i|^2+eps^2)^{3/2}`). Taichi's
outer-loop-over-a-kernel-range is what gets parallelized automatically —
across GPU threads if a real GPU backend is active, across CPU threads
otherwise. Same formula, same softening convention, same G — only the
execution strategy changes.

**Why GPU direct is still O(N²) arithmetic, just faster wall-clock.**
Parallelizing doesn't change the algorithm's complexity class — it's still
N² total pairwise terms — it changes how many of those terms get computed
*at the same time*. That's why this backend is validated against, not
compared favorably to, the NumPy reference: it should reproduce the exact
same sum, just quicker.

**Why NumPy direct remains the numerical reference.** `taichi_direct` is
not an approximation the way Barnes-Hut is — it's the identical O(N²) sum,
so any difference from `physics.forces.compute_accelerations` should be
floating-point summation-order noise (measured: ~1e-16 relative, the same
order as machine epsilon), never a systematic error. `tests/test_taichi_forces.py`
holds it to that tight bar specifically, unlike Barnes-Hut's
tolerance-band tests. The direct solver stays the ground truth every
backend (Barnes-Hut, Taichi, any future one) is checked against.

**The speedup metric.**

```
speedup = runtime_reference / runtime_backend
```

`speedup > 1` means the backend is faster than the reference (NumPy
direct); `speedup < 1` means it's slower. A speedup of 3.0x means the
backend finished in a third of the reference's time.

**Compilation overhead is real and must be excluded from measurement.**
Taichi JIT-compiles each kernel the first time it's called for a given
argument type signature — confirmed directly while building this module: a
"first" call at N=100 took 0.68s (compilation + execution combined), while
an immediately following call at N=300 (same already-compiled kernel) took
0.002s. `validation/gpu_validation.py`'s `warmup_taichi_backend` runs one
small throwaway call before any timed measurement, specifically so
compilation cost never contaminates a benchmark number. Any Taichi
benchmark that skips this step will make whichever N it measures first
look artificially, misleadingly slow.

**GPU availability on this machine: none detected, and that's reported
honestly, not hidden.** `init_taichi()` probes for CUDA, Vulkan, and Metal
backends before falling back to CPU. On the machine this was built and
tested on, none were available — Taichi's CPU backend (multi-threaded, via
LLVM) was used for every number below. Two non-obvious things were found
while getting this far, both worth knowing if you're extending this code:

1. **Taichi silently falls back to CPU** for a GPU arch request that isn't
   really available — no exception is raised, so a plain `try/except`
   around `ti.init(arch=ti.cuda)` cannot detect a fallback. Detecting it
   requires checking the *actual* arch Taichi selected after init
   (`ti.lang.impl.current_cfg().arch`) against what was requested.
2. **Taichi's own GPU arch probing segfaulted the whole Python process**
   on this machine (a Vulkan loader library was present with no working
   ICD behind it, and Taichi's `ti.gpu` meta-arch selector crashed hard
   while probing it — not a catchable Python exception). `init_taichi`
   therefore probes each candidate GPU arch in an *isolated subprocess*
   first, so a crash there can't take down the caller; only a probe that
   both survives and reports a genuine matching arch causes the main
   process to actually initialize on that backend.

**Measured results (`random_cloud`, theta=0.5 for Barnes-Hut, median of 3
runs, Taichi on CPU backend — no GPU present on this machine):**

| N | numpy_direct (s) | taichi_direct (s) | taichi speedup | barnes_hut_cpu (s) | BH speedup |
|---|---|---|---|---|---|
| 100 | 0.0005 | 0.0005 | 1.01x | 0.010 | 0.05x |
| 300 | 0.0062 | 0.0022 | 2.87x | 0.057 | 0.11x |
| 1,000 | 0.057 | 0.023 | 2.46x | 0.311 | 0.18x |
| 3,000 | 0.61 | 0.20 | 3.01x | 1.24 | 0.50x |
| 6,000 | 2.52 | 0.84 | 3.00x | 3.29 | 0.77x |

Taichi's max relative error vs. numpy_direct across every N tested: under
`5e-15` — the floating-point noise floor, not a growing approximation
error, exactly as expected for a non-approximating backend. **Even without
any real GPU**, Taichi's multi-threaded CPU execution beats single-threaded
NumPy by roughly 2.5–3x across this whole range — a genuinely useful
result distinct from (and not dependent on) actual GPU hardware. On a
machine with a working CUDA/Vulkan/Metal backend, `init_taichi()` would
select it automatically and the same code would run on the GPU with no
code changes, likely with a substantially larger speedup at high N; that
claim is not verified here since no such hardware was available to test on.

**Using it:** set `solver="taichi_direct"` on `SimulationConfig` — same
`Simulation` API as every other solver:

```python
config = SimulationConfig(dt=0.005, softening=0.05, solver="taichi_direct")
sim = Simulation(positions, velocities, masses, config)
sim.run(n_steps=1000)
```

**Running the validation script:**

```bash
python scripts/run_gpu_validation.py
```

Produces `outputs/gpu_validation/gpu_validation_results.csv` and four
plots: runtime vs. N, speedup vs. N (both backends against numpy_direct),
Taichi's error vs. N (should stay flat at the noise floor), and a grouped
bar chart comparing all three backends' runtime at each N.

**Convergence results still hold.** Stage 3 changes *which backend*
computes accelerations, not the integrator or the physics — the leapfrog
convergence results validated in Stage 1.5 (`outputs/convergence/`) are
unaffected by which force backend drives them, since `taichi_direct` matches
the direct solver's math exactly, not approximately. The convergence-order
relationship documented there,

```
error(dt) ≈ C · dt^p
p ≈ log(error_1 / error_2) / log(dt_1 / dt_2)
```

with `p ≈ 2` measured for leapfrog, describes integrator behavior that is
independent of which force backend supplies the accelerations each step —
switching to `taichi_direct` (or Barnes-Hut) does not require re-deriving
or re-validating that relationship, only re-running it if you want to
confirm no regression (`test_simulation_can_run_with_taichi_backend`
already spot-checks this end-to-end).

**Limitations/caveats:**
- No real GPU was available to test on. Every number above reflects
  Taichi's CPU backend. The code path that would select a genuine GPU
  backend is implemented and tested for safety (see
  `test_backend_fallback_does_not_crash_when_gpu_unavailable`), but its
  performance on actual GPU hardware is unverified.
- `peak_rss_mb` in the validation CSV is a whole-process memory
  high-water-mark, not a measurement isolated to one solver call — read it
  as a trend across increasing N within one script run, not a precise
  per-call number (see `validation/gpu_validation.py`'s `_peak_rss_mb`
  docstring for why a cleaner per-call measurement would need
  subprocess-level isolation).
- OpenGL was excluded from the GPU arch candidates: it crashed even the
  isolated-subprocess probe on this machine, unlike cuda/vulkan/metal,
  which fail softly (silent CPU fallback).
- Taichi's kernel functions cannot use `from __future__ import annotations`
  (PEP 563) — the `@ti.kernel` decorator needs real, evaluated
  `ti.types.ndarray(...)` objects, not stringified annotations. This is
  why `physics/taichi_forces.py` alone omits that import, unlike the rest
  of this codebase; it's called out explicitly in that file's module
  docstring so it doesn't look like an accidental inconsistency.

## FastAPI Simulation Backend

**The backend wraps the validated Python engine — it does not reimplement
it.** Every route handler in `api/main.py` ultimately calls the same
`simulation.engine.Simulation` class, the same `initial_conditions.presets`
factory functions, and the same `diagnostics.conservation` functions that
`scripts/run_demo.py` and every validation script use. No physics,
integration, or force-law logic lives anywhere under `api/` — that module
is pure HTTP-shaped serialization and orchestration around code that was
already validated in Stages 1 through 3. Choosing a solver via the API
(`"direct"`, `"barnes_hut"`, or `"taichi_direct"`) selects the exact same
`SimulationConfig.solver` field a Python script would set directly.

**The frontend will call this API.** Stage 6 (the React/Three.js
interactive frontend) is not built yet, but this is the contract it will
be built against: create a simulation from a preset, step it forward,
poll for particle state to render, poll for diagnostics to plot. Physics
stays in Python on the server; the frontend's job is rendering and
interaction, never force computation — the project brief's "do not move
physics into JavaScript" constraint, enforced by construction here since
there's simply no force-law code available to move.

**Endpoints:**

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Backend status, available solvers/integrators, active simulation count |
| GET | `/presets` | Available initial-condition presets + their tunable parameters |
| POST | `/simulations` | Create a simulation from a preset + config, returns its id |
| POST | `/simulations/{id}/step` | Advance by `n_steps` (default 1), returns resulting state |
| GET | `/simulations/{id}/state` | Current positions/velocities/masses/time |
| GET | `/simulations/{id}/diagnostics` | Current energy/momentum/angular-momentum/COM, computed live |
| DELETE | `/simulations/{id}` | Remove a simulation from memory |

**Example payload** — create a binary orbit with Barnes-Hut, step it, read diagnostics:

```bash
curl -X POST http://127.0.0.1:8000/simulations \
  -H "Content-Type: application/json" \
  -d '{
    "preset": "binary_orbit",
    "preset_params": {"separation": 2.0},
    "dt": 0.005,
    "softening": 0.01,
    "integrator": "leapfrog",
    "solver": "direct"
  }'
# -> {"simulation_id": "...", "n_particles": 2, "config": {...}}

curl -X POST http://127.0.0.1:8000/simulations/{id}/step -d '{"n_steps": 200}'
# -> {"simulation_id": "...", "time": 1.0, "step_count": 200, "n_particles": 2,
#     "particles": {"positions": [[...], [...]], "velocities": [...], "masses": [...]}}

curl http://127.0.0.1:8000/simulations/{id}/diagnostics
# -> {"total_energy": -0.249994, "momentum": [0,0,0], "angular_momentum": [0,0,1.0], ...}
```

**Solver selection is validated automatically, not by hand-written checks.**
`solver`, `integrator`, and `preset` are all Pydantic `Literal` types
imported directly from `simulation.engine` (not re-declared) and
`api.models`'s own preset registry, so FastAPI generates correct OpenAPI
schema and rejects an invalid value (e.g. `"solver": "quantum_gpu"`) with a
422 automatically — there's exactly one place that defines which solvers
exist, and the API can't drift out of sync with it.

**Design decision: live diagnostics, not accumulated history.**
`Simulation.run()` (used by every script) calls `record()` each step,
appending to an in-memory history/trajectory list meant for post-hoc
plotting of one finite run. The API's `SimulationManager` deliberately
calls `Simulation.step()` directly instead, in a loop, and computes
diagnostics fresh from current particle state on every `/diagnostics`
request. A simulation behind this API might be stepped thousands of times
across many small requests over a long-lived server process — accumulating
full history for all of that would be an unbounded, unnecessary memory
leak for a use case that only ever wants the *current* state.

**Running it locally:**

```bash
uvicorn api.main:app --reload
```

Then visit `http://127.0.0.1:8000/docs` for FastAPI's interactive OpenAPI
UI, or run the smoke test (no server needed, uses an in-process
`TestClient`):

```bash
python scripts/run_api_smoke_test.py
```

**Limitations/caveats:**
- In-memory only: simulations vanish on server restart. No persistence
  layer (database, Redis) yet — the `SimulationManager` class is
  structured so one could be added later without changing any route
  handler, but that's future work, not this stage's job.
- No authentication, rate limiting, or multi-user isolation — anyone who
  can reach the server can create/step/delete any simulation. Fine for a
  local dev backend a frontend talks to; not production-hardened.
- Full trajectory/history export (the CSV export scripts use) isn't
  exposed via the API yet, only live current-state snapshots — deliberate,
  per the "live diagnostics, not accumulated history" design decision above.
- `star_cluster`, `random_cloud`, and `galaxy_merger` presets don't expose
  a `G` parameter (a pre-existing property of those functions in
  `initial_conditions/presets.py`, not something this API layer changes),
  so requesting a non-default `G` with those presets won't affect their
  initial-condition generation, only the dynamics afterward. Documented
  here so it isn't a surprise; not silently patched, since altering
  presets.py wasn't in scope for this stage.
- **CORS** is enabled (`CORSMiddleware`) for local frontend dev-server
  origins (Vite's default `http://localhost:5173`, plus common alternates)
  so the Stage 4B frontend can call this API across origins during local
  development — see `api/main.py`'s `_LOCAL_DEV_ORIGINS`. This is a
  development allowlist, not a production CORS policy; a real deployment
  would restrict this to the frontend's actual deployed origin.

## React/Three.js Cinematic Frontend

**The frontend calls the backend — it never computes physics.** Every
particle position rendered in the browser came from a `POST
/simulations/{id}/step` or `GET /simulations/{id}/state` response; nothing
under `frontend/src/` integrates the equations of motion, evaluates a
force law, or advances time. The validated Python engine (Stages 1–3)
remains the sole source of truth. This isn't just a design preference —
there is no force-law code anywhere in the frontend to have silently
diverged from the backend's, by construction.

**Stack:** React 19 + TypeScript, Vite, `@react-three/fiber` (React
bindings for Three.js — this is real Three.js underneath, not a different
engine), `@react-three/drei` (camera controls, starfield), `@react-three/postprocessing`
(bloom, vignette), Zustand (state), Recharts (diagnostics sparkline),
Tailwind CSS v4.

**Architecture** (matches the requested component breakdown):

```
frontend/src/
  api/client.ts              typed fetch wrapper, one function per backend endpoint
  types/simulation.ts        TypeScript mirror of api/models.py's Pydantic schemas
  store/simulationStore.ts   Zustand store: API orchestration + playback loop + view state
  components/
    App.tsx                  layout: full-bleed canvas + floating HUD panels
    SimulationCanvas.tsx      Three.js Canvas + post-processing composition
    ControlPanel.tsx          playback, preset/solver/integrator, view toggles
    PresetSelector.tsx        preset dropdown + per-preset parameter fields
    SolverSelector.tsx        solver/integrator + dt/softening/theta/G fields
    DiagnosticsPanel.tsx      live energy/momentum/COM readouts + drift sparkline
    TopBar.tsx                connection status, running/paused state
    Legend.tsx                color-encoding key, camera mode, controls hint
    scene/
      Particles.tsx           instanced-mesh particle renderer (the perf-critical piece)
      ParticleTrails.tsx      shader-based fading orbital trails
      CameraRig.tsx            orbit / follow-COM / follow-particle / flythrough
      ComMarker.tsx            pulsing center-of-mass marker
      VelocityVectors.tsx      optional per-particle velocity overlay
      SceneBackground.tsx      starfield, fog, grid toggle
      visualEncoding.ts        mass/velocity -> size/color mapping (shared, testable)
```

**Visual design.** Dark, blue-black deep-space background (not flat pure
black); particles are unlit, additively-blended emissive spheres so bloom
post-processing turns them into genuinely glowing bodies rather than flat
dots; mass maps to size via a fourth-root curve (compresses the ~100–1000x
mass ratios in presets like `disk_galaxy` or `solar_system` so a central
body reads as dominant without swallowing the scene); trails fade smoothly
via a custom per-vertex-alpha shader rather than a flat-opacity line. The
HUD is a floating "mission control" overlay (glass panels, `backdrop-filter`
blur) over a full-bleed canvas, not a boxed sidebar+content dashboard — the
simulation is the page, not a widget on it. Typography pairs Space Grotesk
(display/UI) with IBM Plex Mono (all numeric readouts, so digits don't
jitter as they update) — chosen for a scientific-instrument feel (an
oscilloscope/telemetry aesthetic) rather than a generic accent-color dark
theme.

**The four camera modes**, all in `scene/CameraRig.tsx`:
- **Orbit** — free user-controlled orbit (`OrbitControls` default behavior).
- **Follow center of mass** — the orbit target smoothly lerps to the
  live center of mass every frame.
- **Follow selected particle** — click any body to select it (instanced-mesh
  raycasting); the orbit target lerps to its live position.
- **Cinematic flythrough** — continuous slow auto-orbit around the current
  target (`OrbitControls.autoRotate`). This is an honest, lightweight
  reading of "flythrough": a smooth automatic orbit, not a scripted
  multi-waypoint camera path between points of interest — a real waypoint
  system is a reasonable future enhancement, not something to overclaim here.

**Performance discipline (the brief's explicit requirement, not an
afterthought).** Per-frame particle position/velocity data is read via
`useSimulationStore.getState()` **inside** `useFrame` callbacks, never via
the reactive `useSimulationStore(selector)` hook. Reading via `getState()`
does not subscribe to changes, so a playback tick updating hundreds of
particles' positions mutates the `InstancedMesh`'s GPU buffers directly
and never re-renders the React tree or reflows the HUD. Components that DO
use the reactive hook (`ControlPanel`, `DiagnosticsPanel`, `TopBar`)
deliberately only read low-frequency fields (`isRunning`, `diagnostics`,
UI toggles) for exactly this reason. Backend requests are batched: the
playback loop calls `/step` with a configurable `n_steps` (default 8) on a
configurable interval (default every 100ms), not once per animation frame
— see "Steps / batch" and "Interval" in the control panel.

**API integration.** `api/client.ts` implements exactly the six endpoints
from Stage 4A (`getHealth`, `getPresets`, `createSimulation`,
`stepSimulation`, `getState`, `getDiagnostics`, `deleteSimulation`), typed
against `types/simulation.ts`, which mirrors `api/models.py` field-for-field.
An `ApiError` class distinguishes network-level failures ("backend
unreachable") from HTTP error responses, so the UI can show a specific,
honest status rather than a generic failure.

**Running it:**

```bash
# Terminal 1 -- backend
uvicorn api.main:app --reload

# Terminal 2 -- frontend
cd frontend
npm install
npm run dev
```

Then open `http://localhost:5173`. The frontend expects the backend at
`http://127.0.0.1:8000` by default; override with `VITE_API_BASE_URL` (see
`frontend/.env.example`) if it's running elsewhere.

**Verified working end-to-end in this environment** (no GUI browser was
available to capture a screenshot in the sandbox this was built in — see
limitations below): `npm run build` compiles cleanly; both dev servers
start correctly; a real CORS preflight from `http://localhost:5173` against
the FastAPI backend succeeds; a full `POST /simulations` request sent with
that origin header succeeds end-to-end (201 Created, correct config echo).

**Limitations/caveats:**
- **No visual screenshot was captured.** This was built and verified in a
  sandboxed environment with no GUI browser and no WebGL-capable headless
  renderer available — verification covered the build, the dev servers,
  and the actual HTTP/CORS flow the browser would perform, but not a
  rendered frame. Run it locally to see the actual visual result.
- **No inter-frame interpolation.** Particle positions update once per
  batched backend step (every ~100ms by default), not smoothly
  extrapolated between backend updates. At small `dt`/large batch sizes
  this reads as smooth motion; at large `dt`/small batches, motion can look
  slightly stepped. Interpolating between the last two known states would
  fix this and is a reasonable next visual-polish pass.
- **Bundle size**: the production build is ~1.5MB (~410KB gzipped),
  dominated by Three.js + postprocessing + Recharts. Vite warns about this;
  code-splitting (e.g. lazy-loading Recharts, only used by one panel) would
  reduce initial load time and hasn't been done yet.
- **Trail and velocity-vector overlays rebuild their entire geometry buffer
  on every new backend tick**, not incrementally. Fine at the particle
  counts these presets produce (tens to low thousands); a very large
  system stepped at a very fast interval could make this the bottleneck
  before the backend's own solver does.
- No screenshot/export button, solver-comparison view, or real-time
  side-by-side diagnostic graphs beyond the single energy-drift sparkline
  yet — these were listed as optional/stretch goals; the core visual and
  functional requirements were prioritized first.
- Camera "flythrough" is auto-orbit, not scripted waypoints — see above.

## Quickstart

```bash
pip install -r requirements.txt
pytest                                       # run the full test suite
python scripts/run_demo.py                   # produces plots + CSV in outputs/
python scripts/run_convergence_study.py      # convergence/validation study
python scripts/run_barnes_hut_validation.py  # Barnes-Hut accuracy/performance study
python benchmarks/benchmark_barnes_hut.py    # Barnes-Hut runtime benchmark
python scripts/run_gpu_validation.py         # Taichi GPU/CPU vs. direct vs. Barnes-Hut study
uvicorn api.main:app --reload                # run the FastAPI backend locally
python scripts/run_api_smoke_test.py         # exercise the API without a running server

# Frontend (separate terminal, backend must be running)
cd frontend && npm install && npm run dev    # open http://localhost:5173
```

## Architecture

```
neural_gravity_lab/
  physics/             Force laws (direct + Barnes-Hut + Taichi) + integrators
  diagnostics/         Conservation quantity calculations + drift metrics
  initial_conditions/  Reproducible test-system generators
  simulation/          Simulation class: owns state, stepping loop, collisions
  io_utils/            CSV/JSON export for reproducibility
  visualization/       Matplotlib plots + validation/BH/GPU validation plots
  validation/          Convergence, Barnes-Hut, and GPU validation harnesses
  benchmarks/          Focused runtime benchmarks (regression checks)
  api/                 FastAPI HTTP layer over the Simulation engine
  frontend/            React/Three.js cinematic visualization client
  tests/               Physics correctness + validation-harness + API test suite
  scripts/             Runnable demos, validation studies, and an API smoke test
```

Design principle: `physics/` functions are pure (positions, masses in ->
accelerations out), so Barnes-Hut and Taichi implementations share the same
`accel_fn(positions, masses) -> accelerations` contract and can be
validated against the direct solver and swapped into `Simulation` via
`SimulationConfig.solver` without changing the engine's stepping loop. The
API layer extends this same principle one level up: `api/` depends on
`simulation/`, never the other way around, so the engine stays usable
standalone (scripts, notebooks, tests) with zero HTTP-framework dependency.
`frontend/` extends it one level further still: it depends only on `api/`'s
HTTP contract (never importing or reimplementing physics), so the engine,
the API, and the visualization client can each be validated, run, and
reasoned about independently.

## Roadmap

1. ✅ Minimal correct CPU direct solver
2. ✅ Leapfrog/Verlet integrator + diagnostics
3. ✅ Initial condition presets
4. ✅ Tests and validation
5. ✅ Simple visualization/export
5.5. ✅ Convergence/validation harness (dt & softening sweeps, integrator comparison)
6. ✅ Interactive frontend (React + Three.js, via the FastAPI backend)
7. ✅ GPU acceleration (Taichi), validated against the direct solver
8. ✅ Barnes-Hut O(N log N) approximation, validated against direct solver
4A. ✅ FastAPI backend wrapping the validated engine
4B. ✅ React/Three.js cinematic frontend (this stage)
9. ⬜ Advanced visuals: BH tree overlay, screenshot/export, solver-comparison view
10. ⬜ ML surrogate + uncertainty quantification + chaos/Lyapunov diagnostics

## Why this project

Demonstrates: numerical methods for physical simulation (symplectic
integration, softening, conservation-law validation), scientific software
engineering (modular architecture, reproducibility, test-driven physics),
and a credible path to HPC (GPU) and scientific ML extensions — without
overclaiming features that aren't built yet.
