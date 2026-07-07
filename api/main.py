"""FastAPI application exposing the validated Simulation engine over HTTP.

Design rationale: this is a thin serialization/orchestration layer. Route
handlers translate HTTP requests into calls against `simulation.engine.Simulation`
and `initial_conditions.presets` -- the same objects and functions used by
scripts/run_demo.py and friends -- then translate results back into Pydantic
response models. No physics, integration, or force-law code lives in this
file or anywhere under api/; duplicating any of that here would risk it
silently diverging from the validated engine in physics/, simulation/, and
diagnostics/, which is exactly what this stage is supposed to avoid.

Run locally:
    uvicorn api.main:app --reload

Then, e.g.:
    curl http://127.0.0.1:8000/health
    curl http://127.0.0.1:8000/presets
"""
from __future__ import annotations

import inspect
from typing import Callable

from fastapi import FastAPI, HTTPException

from api.models import (
    DeleteResponse,
    DiagnosticsResponse,
    HealthResponse,
    ParticleState,
    PresetInfo,
    SimulationConfigEcho,
    SimulationCreateRequest,
    SimulationCreateResponse,
    StateResponse,
    StepRequest,
)
from api.simulation_manager import SimulationManager, SimulationNotFoundError, compute_live_diagnostics
from initial_conditions.presets import (
    binary_orbit,
    disk_galaxy,
    galaxy_merger,
    plummer_sphere,
    random_cloud,
    ring_system,
    solar_system_like,
    star_cluster,
    three_body_figure_eight,
)
from simulation.engine import IntegratorName, Simulation, SimulationConfig, SolverName

app = FastAPI(
    title="Neural Gravity Lab API",
    description="HTTP layer over the validated Python N-body simulation engine.",
    version="0.1.0",
)

# Single in-memory registry for this process. See SimulationManager's
# docstring for why this is a class instance, not scattered module globals.
manager = SimulationManager()

PresetFactory = Callable[..., tuple]

# name (as used in the API) -> (factory function from initial_conditions.presets, description)
PRESET_REGISTRY: dict[str, tuple[PresetFactory, str]] = {
    "binary_orbit": (binary_orbit, "Two-body circular/eccentric orbit, center of mass at rest."),
    "figure_eight": (three_body_figure_eight, "Chenciner-Montgomery figure-eight three-body solution."),
    "solar_system": (solar_system_like, "Toy Sun + 4 planets on circular orbits (illustrative, not real data)."),
    "plummer_sphere": (plummer_sphere, "Plummer (1911) self-gravitating star-cluster density profile."),
    "star_cluster": (star_cluster, "Alias for a larger Plummer sphere, framed as a globular cluster."),
    "disk_galaxy": (disk_galaxy, "Rotating disk of test particles around a dominant central mass."),
    "ring_system": (ring_system, "Thin Saturn-ring-like collection of particles on circular orbits."),
    "galaxy_merger": (galaxy_merger, "Two disk galaxies on a collision course."),
    "random_cloud": (random_cloud, "Uniform random cloud -- a neutral solver stress test, not a physical system."),
}


def _default_params(factory: PresetFactory) -> dict[str, float | int]:
    """Introspect a preset factory's numeric keyword defaults, for GET /presets.

    This reads the defaults directly off the actual function signature
    rather than hand-maintaining a parallel description, so /presets can
    never drift out of sync with what initial_conditions/presets.py
    actually accepts.
    """
    params: dict[str, float | int] = {}
    for name, param in inspect.signature(factory).parameters.items():
        if param.default is not inspect.Parameter.empty and isinstance(param.default, (int, float)):
            params[name] = param.default
    return params


def _build_initial_conditions(
    preset: str, preset_params: dict[str, float | int], G: float
) -> tuple:
    """Call a preset factory, auto-passing G through if it accepts one.

    This matters physically, not just cosmetically: several presets (e.g.
    binary_orbit's circular-orbit velocity) use G to compute a velocity
    that's only actually a stable orbit for THAT G. If the API's dynamics G
    silently diverged from the G used to build the initial velocities, a
    caller requesting G=4.0 would get an initial condition secretly built
    for G=1.0 and it would immediately look "wrong" (e.g. an orbit that
    doesn't close) for reasons that have nothing to do with their request.

    Not every preset factory accepts a G parameter -- random_cloud,
    galaxy_merger, and star_cluster don't expose one (star_cluster's
    underlying plummer_sphere call always uses G=1.0 internally regardless
    of this API's G). That's a pre-existing property of those presets, not
    something this API layer changes -- see the README's caveats.
    """
    factory, _description = PRESET_REGISTRY[preset]
    kwargs = dict(preset_params)
    signature = inspect.signature(factory)
    if "G" in signature.parameters and "G" not in kwargs:
        kwargs["G"] = G

    try:
        positions, velocities, masses = factory(**kwargs)
    except TypeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid preset_params for '{preset}': {exc}") from exc
    return positions, velocities, masses


def _state_response(simulation_id: str, sim: Simulation) -> StateResponse:
    return StateResponse(
        simulation_id=simulation_id,
        time=sim.time,
        step_count=sim.step_count,
        n_particles=sim.positions.shape[0],
        particles=ParticleState(
            positions=sim.positions.tolist(),
            velocities=sim.velocities.tolist(),
            masses=sim.masses.tolist(),
        ),
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Backend status and available solvers/integrators."""
    return HealthResponse(
        available_solvers=list(SolverName.__args__),
        available_integrators=list(IntegratorName.__args__),
        active_simulations=manager.count(),
    )


@app.get("/presets", response_model=list[PresetInfo])
def list_presets() -> list[PresetInfo]:
    """Available initial-condition presets and their tunable parameters."""
    return [
        PresetInfo(name=name, description=description, default_params=_default_params(factory))
        for name, (factory, description) in PRESET_REGISTRY.items()
    ]


@app.post("/simulations", response_model=SimulationCreateResponse, status_code=201)
def create_simulation(request: SimulationCreateRequest) -> SimulationCreateResponse:
    """Create a new simulation from a preset + config, return its id."""
    positions, velocities, masses = _build_initial_conditions(request.preset, request.preset_params, request.G)

    config = SimulationConfig(
        dt=request.dt,
        softening=request.softening,
        G=request.G,
        integrator=request.integrator,
        solver=request.solver,
        theta=request.theta,
        seed=request.seed,
        enable_collisions=request.enable_collisions,
        merge_distance=request.merge_distance,
        preset_name=request.preset,
    )

    try:
        simulation_id = manager.create(positions, velocities, masses, config)
    except AssertionError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid initial conditions: {exc}") from exc

    return SimulationCreateResponse(
        simulation_id=simulation_id,
        n_particles=positions.shape[0],
        config=SimulationConfigEcho(
            preset=request.preset,
            preset_params=request.preset_params,
            dt=request.dt,
            softening=request.softening,
            G=request.G,
            integrator=request.integrator,
            solver=request.solver,
            theta=request.theta,
            seed=request.seed,
            enable_collisions=request.enable_collisions,
            merge_distance=request.merge_distance,
        ),
    )


@app.post("/simulations/{simulation_id}/step", response_model=StateResponse)
def step_simulation(simulation_id: str, request: StepRequest | None = None) -> StateResponse:
    """Advance a simulation by one or more steps, returning the resulting state.

    `request` is optional (defaults to a single step) so a caller can POST
    with an empty body for the common "just advance one frame" case.
    """
    n_steps = request.n_steps if request is not None else 1
    try:
        sim = manager.step(simulation_id, n_steps)
    except SimulationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _state_response(simulation_id, sim)


@app.get("/simulations/{simulation_id}/state", response_model=StateResponse)
def get_state(simulation_id: str) -> StateResponse:
    """Current particle positions, velocities, masses, and time."""
    try:
        sim = manager.get(simulation_id)
    except SimulationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _state_response(simulation_id, sim)


@app.get("/simulations/{simulation_id}/diagnostics", response_model=DiagnosticsResponse)
def get_diagnostics(simulation_id: str) -> DiagnosticsResponse:
    """Current energy, momentum, angular momentum, and center-of-mass diagnostics."""
    try:
        sim = manager.get(simulation_id)
    except SimulationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    diagnostics = compute_live_diagnostics(sim)
    return DiagnosticsResponse(simulation_id=simulation_id, **diagnostics)


@app.delete("/simulations/{simulation_id}", response_model=DeleteResponse)
def delete_simulation(simulation_id: str) -> DeleteResponse:
    """Remove a simulation from memory."""
    try:
        manager.delete(simulation_id)
    except SimulationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return DeleteResponse(simulation_id=simulation_id, deleted=True)
