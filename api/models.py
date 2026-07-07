"""Pydantic request/response schemas for the simulation API.

These mirror `simulation.engine.SimulationConfig` and the diagnostics
functions in `diagnostics.conservation` -- they are a serialization layer
over that existing, validated engine, not a new set of physics parameters.
`SolverName`/`IntegratorName` are imported directly from `simulation.engine`
(not re-declared) so there is exactly one place that defines which solvers
and integrators exist; FastAPI/Pydantic still generates correct OpenAPI
schema and rejects invalid values with a 422 automatically, since a
`typing.Literal` works the same way regardless of which module defines it.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from simulation.engine import IntegratorName, SolverName

PresetName = Literal[
    "binary_orbit",
    "figure_eight",
    "solar_system",
    "plummer_sphere",
    "star_cluster",
    "disk_galaxy",
    "ring_system",
    "galaxy_merger",
    "random_cloud",
]


class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: Literal["ok"] = "ok"
    available_solvers: list[SolverName]
    available_integrators: list[IntegratorName]
    active_simulations: int = Field(description="Number of simulations currently held in memory")


class PresetInfo(BaseModel):
    """Response entry for GET /presets: one preset's metadata."""

    name: PresetName
    description: str
    default_params: dict[str, float | int] = Field(
        description="Default keyword arguments for this preset's generator function"
    )


class SimulationCreateRequest(BaseModel):
    """Request body for POST /simulations.

    `preset_params` overrides the chosen preset's defaults (e.g.
    {"n": 300, "seed": 7} for plummer_sphere) -- see GET /presets for each
    preset's tunable parameters. Everything else mirrors
    `simulation.engine.SimulationConfig` field-for-field, with the same
    defaults, so behavior via the API matches direct Python use exactly.
    """

    preset: PresetName
    preset_params: dict[str, float | int] = Field(default_factory=dict)

    dt: float = 0.01
    softening: float = 0.05
    G: float = 1.0
    integrator: IntegratorName = "leapfrog"
    solver: SolverName = "direct"
    theta: float = Field(default=0.5, description="Barnes-Hut opening angle; unused for other solvers")
    seed: int = 0
    enable_collisions: bool = False
    merge_distance: float = 0.05


class SimulationConfigEcho(BaseModel):
    """Echoes the resolved config back to the caller, for reproducibility
    (same purpose as `SimulationConfig` being saved alongside script output
    in io_utils.export -- the API caller should be able to record exactly
    what produced a given run).
    """

    preset: PresetName
    preset_params: dict[str, float | int]
    dt: float
    softening: float
    G: float
    integrator: IntegratorName
    solver: SolverName
    theta: float
    seed: int
    enable_collisions: bool
    merge_distance: float


class SimulationCreateResponse(BaseModel):
    """Response for POST /simulations."""

    simulation_id: str
    n_particles: int
    config: SimulationConfigEcho


class StepRequest(BaseModel):
    """Request body for POST /simulations/{simulation_id}/step."""

    n_steps: int = Field(default=1, ge=1, le=100_000, description="Number of integration steps to advance")


class ParticleState(BaseModel):
    """Positions/velocities/masses for every particle, at one instant."""

    positions: list[list[float]] = Field(description="(N, 3) list of [x, y, z]")
    velocities: list[list[float]] = Field(description="(N, 3) list of [vx, vy, vz]")
    masses: list[float] = Field(description="(N,) list of masses")


class StateResponse(BaseModel):
    """Response for GET /simulations/{simulation_id}/state and for the
    step endpoint (which returns the state resulting from the step, so a
    caller can advance-and-render in one round trip).
    """

    simulation_id: str
    time: float
    step_count: int
    n_particles: int
    particles: ParticleState


class DiagnosticsResponse(BaseModel):
    """Response for GET /simulations/{simulation_id}/diagnostics.

    Computed live from current particle state via diagnostics.conservation
    (NOT from Simulation.history) -- see api/simulation_manager.py's module
    docstring for why the API deliberately doesn't use the history/
    trajectory-accumulating path that scripts.run_demo.py etc. use.
    """

    simulation_id: str
    time: float
    kinetic_energy: float
    potential_energy: float
    total_energy: float
    momentum: list[float] = Field(description="[px, py, pz]")
    angular_momentum: list[float] = Field(description="[Lx, Ly, Lz]")
    center_of_mass: list[float] = Field(description="[x, y, z]")


class DeleteResponse(BaseModel):
    """Response for DELETE /simulations/{simulation_id}."""

    simulation_id: str
    deleted: bool
