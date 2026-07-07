/**
 * TypeScript types mirroring api/models.py exactly.
 *
 * These are a serialization contract, not a place to add computed or UI
 * state -- if a field exists here, it exists in the Pydantic response it
 * mirrors. UI-only state (selected camera mode, trail length, etc.) lives
 * in the Zustand store, not here.
 */

export type SolverName = 'direct' | 'barnes_hut' | 'taichi_direct'

export type IntegratorName = 'euler' | 'leapfrog' | 'velocity_verlet' | 'rk4'

export type PresetName =
  | 'binary_orbit'
  | 'figure_eight'
  | 'solar_system'
  | 'plummer_sphere'
  | 'star_cluster'
  | 'disk_galaxy'
  | 'ring_system'
  | 'galaxy_merger'
  | 'random_cloud'

export interface HealthResponse {
  status: 'ok'
  available_solvers: SolverName[]
  available_integrators: IntegratorName[]
  active_simulations: number
}

export interface PresetInfo {
  name: PresetName
  description: string
  default_params: Record<string, number>
}

export interface SimulationCreateRequest {
  preset: PresetName
  preset_params?: Record<string, number>
  dt?: number
  softening?: number
  G?: number
  integrator?: IntegratorName
  solver?: SolverName
  theta?: number
  seed?: number
  enable_collisions?: boolean
  merge_distance?: number
}

export interface SimulationConfigEcho {
  preset: PresetName
  preset_params: Record<string, number>
  dt: number
  softening: number
  G: number
  integrator: IntegratorName
  solver: SolverName
  theta: number
  seed: number
  enable_collisions: boolean
  merge_distance: number
}

export interface SimulationCreateResponse {
  simulation_id: string
  n_particles: number
  config: SimulationConfigEcho
}

export interface StepRequest {
  n_steps?: number
}

export interface ParticleState {
  positions: number[][]
  velocities: number[][]
  masses: number[]
}

export interface StateResponse {
  simulation_id: string
  time: number
  step_count: number
  n_particles: number
  particles: ParticleState
}

export interface DiagnosticsResponse {
  simulation_id: string
  time: number
  kinetic_energy: number
  potential_energy: number
  total_energy: number
  momentum: [number, number, number]
  angular_momentum: [number, number, number]
  center_of_mass: [number, number, number]
}

export interface DeleteResponse {
  simulation_id: string
  deleted: boolean
}

/** Thrown by api/client.ts for any non-2xx response. */
export class ApiError extends Error {
  status: number
  detail: unknown

  constructor(status: number, detail: unknown) {
    super(typeof detail === 'string' ? detail : `API error (${status})`)
    this.name = 'ApiError'
    this.status = status
    this.detail = detail
  }
}
