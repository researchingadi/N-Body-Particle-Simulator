/**
 * Thin HTTP client for the FastAPI backend (api/main.py).
 *
 * This file's only job is translating between fetch() and the typed
 * request/response shapes in types/simulation.ts -- it holds no simulation
 * state itself (that's the Zustand store's job) and makes no rendering
 * decisions. Every function here corresponds to exactly one backend route.
 */
import type {
  DeleteResponse,
  DiagnosticsResponse,
  HealthResponse,
  PresetInfo,
  SimulationCreateRequest,
  SimulationCreateResponse,
  StateResponse,
  StepRequest,
} from '../types/simulation'
import { ApiError } from '../types/simulation'

// Override at build time with VITE_API_BASE_URL if the backend isn't on
// the default local uvicorn port.
const API_BASE_URL: string = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      headers: { 'Content-Type': 'application/json' },
      ...init,
    })
  } catch (cause) {
    // Network-level failure (backend not running, CORS rejection, etc.)
    // -- surfaced distinctly from an HTTP error status so the UI can show
    // "backend unreachable" rather than a generic error.
    throw new ApiError(0, `Could not reach the backend at ${API_BASE_URL}. Is uvicorn running?`)
  }

  if (!response.ok) {
    let detail: unknown
    try {
      detail = (await response.json()).detail
    } catch {
      detail = response.statusText
    }
    throw new ApiError(response.status, detail)
  }

  // DELETE etc. may return an empty body in some setups; guard just in case.
  const text = await response.text()
  return text ? (JSON.parse(text) as T) : (undefined as T)
}

export const api = {
  /** GET /health */
  getHealth: (): Promise<HealthResponse> => request('/health'),

  /** GET /presets */
  getPresets: (): Promise<PresetInfo[]> => request('/presets'),

  /** POST /simulations */
  createSimulation: (body: SimulationCreateRequest): Promise<SimulationCreateResponse> =>
    request('/simulations', { method: 'POST', body: JSON.stringify(body) }),

  /** POST /simulations/{id}/step */
  stepSimulation: (simulationId: string, body: StepRequest = {}): Promise<StateResponse> =>
    request(`/simulations/${simulationId}/step`, { method: 'POST', body: JSON.stringify(body) }),

  /** GET /simulations/{id}/state */
  getState: (simulationId: string): Promise<StateResponse> =>
    request(`/simulations/${simulationId}/state`),

  /** GET /simulations/{id}/diagnostics */
  getDiagnostics: (simulationId: string): Promise<DiagnosticsResponse> =>
    request(`/simulations/${simulationId}/diagnostics`),

  /** DELETE /simulations/{id} */
  deleteSimulation: (simulationId: string): Promise<DeleteResponse> =>
    request(`/simulations/${simulationId}`, { method: 'DELETE' }),
}
