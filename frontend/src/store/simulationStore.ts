/**
 * Central simulation state.
 *
 * PERFORMANCE-CRITICAL DESIGN NOTE: `latestState.particles` (positions/
 * velocities/masses) updates every playback tick -- potentially several
 * times a second. The Three.js scene (components/scene/Particles.tsx etc.)
 * reads this via `useSimulationStore.getState().latestState` INSIDE
 * useFrame, not via the reactive `useSimulationStore(selector)` hook. That
 * distinction is the whole point: calling `getState()` reads the current
 * value without subscribing, so a particle-buffer update does not
 * re-render the React component tree or the Canvas -- only the imperative
 * Three.js objects mutate, once per rendered frame. Components that DO use
 * the reactive hook (ControlPanel, DiagnosticsPanel, TopBar) intentionally
 * only ever read low-frequency fields (time, step_count, diagnostics
 * summaries, UI toggles) for exactly this reason.
 */
import { create } from 'zustand'
import { api } from '../api/client'
import type {
  DiagnosticsResponse,
  HealthResponse,
  IntegratorName,
  PresetInfo,
  PresetName,
  SolverName,
  StateResponse,
} from '../types/simulation'

export type CameraMode = 'orbit' | 'follow-com' | 'follow-particle' | 'flythrough'
export type ColorEncoding = 'mass' | 'velocity' | 'uniform'

export interface DiagnosticsSample {
  step: number
  time: number
  totalEnergy: number
  relativeDrift: number
}

const MAX_DIAGNOSTICS_HISTORY = 300

interface SimulationConfigState {
  preset: PresetName
  presetParams: Record<string, number>
  dt: number
  softening: number
  G: number
  integrator: IntegratorName
  solver: SolverName
  theta: number
  seed: number
  enableCollisions: boolean
  mergeDistance: number
}

interface SimulationStore {
  // --- connection / catalog ---
  health: HealthResponse | null
  presets: PresetInfo[]
  backendError: string | null
  loadCatalog: () => Promise<void>

  // --- configuration (pre-creation) ---
  config: SimulationConfigState
  setPreset: (preset: PresetName) => void
  setPresetParam: (key: string, value: number) => void
  setConfigField: <K extends keyof SimulationConfigState>(key: K, value: SimulationConfigState[K]) => void

  // --- active simulation ---
  simulationId: string | null
  nParticles: number
  latestState: StateResponse | null
  diagnostics: DiagnosticsResponse | null
  diagnosticsHistory: DiagnosticsSample[]
  initialEnergy: number | null
  isCreating: boolean
  createSimulation: () => Promise<void>
  resetSimulation: () => Promise<void>
  stepManual: (nSteps?: number) => Promise<void>

  // --- playback loop ---
  isRunning: boolean
  stepsPerBatch: number
  intervalMs: number
  _intervalHandle: number | null
  toggleRunning: () => void
  setStepsPerBatch: (n: number) => void
  setIntervalMs: (ms: number) => void

  // --- view / visualization state ---
  selectedParticleIndex: number | null
  cameraMode: CameraMode
  showTrails: boolean
  trailLength: number
  showGrid: boolean
  showVelocityVectors: boolean
  showComMarker: boolean
  colorEncoding: ColorEncoding
  setSelectedParticleIndex: (i: number | null) => void
  setCameraMode: (mode: CameraMode) => void
  toggleTrails: () => void
  setTrailLength: (n: number) => void
  toggleGrid: () => void
  toggleVelocityVectors: () => void
  toggleComMarker: () => void
  setColorEncoding: (mode: ColorEncoding) => void
}

const DEFAULT_CONFIG: SimulationConfigState = {
  preset: 'binary_orbit',
  presetParams: { separation: 2.0 },
  dt: 0.005,
  softening: 0.05,
  G: 1.0,
  integrator: 'leapfrog',
  solver: 'direct',
  theta: 0.5,
  seed: 0,
  enableCollisions: false,
  mergeDistance: 0.05,
}

async function fetchDiagnosticsAndRecord(
  simulationId: string,
  get: () => SimulationStore,
  set: (partial: Partial<SimulationStore>) => void,
) {
  const diagnostics = await api.getDiagnostics(simulationId)
  const initialEnergy = get().initialEnergy ?? diagnostics.total_energy
  const relativeDrift =
    initialEnergy !== 0 ? (diagnostics.total_energy - initialEnergy) / Math.abs(initialEnergy) : 0

  const sample: DiagnosticsSample = {
    step: get().latestState?.step_count ?? 0,
    time: diagnostics.time,
    totalEnergy: diagnostics.total_energy,
    relativeDrift,
  }
  const history = [...get().diagnosticsHistory, sample].slice(-MAX_DIAGNOSTICS_HISTORY)

  set({ diagnostics, diagnosticsHistory: history, initialEnergy })
}

export const useSimulationStore = create<SimulationStore>((set, get) => ({
  health: null,
  presets: [],
  backendError: null,

  loadCatalog: async () => {
    try {
      const [health, presets] = await Promise.all([api.getHealth(), api.getPresets()])
      set({ health, presets, backendError: null })
    } catch (err) {
      set({ backendError: err instanceof Error ? err.message : String(err) })
    }
  },

  config: DEFAULT_CONFIG,

  setPreset: (preset) => {
    const presetInfo = get().presets.find((p) => p.name === preset)
    set((s) => ({
      config: {
        ...s.config,
        preset,
        presetParams: presetInfo?.default_params ?? {},
      },
    }))
  },

  setPresetParam: (key, value) =>
    set((s) => ({ config: { ...s.config, presetParams: { ...s.config.presetParams, [key]: value } } })),

  setConfigField: (key, value) => set((s) => ({ config: { ...s.config, [key]: value } })),

  simulationId: null,
  nParticles: 0,
  latestState: null,
  diagnostics: null,
  diagnosticsHistory: [],
  initialEnergy: null,
  isCreating: false,

  createSimulation: async () => {
    const { config, simulationId } = get()
    if (simulationId) {
      try {
        await api.deleteSimulation(simulationId)
      } catch {
        // Simulation may already be gone (e.g. backend restarted) -- fine, proceed to create a new one.
      }
    }
    set({ isCreating: true, backendError: null })
    try {
      const created = await api.createSimulation({
        preset: config.preset,
        preset_params: config.presetParams,
        dt: config.dt,
        softening: config.softening,
        G: config.G,
        integrator: config.integrator,
        solver: config.solver,
        theta: config.theta,
        seed: config.seed,
        enable_collisions: config.enableCollisions,
        merge_distance: config.mergeDistance,
      })
      const state = await api.getState(created.simulation_id)
      set({
        simulationId: created.simulation_id,
        nParticles: created.n_particles,
        latestState: state,
        diagnosticsHistory: [],
        initialEnergy: null,
        isRunning: false,
        selectedParticleIndex: null,
      })
      await fetchDiagnosticsAndRecord(created.simulation_id, get, set)
    } catch (err) {
      set({ backendError: err instanceof Error ? err.message : String(err) })
    } finally {
      set({ isCreating: false })
    }
  },

  resetSimulation: async () => {
    if (get().isRunning) {
      get().toggleRunning() // stops the playback loop; no-op path below would incorrectly start it
    }
    await get().createSimulation()
  },

  stepManual: async (nSteps = 1) => {
    const { simulationId } = get()
    if (!simulationId) return
    try {
      const state = await api.stepSimulation(simulationId, { n_steps: nSteps })
      set({ latestState: state })
      await fetchDiagnosticsAndRecord(simulationId, get, set)
    } catch (err) {
      set({ backendError: err instanceof Error ? err.message : String(err), isRunning: false })
      const handle = get()._intervalHandle
      if (handle !== null) window.clearInterval(handle)
      set({ _intervalHandle: null })
    }
  },

  isRunning: false,
  stepsPerBatch: 8,
  intervalMs: 100,
  _intervalHandle: null,

  toggleRunning: () => {
    const { isRunning, _intervalHandle } = get()
    if (isRunning) {
      if (_intervalHandle !== null) window.clearInterval(_intervalHandle)
      set({ isRunning: false, _intervalHandle: null })
      return
    }
    if (!get().simulationId) return
    const handle = window.setInterval(() => {
      void get().stepManual(get().stepsPerBatch)
    }, get().intervalMs)
    set({ isRunning: true, _intervalHandle: handle })
  },

  setStepsPerBatch: (n) => set({ stepsPerBatch: Math.max(1, Math.round(n)) }),

  setIntervalMs: (ms) => {
    set({ intervalMs: Math.max(16, Math.round(ms)) })
    // Restart the loop at the new cadence if currently running.
    if (get().isRunning) {
      const handle = get()._intervalHandle
      if (handle !== null) window.clearInterval(handle)
      const newHandle = window.setInterval(() => {
        void get().stepManual(get().stepsPerBatch)
      }, get().intervalMs)
      set({ _intervalHandle: newHandle })
    }
  },

  selectedParticleIndex: null,
  cameraMode: 'orbit',
  showTrails: true,
  trailLength: 120,
  showGrid: false,
  showVelocityVectors: false,
  showComMarker: true,
  colorEncoding: 'mass',

  setSelectedParticleIndex: (i) => set({ selectedParticleIndex: i }),
  setCameraMode: (mode) => set({ cameraMode: mode }),
  toggleTrails: () => set((s) => ({ showTrails: !s.showTrails })),
  setTrailLength: (n) => set({ trailLength: Math.max(2, Math.round(n)) }),
  toggleGrid: () => set((s) => ({ showGrid: !s.showGrid })),
  toggleVelocityVectors: () => set((s) => ({ showVelocityVectors: !s.showVelocityVectors })),
  toggleComMarker: () => set((s) => ({ showComMarker: !s.showComMarker })),
  setColorEncoding: (mode) => set({ colorEncoding: mode }),
}))
