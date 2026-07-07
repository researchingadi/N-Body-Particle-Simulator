import { useSimulationStore } from '../store/simulationStore'
import type { CameraMode, ColorEncoding } from '../store/simulationStore'
import { PresetSelector } from './PresetSelector'
import { SolverSelector } from './SolverSelector'

const CAMERA_MODES: { value: CameraMode; label: string }[] = [
  { value: 'orbit', label: 'Free orbit' },
  { value: 'follow-com', label: 'Follow center of mass' },
  { value: 'follow-particle', label: 'Follow selected particle' },
  { value: 'flythrough', label: 'Cinematic flythrough' },
]

const COLOR_ENCODINGS: { value: ColorEncoding; label: string }[] = [
  { value: 'mass', label: 'Mass' },
  { value: 'velocity', label: 'Velocity' },
  { value: 'uniform', label: 'Uniform' },
]

function PrimaryButton({
  onClick,
  disabled,
  children,
  variant = 'default',
}: {
  onClick: () => void
  disabled?: boolean
  children: React.ReactNode
  variant?: 'default' | 'accent'
}) {
  const base =
    'flex-1 rounded-md px-3 py-2 text-sm font-medium transition-colors disabled:opacity-40 disabled:cursor-not-allowed'
  const style =
    variant === 'accent'
      ? 'bg-[var(--phosphor-dim)] text-[var(--void)] hover:bg-[var(--phosphor)]'
      : 'bg-[var(--panel-raised)] border border-[var(--panel-border)] text-[var(--text-primary)] hover:border-[var(--panel-border-bright)]'
  return (
    <button className={`${base} ${style}`} onClick={onClick} disabled={disabled}>
      {children}
    </button>
  )
}

function ToggleChip({
  active,
  onClick,
  children,
}: {
  active: boolean
  onClick: () => void
  children: React.ReactNode
}) {
  return (
    <button
      onClick={onClick}
      className={`rounded-full px-2.5 py-1 text-[11px] border transition-colors ${
        active
          ? 'border-[var(--phosphor)] text-[var(--phosphor)] bg-[color-mix(in_srgb,var(--phosphor)_12%,transparent)]'
          : 'border-[var(--panel-border)] text-[var(--text-dim)] hover:border-[var(--panel-border-bright)]'
      }`}
    >
      {children}
    </button>
  )
}

export function ControlPanel() {
  const simulationId = useSimulationStore((s) => s.simulationId)
  const isCreating = useSimulationStore((s) => s.isCreating)
  const isRunning = useSimulationStore((s) => s.isRunning)
  const createSimulation = useSimulationStore((s) => s.createSimulation)
  const resetSimulation = useSimulationStore((s) => s.resetSimulation)
  const stepManual = useSimulationStore((s) => s.stepManual)
  const toggleRunning = useSimulationStore((s) => s.toggleRunning)

  const stepsPerBatch = useSimulationStore((s) => s.stepsPerBatch)
  const setStepsPerBatch = useSimulationStore((s) => s.setStepsPerBatch)
  const intervalMs = useSimulationStore((s) => s.intervalMs)
  const setIntervalMs = useSimulationStore((s) => s.setIntervalMs)

  const cameraMode = useSimulationStore((s) => s.cameraMode)
  const setCameraMode = useSimulationStore((s) => s.setCameraMode)
  const colorEncoding = useSimulationStore((s) => s.colorEncoding)
  const setColorEncoding = useSimulationStore((s) => s.setColorEncoding)

  const showTrails = useSimulationStore((s) => s.showTrails)
  const toggleTrails = useSimulationStore((s) => s.toggleTrails)
  const trailLength = useSimulationStore((s) => s.trailLength)
  const setTrailLength = useSimulationStore((s) => s.setTrailLength)
  const showGrid = useSimulationStore((s) => s.showGrid)
  const toggleGrid = useSimulationStore((s) => s.toggleGrid)
  const showVelocityVectors = useSimulationStore((s) => s.showVelocityVectors)
  const toggleVelocityVectors = useSimulationStore((s) => s.toggleVelocityVectors)
  const showComMarker = useSimulationStore((s) => s.showComMarker)
  const toggleComMarker = useSimulationStore((s) => s.toggleComMarker)

  return (
    <div className="hud-panel hud-scroll w-72 max-h-[calc(100vh-2rem)] overflow-y-auto p-3.5 space-y-4 pointer-events-auto">
      <div>
        <h2 className="hud-label mb-2">Playback</h2>
        <div className="flex gap-2">
          {!simulationId ? (
            <PrimaryButton onClick={createSimulation} disabled={isCreating} variant="accent">
              {isCreating ? 'Creating…' : 'Create simulation'}
            </PrimaryButton>
          ) : (
            <PrimaryButton onClick={toggleRunning} variant="accent">
              {isRunning ? 'Pause' : 'Start'}
            </PrimaryButton>
          )}
        </div>
        {simulationId && (
          <div className="flex gap-2 mt-2">
            <PrimaryButton onClick={() => void stepManual(1)} disabled={isRunning}>
              Step
            </PrimaryButton>
            <PrimaryButton onClick={resetSimulation}>Reset</PrimaryButton>
          </div>
        )}

        <div className="grid grid-cols-2 gap-2 mt-3">
          <label className="text-xs text-[var(--text-dim)]">
            Steps / batch
            <input
              type="number"
              min={1}
              className="mt-0.5 w-full rounded bg-[var(--panel-raised)] border border-[var(--panel-border)] px-1.5 py-1 text-xs font-mono text-[var(--text-primary)]"
              value={stepsPerBatch}
              onChange={(e) => setStepsPerBatch(Number(e.target.value))}
            />
          </label>
          <label className="text-xs text-[var(--text-dim)]">
            Interval (ms)
            <input
              type="number"
              min={16}
              step={10}
              className="mt-0.5 w-full rounded bg-[var(--panel-raised)] border border-[var(--panel-border)] px-1.5 py-1 text-xs font-mono text-[var(--text-primary)]"
              value={intervalMs}
              onChange={(e) => setIntervalMs(Number(e.target.value))}
            />
          </label>
        </div>
      </div>

      <div className="border-t border-[var(--panel-border)] pt-3">
        <h2 className="hud-label mb-2">Initial condition</h2>
        <PresetSelector />
      </div>

      <div className="border-t border-[var(--panel-border)] pt-3">
        <h2 className="hud-label mb-2">Physics</h2>
        <SolverSelector />
      </div>

      <div className="border-t border-[var(--panel-border)] pt-3">
        <h2 className="hud-label mb-2">Camera</h2>
        <select
          className="w-full rounded-md bg-[var(--panel-raised)] border border-[var(--panel-border)] px-2 py-1.5 text-sm text-[var(--text-primary)]"
          value={cameraMode}
          onChange={(e) => setCameraMode(e.target.value as CameraMode)}
        >
          {CAMERA_MODES.map((m) => (
            <option key={m.value} value={m.value}>
              {m.label}
            </option>
          ))}
        </select>
      </div>

      <div className="border-t border-[var(--panel-border)] pt-3">
        <h2 className="hud-label mb-2">Visualization</h2>
        <div className="flex flex-wrap gap-1.5 mb-2">
          <ToggleChip active={showTrails} onClick={toggleTrails}>
            Trails
          </ToggleChip>
          <ToggleChip active={showGrid} onClick={toggleGrid}>
            Grid
          </ToggleChip>
          <ToggleChip active={showVelocityVectors} onClick={toggleVelocityVectors}>
            Velocity vectors
          </ToggleChip>
          <ToggleChip active={showComMarker} onClick={toggleComMarker}>
            COM marker
          </ToggleChip>
        </div>
        {showTrails && (
          <label className="text-xs text-[var(--text-dim)] block mb-2">
            Trail length: <span className="hud-value">{trailLength}</span>
            <input
              type="range"
              min={10}
              max={300}
              step={10}
              value={trailLength}
              onChange={(e) => setTrailLength(Number(e.target.value))}
              className="w-full mt-1 accent-[var(--phosphor)]"
            />
          </label>
        )}
        <div>
          <span className="hud-label block mb-1">Color encoding</span>
          <div className="flex gap-1.5">
            {COLOR_ENCODINGS.map((c) => (
              <ToggleChip key={c.value} active={colorEncoding === c.value} onClick={() => setColorEncoding(c.value)}>
                {c.label}
              </ToggleChip>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
