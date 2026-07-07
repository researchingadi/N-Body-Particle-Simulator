import { useSimulationStore } from '../store/simulationStore'

export function TopBar() {
  const health = useSimulationStore((s) => s.health)
  const backendError = useSimulationStore((s) => s.backendError)
  const simulationId = useSimulationStore((s) => s.simulationId)
  const isRunning = useSimulationStore((s) => s.isRunning)

  const connected = health !== null && backendError === null

  return (
    <div className="hud-panel pointer-events-auto flex items-center gap-4 px-4 py-2.5">
      <div>
        <h1 className="text-sm font-semibold tracking-wide">NEURAL GRAVITY LAB</h1>
        <p className="text-[10px] text-[var(--text-faint)] font-mono uppercase tracking-wider -mt-0.5">
          N-body simulation platform
        </p>
      </div>
      <div className="h-6 w-px bg-[var(--panel-border)]" />
      <div className="flex items-center gap-1.5">
        <span
          className={`h-1.5 w-1.5 rounded-full ${connected ? 'bg-[var(--phosphor)]' : 'bg-[var(--danger)]'}`}
          style={{ boxShadow: connected ? '0 0 6px var(--phosphor)' : '0 0 6px var(--danger)' }}
        />
        <span className="hud-label">{connected ? 'Backend connected' : 'Backend unreachable'}</span>
      </div>
      {simulationId && (
        <>
          <div className="h-6 w-px bg-[var(--panel-border)]" />
          <div className="flex items-center gap-1.5">
            <span
              className={`h-1.5 w-1.5 rounded-full ${isRunning ? 'bg-[var(--signal-amber)] animate-pulse' : 'bg-[var(--text-faint)]'}`}
            />
            <span className="hud-label">{isRunning ? 'Running' : 'Paused'}</span>
          </div>
        </>
      )}
      {backendError && <span className="text-xs text-[var(--danger)] ml-auto">{backendError}</span>}
    </div>
  )
}
