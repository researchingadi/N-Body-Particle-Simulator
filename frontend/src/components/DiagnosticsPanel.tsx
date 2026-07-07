import { Line, LineChart, ResponsiveContainer, YAxis } from 'recharts'
import { useSimulationStore } from '../store/simulationStore'

function vectorMagnitude([x, y, z]: [number, number, number]): number {
  return Math.sqrt(x * x + y * y + z * z)
}

function Readout({ label, value, danger }: { label: string; value: string; danger?: boolean }) {
  return (
    <div className="flex items-baseline justify-between gap-3">
      <span className="hud-label">{label}</span>
      <span className={`hud-value text-sm ${danger ? 'hud-value--danger' : 'hud-value--phosphor'}`}>{value}</span>
    </div>
  )
}

export function DiagnosticsPanel() {
  const simulationId = useSimulationStore((s) => s.simulationId)
  const nParticles = useSimulationStore((s) => s.nParticles)
  const diagnostics = useSimulationStore((s) => s.diagnostics)
  const diagnosticsHistory = useSimulationStore((s) => s.diagnosticsHistory)
  const config = useSimulationStore((s) => s.config)
  const latestState = useSimulationStore((s) => s.latestState)

  if (!simulationId || !diagnostics) {
    return (
      <div className="hud-panel w-72 p-3.5 pointer-events-auto">
        <h2 className="hud-label mb-1">Diagnostics</h2>
        <p className="text-xs text-[var(--text-faint)]">No active simulation.</p>
      </div>
    )
  }

  const drift = diagnosticsHistory.at(-1)?.relativeDrift ?? 0
  const driftIsLarge = Math.abs(drift) > 0.01
  const angMomMag = vectorMagnitude(diagnostics.angular_momentum)
  const com = diagnostics.center_of_mass

  return (
    <div className="hud-panel w-72 p-3.5 space-y-3 pointer-events-auto">
      <div className="flex items-center justify-between">
        <h2 className="hud-label">Diagnostics</h2>
        <span className="text-[10px] font-mono text-[var(--phosphor)] uppercase tracking-wider">
          {config.solver.replace('_', ' ')}
        </span>
      </div>

      <div>
        <div className="hud-label mb-0.5">Total energy</div>
        <div className="hud-value hud-value--phosphor text-xl">{diagnostics.total_energy.toExponential(4)}</div>
      </div>

      <div className="h-12 -mx-1">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={diagnosticsHistory}>
            <YAxis domain={['auto', 'auto']} hide />
            <Line
              type="monotone"
              dataKey="relativeDrift"
              stroke="var(--phosphor)"
              strokeWidth={1.5}
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <Readout label="Relative energy drift" value={drift.toExponential(3)} danger={driftIsLarge} />

      <div className="border-t border-[var(--panel-border)] pt-2 space-y-1.5">
        <Readout label="Kinetic energy" value={diagnostics.kinetic_energy.toExponential(3)} />
        <Readout label="Potential energy" value={diagnostics.potential_energy.toExponential(3)} />
        <Readout label="|Angular momentum|" value={angMomMag.toExponential(3)} />
        <Readout label="Center of mass" value={`${com[0].toFixed(2)}, ${com[1].toFixed(2)}, ${com[2].toFixed(2)}`} />
      </div>

      <div className="border-t border-[var(--panel-border)] pt-2 space-y-1.5">
        <Readout label="Particles" value={String(nParticles)} />
        <Readout label="Time" value={diagnostics.time.toFixed(3)} />
        <Readout label="Step" value={String(latestState?.step_count ?? 0)} />
      </div>
    </div>
  )
}
