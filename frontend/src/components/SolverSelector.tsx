import { useSimulationStore } from '../store/simulationStore'
import type { IntegratorName, SolverName } from '../types/simulation'

const SOLVER_LABELS: Record<SolverName, string> = {
  direct: 'Direct O(N²)',
  barnes_hut: 'Barnes-Hut O(N log N)',
  taichi_direct: 'Taichi Direct (GPU/CPU)',
}

const INTEGRATOR_LABELS: Record<IntegratorName, string> = {
  leapfrog: 'Leapfrog (KDK)',
  velocity_verlet: 'Velocity Verlet',
  rk4: 'RK4',
  euler: 'Euler (unstable baseline)',
}

function NumberField({
  label,
  value,
  onChange,
  step,
  min,
}: {
  label: string
  value: number
  onChange: (v: number) => void
  step?: string
  min?: number
}) {
  return (
    <label className="text-xs text-[var(--text-dim)]">
      {label}
      <input
        type="number"
        className="mt-0.5 w-full rounded bg-[var(--panel-raised)] border border-[var(--panel-border)] px-1.5 py-1 text-xs text-[var(--text-primary)] font-mono"
        value={value}
        step={step ?? 'any'}
        min={min}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </label>
  )
}

export function SolverSelector() {
  const health = useSimulationStore((s) => s.health)
  const config = useSimulationStore((s) => s.config)
  const setConfigField = useSimulationStore((s) => s.setConfigField)

  const solvers = health?.available_solvers ?? ['direct', 'barnes_hut', 'taichi_direct']
  const integrators = health?.available_integrators ?? ['leapfrog', 'velocity_verlet', 'rk4', 'euler']

  return (
    <div className="space-y-2">
      <div>
        <label className="hud-label block mb-1">Solver</label>
        <select
          className="w-full rounded-md bg-[var(--panel-raised)] border border-[var(--panel-border)] px-2 py-1.5 text-sm text-[var(--text-primary)]"
          value={config.solver}
          onChange={(e) => setConfigField('solver', e.target.value as SolverName)}
        >
          {solvers.map((s) => (
            <option key={s} value={s}>
              {SOLVER_LABELS[s] ?? s}
            </option>
          ))}
        </select>
      </div>

      <div>
        <label className="hud-label block mb-1">Integrator</label>
        <select
          className="w-full rounded-md bg-[var(--panel-raised)] border border-[var(--panel-border)] px-2 py-1.5 text-sm text-[var(--text-primary)]"
          value={config.integrator}
          onChange={(e) => setConfigField('integrator', e.target.value as IntegratorName)}
        >
          {integrators.map((i) => (
            <option key={i} value={i}>
              {INTEGRATOR_LABELS[i] ?? i}
            </option>
          ))}
        </select>
      </div>

      <div className="grid grid-cols-2 gap-2">
        <NumberField label="dt" value={config.dt} step="0.001" onChange={(v) => setConfigField('dt', v)} />
        <NumberField
          label="softening"
          value={config.softening}
          step="0.001"
          onChange={(v) => setConfigField('softening', v)}
        />
        {config.solver === 'barnes_hut' && (
          <NumberField
            label="theta"
            value={config.theta}
            step="0.05"
            onChange={(v) => setConfigField('theta', v)}
          />
        )}
        <NumberField label="G" value={config.G} step="0.1" onChange={(v) => setConfigField('G', v)} />
      </div>
    </div>
  )
}
