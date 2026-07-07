import { useSimulationStore } from '../store/simulationStore'
import type { PresetName } from '../types/simulation'

/** Friendlier labels than the raw API preset names. */
const PRESET_LABELS: Record<PresetName, string> = {
  binary_orbit: 'Binary Orbit',
  figure_eight: 'Figure-Eight Three-Body',
  solar_system: 'Toy Solar System',
  plummer_sphere: 'Plummer Sphere',
  star_cluster: 'Star Cluster',
  disk_galaxy: 'Disk Galaxy',
  ring_system: 'Ring System',
  galaxy_merger: 'Galaxy Merger',
  random_cloud: 'Random Cloud',
}

export function PresetSelector() {
  const presets = useSimulationStore((s) => s.presets)
  const selectedPreset = useSimulationStore((s) => s.config.preset)
  const presetParams = useSimulationStore((s) => s.config.presetParams)
  const setPreset = useSimulationStore((s) => s.setPreset)
  const setPresetParam = useSimulationStore((s) => s.setPresetParam)

  const activePreset = presets.find((p) => p.name === selectedPreset)

  return (
    <div className="space-y-2">
      <label className="hud-label block">Initial condition</label>
      <select
        className="w-full rounded-md bg-[var(--panel-raised)] border border-[var(--panel-border)] px-2 py-1.5 text-sm text-[var(--text-primary)]"
        value={selectedPreset}
        onChange={(e) => setPreset(e.target.value as PresetName)}
      >
        {presets.map((p) => (
          <option key={p.name} value={p.name}>
            {PRESET_LABELS[p.name] ?? p.name}
          </option>
        ))}
      </select>
      {activePreset && <p className="text-xs text-[var(--text-dim)] leading-snug">{activePreset.description}</p>}

      {activePreset && Object.keys(activePreset.default_params).length > 0 && (
        <div className="grid grid-cols-2 gap-2 pt-1">
          {Object.entries(activePreset.default_params).map(([key, defaultValue]) => (
            <label key={key} className="text-xs text-[var(--text-dim)]">
              {key}
              <input
                type="number"
                className="mt-0.5 w-full rounded bg-[var(--panel-raised)] border border-[var(--panel-border)] px-1.5 py-1 text-xs text-[var(--text-primary)] font-mono"
                value={presetParams[key] ?? defaultValue}
                step="any"
                onChange={(e) => setPresetParam(key, Number(e.target.value))}
              />
            </label>
          ))}
        </div>
      )}
    </div>
  )
}
