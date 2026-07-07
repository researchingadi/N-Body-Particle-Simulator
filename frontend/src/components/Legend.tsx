import { useSimulationStore } from '../store/simulationStore'

const ENCODING_COPY: Record<string, { low: string; high: string; label: string }> = {
  mass: { low: 'Low mass', high: 'High mass', label: 'Color encodes mass' },
  velocity: { low: 'Slow', high: 'Fast', label: 'Color encodes speed' },
  uniform: { low: '', high: '', label: 'Uniform color' },
}

export function Legend() {
  const colorEncoding = useSimulationStore((s) => s.colorEncoding)
  const cameraMode = useSimulationStore((s) => s.cameraMode)
  const selectedParticleIndex = useSimulationStore((s) => s.selectedParticleIndex)
  const copy = ENCODING_COPY[colorEncoding]

  const gradientClass =
    colorEncoding === 'velocity'
      ? 'bg-gradient-to-r from-[#7fe7c4] to-[#a78bfa]'
      : colorEncoding === 'mass'
        ? 'bg-gradient-to-r from-[#3d7a68] to-[#ffd699]'
        : 'bg-[#7fe7c4]'

  return (
    <div className="hud-panel px-3.5 py-2.5 pointer-events-auto space-y-1.5 w-64">
      {colorEncoding !== 'uniform' && (
        <div>
          <div className="hud-label mb-1">{copy.label}</div>
          <div className={`h-1.5 rounded-full ${gradientClass}`} />
          <div className="flex justify-between text-[10px] text-[var(--text-dim)] mt-0.5 font-mono">
            <span>{copy.low}</span>
            <span>{copy.high}</span>
          </div>
        </div>
      )}
      <div className="text-[11px] text-[var(--text-dim)] leading-relaxed pt-1 border-t border-[var(--panel-border)] mt-1.5">
        Drag to orbit &middot; scroll to zoom &middot; click a body to select it
        {selectedParticleIndex !== null && (
          <span className="block text-[var(--phosphor)] mt-0.5">Selected particle #{selectedParticleIndex}</span>
        )}
      </div>
      <div className="text-[10px] text-[var(--text-faint)] font-mono uppercase tracking-wider">
        Camera: {cameraMode.replace('-', ' ')}
      </div>
    </div>
  )
}
