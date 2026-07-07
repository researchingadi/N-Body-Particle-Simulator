/**
 * Static deep-space dressing: a distant starfield and subtle depth fog.
 * Purely atmospheric -- never mistaken for simulation data.
 */
import { Stars } from '@react-three/drei'
import { useSimulationStore } from '../../store/simulationStore'

export function SceneBackground() {
  const showGrid = useSimulationStore((s) => s.showGrid)
  return (
    <>
      <color attach="background" args={['#05060a']} />
      <fog attach="fog" args={['#05060a', 25, 140]} />
      <Stars radius={120} depth={60} count={3500} factor={3} saturation={0} fade speed={0.3} />
      {showGrid && (
        <gridHelper args={[40, 40, '#1c2230', '#12161f']} position={[0, -6, 0]} />
      )}
      <ambientLight intensity={0.15} />
    </>
  )
}
