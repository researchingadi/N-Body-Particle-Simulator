/**
 * A small crosshair-like marker at the current center of mass.
 * Position is updated imperatively in useFrame (not via React state) since
 * it moves every time new diagnostics arrive, same discipline as Particles.
 */
import { useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { useSimulationStore } from '../../store/simulationStore'

export function ComMarker() {
  const showComMarker = useSimulationStore((s) => s.showComMarker)
  const groupRef = useRef<THREE.Group>(null)

  useFrame(({ clock }) => {
    const group = groupRef.current
    if (!group) return
    const { diagnostics } = useSimulationStore.getState()
    if (!diagnostics) return
    const [x, y, z] = diagnostics.center_of_mass
    group.position.set(x, y, z)
    const pulse = 1 + 0.15 * Math.sin(clock.elapsedTime * 2)
    group.scale.setScalar(pulse)
  })

  if (!showComMarker) return null

  return (
    <group ref={groupRef}>
      <mesh>
        <ringGeometry args={[0.12, 0.15, 32]} />
        <meshBasicMaterial color="#e8a94c" transparent opacity={0.85} side={THREE.DoubleSide} />
      </mesh>
      <mesh rotation={[Math.PI / 2, 0, 0]}>
        <ringGeometry args={[0.12, 0.15, 32]} />
        <meshBasicMaterial color="#e8a94c" transparent opacity={0.45} side={THREE.DoubleSide} />
      </mesh>
    </group>
  )
}
