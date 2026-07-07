/**
 * Renders every particle as a glowing instanced sphere.
 *
 * PERFORMANCE: this component subscribes to almost nothing reactively.
 * `nParticles` and `colorEncoding` are read via the normal (reactive) hook
 * because they change rarely and force a deliberate remount/recolor; the
 * actual per-frame position/velocity/mass data is read via
 * `useSimulationStore.getState()` inside `useFrame`, which does NOT
 * subscribe -- so a playback tick (many times a second) mutates the
 * instanced mesh's GPU buffers directly without ever triggering a React
 * re-render. This is the load-bearing performance decision for the whole
 * scene; see store/simulationStore.ts's module docstring for the other half.
 */
import { useRef, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { useSimulationStore } from '../../store/simulationStore'
import { computeMedian, encodeColor, massToScale } from './visualEncoding'

const _matrix = new THREE.Object3D()
const _color = new THREE.Color()

export function Particles() {
  const nParticles = useSimulationStore((s) => s.nParticles)
  const setSelectedParticleIndex = useSimulationStore((s) => s.setSelectedParticleIndex)

  const meshRef = useRef<THREE.InstancedMesh>(null)

  // Recreated only when particle count changes (a new simulation), not per frame.
  const geometry = useMemo(() => new THREE.SphereGeometry(1, 14, 14), [])
  const material = useMemo(
    () =>
      new THREE.MeshBasicMaterial({
        vertexColors: true,
        transparent: true,
        opacity: 0.95,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      }),
    [],
  )

  useFrame(() => {
    const mesh = meshRef.current
    if (!mesh) return
    const { latestState, selectedParticleIndex, colorEncoding: encoding } = useSimulationStore.getState()
    const particles = latestState?.particles
    if (!particles || particles.positions.length === 0) return

    const { positions, velocities, masses } = particles
    const medianMass = computeMedian(masses)

    let maxSpeed = 0
    const speeds: number[] = new Array(velocities.length)
    for (let i = 0; i < velocities.length; i++) {
      const [vx, vy, vz] = velocities[i]
      const speed = Math.sqrt(vx * vx + vy * vy + vz * vz)
      speeds[i] = speed
      if (speed > maxSpeed) maxSpeed = speed
    }

    const count = Math.min(positions.length, mesh.count)
    for (let i = 0; i < count; i++) {
      const [x, y, z] = positions[i]
      const scale = massToScale(masses[i], medianMass)
      _matrix.position.set(x, y, z)
      _matrix.scale.setScalar(scale)
      _matrix.updateMatrix()
      mesh.setMatrixAt(i, _matrix.matrix)

      if (i === selectedParticleIndex) {
        _color.set('#ffffff')
      } else {
        encodeColor(encoding, masses[i], medianMass, speeds[i], maxSpeed, _color)
      }
      mesh.setColorAt(i, _color)
    }

    mesh.instanceMatrix.needsUpdate = true
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true
  })

  if (nParticles === 0) return null

  return (
    <instancedMesh
      key={nParticles}
      ref={meshRef}
      args={[geometry, material, nParticles]}
      onClick={(event) => {
        event.stopPropagation()
        if (event.instanceId !== undefined) setSelectedParticleIndex(event.instanceId)
      }}
    />
  )
}
