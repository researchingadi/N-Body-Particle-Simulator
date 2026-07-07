/**
 * Optional overlay: a short line per particle pointing along its current
 * velocity, scaled for legibility rather than physical accuracy. Rebuilt
 * only when new backend data arrives (same discipline as ParticleTrails).
 */
import { useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { useSimulationStore } from '../../store/simulationStore'

const VECTOR_SCALE = 0.6

export function VelocityVectors() {
  const nParticles = useSimulationStore((s) => s.nParticles)
  const showVelocityVectors = useSimulationStore((s) => s.showVelocityVectors)
  const lastStepRef = useRef(-1)

  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(Math.max(nParticles, 1) * 2 * 3), 3))
    return geo
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nParticles])

  const material = useMemo(
    () =>
      new THREE.LineBasicMaterial({
        color: '#a78bfa',
        transparent: true,
        opacity: 0.5,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      }),
    [],
  )

  useFrame(() => {
    if (!showVelocityVectors) return
    const { latestState } = useSimulationStore.getState()
    const particles = latestState?.particles
    const stepCount = latestState?.step_count
    if (!particles || stepCount === undefined || stepCount === lastStepRef.current) return
    lastStepRef.current = stepCount

    const posAttr = geometry.getAttribute('position') as THREE.BufferAttribute
    const n = Math.min(particles.positions.length, nParticles)
    for (let i = 0; i < n; i++) {
      const [x, y, z] = particles.positions[i]
      const [vx, vy, vz] = particles.velocities[i]
      posAttr.setXYZ(i * 2, x, y, z)
      posAttr.setXYZ(i * 2 + 1, x + vx * VECTOR_SCALE, y + vy * VECTOR_SCALE, z + vz * VECTOR_SCALE)
    }
    posAttr.needsUpdate = true
  })

  if (!showVelocityVectors || nParticles === 0) return null

  return <lineSegments geometry={geometry} material={material} />
}
