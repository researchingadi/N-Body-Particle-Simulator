/**
 * Fading orbital trails, one ring-buffered history per particle.
 *
 * Rendered as a single THREE.LineSegments draw call for ALL particles'
 * trails combined (not one Line object per particle -- that would mean
 * hundreds of draw calls for a large system). A tiny custom shader fades
 * each segment's alpha by age (older = more transparent), and additive
 * blending gives the "glowing plasma trail" look rather than flat lines.
 *
 * The ring buffer only advances when NEW simulation data arrives (detected
 * via step_count changing), not every rendered frame -- trails are a
 * history of actual physics states, not an animation of the render loop.
 */
import { useEffect, useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { useSimulationStore } from '../../store/simulationStore'
import { computeMedian, encodeColor } from './visualEncoding'

const trailVertexShader = /* glsl */ `
  attribute float aAge;
  attribute vec3 aColor;
  varying float vAge;
  varying vec3 vColor;
  void main() {
    vAge = aAge;
    vColor = aColor;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`

const trailFragmentShader = /* glsl */ `
  varying float vAge;
  varying vec3 vColor;
  void main() {
    float alpha = smoothstep(0.0, 1.0, vAge) * 0.55;
    gl_FragColor = vec4(vColor, alpha);
  }
`

export function ParticleTrails() {
  const nParticles = useSimulationStore((s) => s.nParticles)
  const trailLength = useSimulationStore((s) => s.trailLength)
  const showTrails = useSimulationStore((s) => s.showTrails)

  const lineRef = useRef<THREE.LineSegments>(null)
  const lastStepRef = useRef<number>(-1)

  // Ring buffer: [particle][slot][xyz], chronological order (0=oldest).
  const history = useMemo(
    () => new Float32Array(Math.max(nParticles, 1) * Math.max(trailLength, 2) * 3),
    [nParticles, trailLength],
  )
  const filledRef = useRef(0)

  const segmentsPerParticle = Math.max(trailLength - 1, 1)
  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry()
    const vertexCount = Math.max(nParticles, 1) * segmentsPerParticle * 2
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(vertexCount * 3), 3))
    geo.setAttribute('aAge', new THREE.BufferAttribute(new Float32Array(vertexCount), 1))
    geo.setAttribute('aColor', new THREE.BufferAttribute(new Float32Array(vertexCount * 3), 3))
    return geo
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nParticles, trailLength])

  const material = useMemo(
    () =>
      new THREE.ShaderMaterial({
        vertexShader: trailVertexShader,
        fragmentShader: trailFragmentShader,
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      }),
    [],
  )

  // Reset the buffer whenever a new simulation starts (different N) or trail length changes.
  useEffect(() => {
    filledRef.current = 0
    lastStepRef.current = -1
  }, [nParticles, trailLength])

  useFrame(() => {
    if (!showTrails) return
    const { latestState, colorEncoding } = useSimulationStore.getState()
    const particles = latestState?.particles
    const stepCount = latestState?.step_count
    if (!particles || stepCount === undefined || stepCount === lastStepRef.current) return
    lastStepRef.current = stepCount

    const n = Math.min(particles.positions.length, nParticles)
    const medianMass = computeMedian(particles.masses)
    let maxSpeed = 0
    const speeds = new Array(n)
    for (let i = 0; i < n; i++) {
      const [vx, vy, vz] = particles.velocities[i]
      const speed = Math.sqrt(vx * vx + vy * vy + vz * vz)
      speeds[i] = speed
      if (speed > maxSpeed) maxSpeed = speed
    }

    // Advance the ring buffer: shift each particle's row left by one sample
    // (native copyWithin, not a per-element JS loop) and write the new
    // sample into the last slot.
    for (let p = 0; p < n; p++) {
      const rowStart = p * trailLength * 3
      history.copyWithin(rowStart, rowStart + 3, rowStart + trailLength * 3)
      const [x, y, z] = particles.positions[p]
      const lastSlot = rowStart + (trailLength - 1) * 3
      history[lastSlot] = x
      history[lastSlot + 1] = y
      history[lastSlot + 2] = z
    }
    filledRef.current = Math.min(filledRef.current + 1, trailLength)

    // Rebuild the segment geometry from the ring buffer.
    const posAttr = geometry.getAttribute('position') as THREE.BufferAttribute
    const ageAttr = geometry.getAttribute('aAge') as THREE.BufferAttribute
    const colorAttr = geometry.getAttribute('aColor') as THREE.BufferAttribute
    const filled = filledRef.current
    const color = new THREE.Color()

    for (let p = 0; p < n; p++) {
      encodeColor(colorEncoding, particles.masses[p], medianMass, speeds[p], maxSpeed, color)
      const rowStart = p * trailLength * 3
      const firstValidSlot = trailLength - filled
      for (let seg = 0; seg < segmentsPerParticle; seg++) {
        const vBase = (p * segmentsPerParticle + seg) * 2
        const valid = seg >= firstValidSlot && seg + 1 < trailLength
        const aIdx = rowStart + seg * 3
        const bIdx = rowStart + (seg + 1) * 3

        posAttr.setXYZ(vBase, history[aIdx], history[aIdx + 1], history[aIdx + 2])
        posAttr.setXYZ(vBase + 1, history[bIdx], history[bIdx + 1], history[bIdx + 2])

        const ageA = valid ? Math.max(0, (seg - firstValidSlot) / Math.max(filled - 1, 1)) : 0
        const ageB = valid ? Math.max(0, (seg + 1 - firstValidSlot) / Math.max(filled - 1, 1)) : 0
        ageAttr.setX(vBase, valid ? ageA : 0)
        ageAttr.setX(vBase + 1, valid ? ageB : 0)

        colorAttr.setXYZ(vBase, color.r, color.g, color.b)
        colorAttr.setXYZ(vBase + 1, color.r, color.g, color.b)
      }
    }

    posAttr.needsUpdate = true
    ageAttr.needsUpdate = true
    colorAttr.needsUpdate = true
  })

  if (!showTrails || nParticles === 0) return null

  return <lineSegments ref={lineRef} geometry={geometry} material={material} />
}
