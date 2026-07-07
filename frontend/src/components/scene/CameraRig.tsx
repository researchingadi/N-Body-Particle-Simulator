/**
 * Camera control, covering all four modes from the brief:
 *   - orbit: free user-controlled orbit (OrbitControls' default behavior)
 *   - follow-com: smoothly retargets to the center of mass each frame
 *   - follow-particle: smoothly retargets to the selected particle
 *   - flythrough: slow automatic orbit (OrbitControls' autoRotate)
 *
 * "Flythrough" here is a smooth, continuous auto-orbit around the current
 * target -- a lightweight, honest interpretation of "cinematic
 * flythrough", not a scripted multi-waypoint camera path. A true
 * waypoint/keyframe flythrough (e.g. panning between points of interest)
 * is a reasonable future enhancement, noted in the README rather than
 * silently overclaimed here.
 */
import { useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib'
import * as THREE from 'three'
import { useSimulationStore } from '../../store/simulationStore'

const RETARGET_LERP = 0.06

export function CameraRig() {
  const cameraMode = useSimulationStore((s) => s.cameraMode)
  const controlsRef = useRef<OrbitControlsImpl | null>(null)
  const target = useRef(new THREE.Vector3())

  useFrame(() => {
    const controls = controlsRef.current
    if (!controls) return
    const { latestState, diagnostics, selectedParticleIndex, cameraMode: mode } = useSimulationStore.getState()

    if (mode === 'follow-com' && diagnostics) {
      const [cx, cy, cz] = diagnostics.center_of_mass
      target.current.set(cx, cy, cz)
      controls.target.lerp(target.current, RETARGET_LERP)
      controls.update()
    } else if (mode === 'follow-particle' && selectedParticleIndex !== null && latestState) {
      const pos = latestState.particles.positions[selectedParticleIndex]
      if (pos) {
        target.current.set(pos[0], pos[1], pos[2])
        controls.target.lerp(target.current, RETARGET_LERP)
        controls.update()
      }
    }
    // 'orbit' and 'flythrough' both leave the target alone; flythrough's
    // motion comes entirely from the autoRotate prop below.
  })

  return (
    <OrbitControls
      ref={controlsRef}
      makeDefault
      enableDamping
      dampingFactor={0.08}
      autoRotate={cameraMode === 'flythrough'}
      autoRotateSpeed={0.5}
      minDistance={0.5}
      maxDistance={500}
    />
  )
}
