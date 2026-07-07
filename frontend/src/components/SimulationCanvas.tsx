/**
 * The full 3D stage. This component's own React body is intentionally
 * tiny and static -- all the per-frame work happens inside the scene/
 * subcomponents via useFrame, so this file re-rendering (which it almost
 * never needs to) would be cheap even if it did.
 */
import { Canvas } from '@react-three/fiber'
import { EffectComposer, Bloom, Vignette } from '@react-three/postprocessing'
import { SceneBackground } from './scene/SceneBackground'
import { Particles } from './scene/Particles'
import { ParticleTrails } from './scene/ParticleTrails'
import { ComMarker } from './scene/ComMarker'
import { VelocityVectors } from './scene/VelocityVectors'
import { CameraRig } from './scene/CameraRig'

export function SimulationCanvas() {
  return (
    <Canvas
      camera={{ position: [0, 3, 9], fov: 50, near: 0.01, far: 2000 }}
      gl={{ antialias: true, powerPreference: 'high-performance' }}
      dpr={[1, 2]}
    >
      <SceneBackground />
      <Particles />
      <ParticleTrails />
      <VelocityVectors />
      <ComMarker />
      <CameraRig />
      <EffectComposer multisampling={0}>
        <Bloom luminanceThreshold={0.15} luminanceSmoothing={0.9} intensity={1.4} mipmapBlur radius={0.8} />
        <Vignette eskil={false} offset={0.15} darkness={0.85} />
      </EffectComposer>
    </Canvas>
  )
}
