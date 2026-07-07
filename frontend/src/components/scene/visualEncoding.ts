/**
 * Shared visual-encoding helpers: how mass/velocity map to on-screen size
 * and color. Kept separate from the rendering components so the mapping
 * logic is testable/tunable in one place.
 */
import * as THREE from 'three'
import type { ColorEncoding } from '../../store/simulationStore'

export const BASE_PARTICLE_SIZE = 0.09
export const MIN_PARTICLE_SCALE = 0.35
export const MAX_PARTICLE_SCALE = 8.0

/**
 * Mass -> render scale. Real N-body mass ratios can span 2-4 orders of
 * magnitude (a central "sun" vs. a test particle), which would make a
 * linear or even cube-root mapping either invisible or absurdly dominant.
 * A fourth-root compresses that range gently: big bodies still read as
 * visually dominant (a real requirement from the brief), without a central
 * mass eating the whole scene.
 */
export function massToScale(mass: number, medianMass: number): number {
  const ratio = medianMass > 0 ? mass / medianMass : 1
  const raw = Math.pow(Math.max(ratio, 1e-6), 0.25)
  return THREE.MathUtils.clamp(raw, MIN_PARTICLE_SCALE, MAX_PARTICLE_SCALE) * BASE_PARTICLE_SIZE
}

const MASS_COLOR_LOW = new THREE.Color('#3d7a68') // dim phosphor
const MASS_COLOR_HIGH = new THREE.Color('#ffd699') // hot amber-white
const VELOCITY_COLOR_LOW = new THREE.Color('#7fe7c4') // phosphor teal (slow)
const VELOCITY_COLOR_HIGH = new THREE.Color('#a78bfa') // violet (fast)
const UNIFORM_COLOR = new THREE.Color('#7fe7c4')

const _tmpColor = new THREE.Color()

export function encodeColor(
  encoding: ColorEncoding,
  mass: number,
  medianMass: number,
  speed: number,
  maxSpeed: number,
  out: THREE.Color = _tmpColor,
): THREE.Color {
  if (encoding === 'uniform') {
    return out.copy(UNIFORM_COLOR)
  }
  if (encoding === 'velocity') {
    const t = maxSpeed > 0 ? THREE.MathUtils.clamp(speed / maxSpeed, 0, 1) : 0
    return out.copy(VELOCITY_COLOR_LOW).lerp(VELOCITY_COLOR_HIGH, t)
  }
  // mass
  const ratio = medianMass > 0 ? mass / medianMass : 1
  const t = THREE.MathUtils.clamp(Math.log10(1 + ratio) / 2.5, 0, 1)
  return out.copy(MASS_COLOR_LOW).lerp(MASS_COLOR_HIGH, t)
}

export function computeMedian(values: number[]): number {
  if (values.length === 0) return 1
  const sorted = [...values].sort((a, b) => a - b)
  const mid = Math.floor(sorted.length / 2)
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid]
}
