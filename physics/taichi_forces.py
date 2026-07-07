"""GPU-accelerated (or CPU-parallel, if no GPU) direct O(N^2) gravity via Taichi.

Direct summation is O(N^2) arithmetic no matter how it's implemented -- what
changes here is *wall-clock* performance, not the algorithm: every particle
pair still gets computed, but each particle's force sum runs on its own
parallel thread (a GPU thread if a real GPU backend is active, a CPU thread
otherwise) instead of NumPy's single-threaded (if vectorized) loop. This
module is an additional backend, not a replacement: physics.forces remains
the validated numerical ground truth every other backend (this one,
Barnes-Hut) is checked against.

GPU/CPU backend selection is more delicate than it looks. Taichi's own
adaptive arch selection has been observed, empirically, on this project's
sandboxed environment, to SEGFAULT the whole process when probing a GPU
backend that has a partial/broken userspace driver present (specifically: a
Vulkan loader library exists but no working ICD behind it) -- a crash a
plain Python try/except cannot catch. `init_taichi` below therefore probes
each candidate GPU backend in an isolated subprocess first, and only
initializes Taichi in the current process once a probe has confirmed that
backend is both crash-safe AND genuinely active (not a silent fallback to
CPU, which Taichi does for cuda/vulkan/metal requests without comment).

NOTE: this file deliberately does NOT use `from __future__ import annotations`
(unlike the rest of this project). Taichi's `@ti.kernel` decorator inspects
argument annotations at decoration time and needs the actual evaluated
`ti.types.ndarray(...)` object -- PEP 563's lazy/stringified annotations
break this with a `TaichiSyntaxError: Invalid type annotation`, confirmed
empirically while building this module.
"""

import subprocess
import sys

import numpy as np

try:
    import taichi as ti
except ImportError:
    ti = None  # type: ignore[assignment]

_TAICHI_INITIALIZED = False
_TAICHI_ARCH_NAME = "not_initialized"

# Candidate GPU backends to probe, in priority order. OpenGL is deliberately
# excluded: in testing on this project it crashed even in the isolated
# subprocess probe, unlike cuda/vulkan/metal which fail softly (silent CPU
# fallback) when unavailable.
_GPU_ARCH_CANDIDATES = ("cuda", "vulkan", "metal")

_PROBE_SCRIPT_TEMPLATE = """
import sys
try:
    import taichi as ti
    ti.init(arch=ti.{arch_name})
    print(str(ti.lang.impl.current_cfg().arch))
    sys.exit(0)
except Exception:
    sys.exit(1)
"""


def _probe_arch_is_real(arch_name: str, timeout: float = 25.0) -> bool:
    """Check, in an isolated subprocess, whether `arch_name` is a genuine,
    working GPU backend here -- not just "didn't raise an exception".

    Two things this guards against, both observed empirically rather than
    hypothetically: (1) Taichi silently falling back to a CPU arch when the
    requested GPU backend isn't really usable, which a plain try/except
    can't detect since no exception is raised; (2) Taichi hard-crashing the
    process (segfault, not a Python exception) while probing certain broken
    GPU backends, which only subprocess isolation can contain.
    """
    script = _PROBE_SCRIPT_TEMPLATE.format(arch_name=arch_name)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script], timeout=timeout, capture_output=True, text=True
        )
    except subprocess.TimeoutExpired:
        return False
    if proc.returncode != 0:
        return False
    actual_arch = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    return arch_name in actual_arch.lower()


def init_taichi(prefer_gpu: bool = True) -> str:
    """Initialize the Taichi runtime once per process.

    Idempotent: Taichi doesn't support cheap re-initialization mid-process,
    so the first call's outcome is cached and simply returned on any
    subsequent call (even if `prefer_gpu` differs -- switching backends
    requires a fresh process).

    Args:
        prefer_gpu: if True (default), probe for a genuine GPU backend
            before falling back to CPU. If False, initialize directly on
            CPU (useful for a fast, deterministic test/benchmark baseline).

    Returns:
        "cuda", "vulkan", or "metal" if a genuine GPU backend is active;
        "cpu" if Taichi initialized on its CPU backend (either because no
        GPU was found, or `prefer_gpu=False`); "unavailable" if the
        `taichi` package itself isn't installed.
    """
    global _TAICHI_INITIALIZED, _TAICHI_ARCH_NAME

    if ti is None:
        _TAICHI_ARCH_NAME = "unavailable"
        return _TAICHI_ARCH_NAME

    if _TAICHI_INITIALIZED:
        return _TAICHI_ARCH_NAME

    chosen_arch = None
    if prefer_gpu:
        for arch_name in _GPU_ARCH_CANDIDATES:
            if _probe_arch_is_real(arch_name):
                chosen_arch = arch_name
                break

    if chosen_arch is not None:
        ti.init(arch=getattr(ti, chosen_arch), default_fp=ti.f64)
        _TAICHI_ARCH_NAME = chosen_arch
    else:
        # Known crash-safe in isolation (confirmed by direct testing on this
        # project): requesting ti.cpu directly, without any prior GPU arch
        # probing in-process, does not trigger the segfault described above.
        ti.init(arch=ti.cpu, default_fp=ti.f64)
        _TAICHI_ARCH_NAME = "cpu"

    _TAICHI_INITIALIZED = True
    return _TAICHI_ARCH_NAME


def taichi_backend_info() -> dict:
    """Report current Taichi availability/arch -- for logging, diagnostics,
    and test assertions (e.g. "the fallback path was exercised, not skipped").
    """
    return {
        "taichi_installed": ti is not None,
        "initialized": _TAICHI_INITIALIZED,
        "arch": _TAICHI_ARCH_NAME,
    }


_direct_accel_kernel = None  # built lazily by _ensure_kernel_built(), after init


def _ensure_kernel_built() -> None:
    """Define the Taichi kernel, but only after `ti.init()` has run.

    This has to be lazy, not a module-level `@ti.kernel` function: Taichi
    requires `ti.init()` to have already been called before a kernel is even
    *defined* (not just before it's invoked) -- defining one earlier raises
    a TaichiSyntaxError at import time. Since this project also wants
    `init_taichi()` itself to be lazy (only initializing Taichi if/when a
    Taichi-backed solver is actually used, not merely imported), the kernel
    has to be built lazily too, after the first successful `init_taichi()`.
    """
    global _direct_accel_kernel
    if _direct_accel_kernel is not None:
        return

    @ti.kernel
    def _kernel(
        positions: ti.types.ndarray(dtype=ti.f64, ndim=2),
        masses: ti.types.ndarray(dtype=ti.f64, ndim=1),
        accel: ti.types.ndarray(dtype=ti.f64, ndim=2),
        G: ti.f64,
        softening2: ti.f64,
        n: ti.i32,
    ):
        """Direct O(N^2) summation, one parallel thread per particle i.

        The outer loop over `i` is automatically parallelized by Taichi
        across whatever hardware is active (GPU threads for a real GPU
        backend, CPU threads otherwise); the inner loop over j runs
        sequentially within each thread. Same Newton + Plummer-softening
        formula as physics.forces.compute_accelerations:

            a_i = G * sum_{j != i} m_j * (r_j - r_i) / (|r_j - r_i|^2 + eps^2)^(3/2)
        """
        for i in range(n):
            ax, ay, az = 0.0, 0.0, 0.0
            for j in range(n):
                if i != j:
                    dx = positions[j, 0] - positions[i, 0]
                    dy = positions[j, 1] - positions[i, 1]
                    dz = positions[j, 2] - positions[i, 2]
                    dist2 = dx * dx + dy * dy + dz * dz + softening2
                    inv_dist3 = dist2 ** (-1.5)
                    factor = G * masses[j] * inv_dist3
                    ax += factor * dx
                    ay += factor * dy
                    az += factor * dz
            accel[i, 0] = ax
            accel[i, 1] = ay
            accel[i, 2] = az

    _direct_accel_kernel = _kernel


def taichi_direct_accelerations(
    positions: np.ndarray,
    masses: np.ndarray,
    softening: float,
    G: float = 1.0,
) -> np.ndarray:
    """Direct O(N^2) gravitational acceleration, evaluated via Taichi.

    Same contract as physics.forces.compute_accelerations: same argument
    order (positions, masses, softening, G), same Plummer softening
    convention, same (N, 3) in/out shapes -- a drop-in accel_fn anywhere the
    NumPy direct solver is used, including simulation.engine.Simulation
    (solver="taichi_direct").

    Unlike Barnes-Hut, this is NOT an approximation of the direct solver --
    it computes the exact same O(N^2) sum, just distributed across parallel
    threads instead of NumPy's vectorized loop. Differences from
    physics.forces.compute_accelerations should be floating-point
    summation-order noise only (~1e-14 relative in testing), never a
    systematic error -- see tests/test_taichi_forces.py.

    Args:
        positions: (N, 3) array of positions.
        masses: (N,) array of masses.
        softening: Plummer softening length epsilon.
        G: gravitational constant.

    Returns:
        (N, 3) array of accelerations.

    Raises:
        RuntimeError: if the `taichi` package is not installed.
    """
    n = positions.shape[0]
    if n == 0:
        return np.zeros((0, 3))
    if n == 1:
        return np.zeros((1, 3))

    arch = init_taichi()
    if arch == "unavailable":
        raise RuntimeError(
            "Taichi is not installed. Install with `pip install taichi` to use "
            "solver='taichi_direct', or use solver='direct' / 'barnes_hut' instead."
        )
    _ensure_kernel_built()

    positions_f64 = np.ascontiguousarray(positions, dtype=np.float64)
    masses_f64 = np.ascontiguousarray(masses, dtype=np.float64)
    accel = np.zeros((n, 3), dtype=np.float64)

    _direct_accel_kernel(positions_f64, masses_f64, accel, float(G), float(softening) ** 2, n)
    return accel
