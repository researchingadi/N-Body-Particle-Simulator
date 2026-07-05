"""Scientific validation: convergence sweeps over dt and softening.

This package exists to answer one question before Barnes-Hut, GPU, or a
frontend get built on top of the engine: does the solver behave the way
the numerics say it should? Specifically: does relative energy error shrink
as dt shrinks, and does leapfrog stay bounded/oscillatory rather than
drifting secularly the way Euler does?
"""
