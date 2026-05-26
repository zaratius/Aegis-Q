"""
Strong-convergence test for the Milstein integrator (Section IV-H).

A reference path is integrated at a fine step; coarser steps reuse the same
Brownian path by summing increments.  The strong error E||x_T - x_T^ref||
is fitted on a log-log scale.  Milstein should show slope ~ 1, Euler ~ 1/2.
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from quantumdbc import qubit, to_bloch, milstein_step, euler_step


def strong_error(step_fn, sys_, x0, u, g, T, dt_fine, levels, n_paths, seed):
    """Return (dt array, mean strong error array) for the given stepper."""
    rng = np.random.default_rng(seed)
    n_fine = int(round(T / dt_fine))
    dts = dt_fine * (2 ** np.arange(1, levels + 1))
    errs = np.zeros(levels)

    for _ in range(n_paths):
        dW_fine = rng.normal(0.0, np.sqrt(dt_fine), size=n_fine)
        # reference: Milstein at the fine step
        xr = x0.copy()
        for k in range(n_fine):
            xr = milstein_step(sys_, xr, u, g, dt_fine, dW_fine[k])
        for li, L in enumerate(range(1, levels + 1)):
            stride = 2 ** L
            dt = dt_fine * stride
            n = n_fine // stride
            dW = dW_fine[:n * stride].reshape(n, stride).sum(axis=1)
            x = x0.copy()
            for k in range(n):
                x = step_fn(sys_, x, u, g, dt, dW[k])
            errs[li] += np.linalg.norm(x - xr)
    return dts, errs / n_paths


def test_strong_order():
    sys_ = qubit()
    # mixed initial state inside the Bloch ball
    rho0 = np.array([[0.7, 0.2 + 0.1j], [0.2 - 0.1j, 0.3]], dtype=complex)
    x0 = to_bloch(sys_, rho0)
    u, g = np.array([0.4]), np.array([0.6])
    T, dt_fine, levels, n_paths = 0.5, 0.5 / 4096, 4, 48

    dts_m, em = strong_error(milstein_step, sys_, x0, u, g,
                             T, dt_fine, levels, n_paths, seed=20)
    dts_e, ee = strong_error(euler_step, sys_, x0, u, g,
                             T, dt_fine, levels, n_paths, seed=20)

    slope_m = np.polyfit(np.log(dts_m), np.log(em), 1)[0]
    slope_e = np.polyfit(np.log(dts_e), np.log(ee), 1)[0]

    print(f"  Milstein strong errors: {np.array2string(em, precision=2)}")
    print(f"  Euler    strong errors: {np.array2string(ee, precision=2)}")
    print(f"  [ok] Milstein strong-convergence slope {slope_m:.2f} (expect ~1)")
    print(f"  [ok] Euler    strong-convergence slope {slope_e:.2f} (expect ~0.5)")
    assert slope_m > 0.85, f"Milstein slope {slope_m:.2f} below 0.85"
    assert slope_e < 0.75, f"Euler slope {slope_e:.2f} not sub-linear"


if __name__ == "__main__":
    print("integrator verification")
    test_strong_order()
    print("integrator test passed")
