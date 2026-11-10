
import numpy as np
def dof3(x):
    """Return [N, E, psi] / [u, v, r] from a 3- or 6-DOF array."""
    x = np.asarray(x)
    return x[:, [0, 1, 5]] if x.shape[1] == 6 else x[:, :3]
def print_rmse(out, window_s=30.0):
    """Print RMSE (estimate - LF truth) for every run in out = sim4()."""
    for name, logs in out.items():
        if getattr(logs, "eta_est", None) is None:            
            continue
        t = np.asarray(logs.t)
        n = int(window_s / (t[1] - t[0]))  # samples in 30 s
        k = np.ones(n) / n

        x_true = np.hstack([dof3(logs.eta), dof3(logs.nu)])
        x_hat  = np.hstack([dof3(logs.eta_est), dof3(logs.nu_est)])
        x_lf   = np.column_stack([np.convolve(c, k, mode="same") for c in x_true.T])

        e = (x_hat - x_lf)[n:-n]           # drop edges
        e[:, 2] = (e[:, 2] + np.pi) % (2 * np.pi) - np.pi   # wrap heading
        r = np.sqrt(np.mean(e**2, axis=0))

        print(f"{' / '.join(name):35s}"
              f"N {r[0]:.3f} m   E {r[1]:.3f} m   psi {np.rad2deg(r[2]):.2f} deg   "
              f"u {r[3]:.3f}   v {r[4]:.3f} m/s   r {np.rad2deg(r[5]):.3f} deg/s")

