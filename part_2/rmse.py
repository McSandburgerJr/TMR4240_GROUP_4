import numpy as np

def dof3(x):
    """Return [N, E, psi] / [u, v, r] from a 3- or 6-DOF array."""
    x = np.asarray(x)
    return x[:, [0, 1, 5]] if x.shape[1] == 6 else x[:, :3]

def print_rmse(out, window_s=30.0, t_skip=30.0, t_end = None):
    """Print RMSE (estimate - LF truth) for every run in out = sim4().

    For closed-loop runs, also print the station-keeping error
    (true pose - origin) and the actuator activity.
    """
    for name, logs in out.items():                  
        if getattr(logs, "eta_est", None) is None:
            continue
        t = np.asarray(logs.t)
        h = t[1] - t[0]
        n = int(window_s / h)                                   # number of samples in 30 s
        k = np.ones(n) / n                                      

        x_true = np.hstack([dof3(logs.eta), dof3(logs.nu)])
        x_hat  = np.hstack([dof3(logs.eta_est), dof3(logs.nu_est)])
        x_lf   = np.column_stack([np.convolve(c, k, mode="same") for c in x_true.T])

        k1 = len(t) if t_end is None else int(t_end / h)        # end of window
        e = (x_hat - x_lf)[n:k1 - n]                            # drop edges
        e[:, 2] = (e[:, 2] + np.pi) % (2 * np.pi) - np.pi       # wrap heading
        r = np.sqrt(np.mean(e**2, axis=0))

        print(f"{' / '.join(name):35s}"
              f"N {r[0]:.3f} m   E {r[1]:.3f} m   psi {np.rad2deg(r[2]):.2f} deg   "
              f"u {r[3]:.3f}   v {r[4]:.3f} m/s   r {np.rad2deg(r[5]):.3f} deg/s")

        if name[0] == "closed loop":
            k0 = int(t_skip / h)                                # skip initial transient

            # Station keeping: true pose relative to the origin
            sk = dof3(logs.eta)[k0:k1].copy()
            sk[:, 2] = (sk[:, 2] + np.pi) % (2 * np.pi) - np.pi
            s = np.sqrt(np.mean(sk**2, axis=0))

            # Actuator activity: mean |thrust| and mean |thrust rate|
            u = np.asarray(logs.u)[k0:k1]
            mean_u = np.mean(np.abs(u)) / 1e3
            mean_du = np.mean(np.abs(np.diff(u, axis=0))) / h / 1e3

            print(f"{'   station keeping':35s}"
                  f"N {s[0]:.3f} m   E {s[1]:.3f} m   psi {np.rad2deg(s[2]):.2f} deg   "
                  f"mean|u| {mean_u:.1f} kN   mean|du/dt| {mean_du:.1f} kN/s")