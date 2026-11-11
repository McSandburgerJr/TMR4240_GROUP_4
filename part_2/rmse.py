import numpy as np


def dof3(x):
    """Return [N, E, psi] / [u, v, r] from a 3- or 6-DOF array."""
    x = np.asarray(x)
    return x[:, [0, 1, 5]] if x.shape[1] == 6 else x[:, :3]


def _get_thruster_forces_kN(logs):
    """Extract the (K, n_thr) array of actual thruster forces in [kN]."""
    # Check standard thruster force attributes in Part 2 simulator logs
    for attr in ("u_thr", "thrust", "u_actual", "f_thr", "u_cmd", "u"):
        val = getattr(logs, attr, None)
        if val is not None:
            arr = np.asarray(val, dtype=float)
            if arr.ndim == 2:
                # If array has 6 columns [T1, T2, T3, alpha1, alpha2, alpha3], keep only forces [:3]
                if arr.shape[1] >= 5:
                    arr = arr[:, :3]
                # Convert N -> kN if stored in Newtons
                if np.max(np.abs(arr)) > 500.0:
                    arr = arr * 1e-3
                return arr
    return None


def print_rmse(out, window_s=30.0, t_end=None):
    """Print RMSE (Eq. 150) and closed-loop station-keeping & thruster metrics (Eq. 151-152).

    t_end: end of the evaluation window [s]. None = use the whole run.
    """

    print("\n----------------------------------- 1. OBSERVER ESTIMATION RMSE (Eq. 150) -------------------------------------------")
    for name, logs in out.items():
        if getattr(logs, "eta_est", None) is None:
            continue
        t = np.asarray(logs.t)
        dt = t[1] - t[0]
        n = int(window_s / dt)  # samples in 30 s
        k = np.ones(n) / n
        k1 = len(t) if t_end is None else int(t_end / dt)  # end of evaluation window

        x_true = np.hstack([dof3(logs.eta), dof3(logs.nu)]).copy()
        x_true[:, 2] = np.unwrap(x_true[:, 2])
        x_hat = np.hstack([dof3(logs.eta_est), dof3(logs.nu_est)])
        x_lf = np.column_stack([np.convolve(c, k, mode="same") for c in x_true.T])

        e = (x_hat - x_lf)[n:k1 - n]       # drop edges
        e[:, 2] = (e[:, 2] + np.pi) % (2 * np.pi) - np.pi   # wrap heading
        r = np.sqrt(np.mean(e**2, axis=0))

        label = " / ".join(name) if isinstance(name, tuple) else str(name)
        print(f"{label:40s} "
              f"N {r[0]:.3f} m   E {r[1]:.3f} m   psi {np.rad2deg(r[2]):.2f} deg   "
              f"u {r[3]:.3f}   v {r[4]:.3f} m/s   r {np.rad2deg(r[5]):.3f} deg/s")

    print("\n------------------------------------ 2. CLOSED-LOOP STATION-KEEPING (Eq. 151) & ACTUATOR ACTIVITY (Eq. 152) ------------------------------------")
    for name, logs in out.items():
        label = " / ".join(name) if isinstance(name, tuple) else str(name)
        if "closed loop" not in label.lower():
            continue

        t = np.asarray(logs.t)
        dt = t[1] - t[0]
        n = int(window_s / dt)  # exclude first 30 s transient
        k1 = len(t) if t_end is None else int(t_end / dt)  # end of evaluation window

        # Eq. (151): RMS of true pose relative to origin
        eta = dof3(logs.eta)[n:k1].copy()
        eta[:, 2] = (eta[:, 2] + np.pi) % (2 * np.pi) - np.pi
        rms_sk = np.sqrt(np.mean(eta**2, axis=0))

        # Eq. (152): Mean |u| and Mean |du/dt| over all thrusters
        u_kN = _get_thruster_forces_kN(logs)
        if u_kN is not None:
            u_kN = u_kN[n:k1]
            mean_u = np.mean(np.abs(u_kN))
            mean_du = np.mean(np.abs(np.diff(u_kN, axis=0)) / dt)
            # Also compute sum over thrusters in case Table XIV summed across the 3 thrusters:
            sum_u = np.mean(np.sum(np.abs(u_kN), axis=1))
            sum_du = np.mean(np.sum(np.abs(np.diff(u_kN, axis=0)) / dt, axis=1))
            thr_info = (f"Mean |u|={mean_u:.2f} (Sum={sum_u:.1f}) kN | "
                        f"Mean |du/dt|={mean_du:.2f} (Sum={sum_du:.1f}) kN/s")
        else:
            thr_info = "Thruster array not found"

        print(f"{label:40s} "
              f"N {rms_sk[0]:.3f} m   E {rms_sk[1]:.3f} m   psi {np.rad2deg(rms_sk[2]):.2f} deg | "
              f"{thr_info}")