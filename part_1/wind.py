"""
Wind template

Students should compute generalized BODY-frame wind loads:
    tau_w6 = [Fx, Fy, Fz, Mx, My, Mz]

The simulator uses the 3-DOF subset [Fx, Fy, Mz] = tau_w6 indices [0, 1, 5]
and calls, once per step:

    wind.step(t, dt, eta, nu) -> (tau_w6, info)

Inputs (full 6-DOF state — use what your model needs):
    t    : current simulation time [s]        (gust spectra, time variation)
    dt   : time step [s]                      (slowly-varying components)
    eta  : (6,) vessel state [N, E, z, phi, theta, psi] in NED
           (heading is eta[5])
    nu   : (6,) vessel BODY velocities [u, v, w, p, q, r]
           (RELATIVE wind: compute the loads from V_rw = V_wind - V_vessel,
            using the horizontal components nu[0], nu[1])

Outputs:
    tau_w6 : (6,) BODY loads
    info   : optional dict for logging, e.g.
             {"U": ambient speed, "beta_ned": direction (towards, rad),
              "alpha_body": relative wind angle in BODY (rad)}
             Return {} (or None) if you do not need it.
             NOTE: "beta_ned" is always the direction the wind blows
             TOWARDS, even when the constructor semantics is "from" —
             convert before logging, do not log the raw constructor value.

Wind coefficient data
---------------------
The vessel wind coefficients C(alpha) = [Cx, Cy, Cz, Cphi, Ctheta, Cpsi] are
provided in `data/wind_coeff.csv` (repository root), tabulated against the relative
wind angle alpha in degrees (0..360). Load them with:

    alpha_deg, C6 = load_wind_coefficients()

The wind loads are then computed as F_wind = U_rw^2 * C(alpha_rw), where U_rw
and alpha_rw are the relative wind speed and angle in the BODY frame.
"""
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
from part_1.config import WindConfig

# Wind coefficient table file path
_WIND_COEFF_FILE = Path(__file__).resolve().parent.parent / "data" / "wind_coeff.csv"

# Load the wind coefficient table from the CSV file
def load_wind_coefficients() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the vessel wind coefficient table.u
    Returns
    -------
    alpha_deg : (M,) ndarray
        Relative wind angle grid [deg], from 0 to 360.
    C6 : (M, 6) ndarray
        Coefficients [Cx, Cy, Cz, Cphi, Ctheta, Cpsi] at each angle.
    """
    table = np.loadtxt(_WIND_COEFF_FILE, delimiter=",", skiprows=1)
    return table[:, 0], table[:, 1:]

# Wind model class
class Wind:
    """Template for student wind model.

    Constructor contract — the automated checks (``python check.py``,
    ``pytest``, ``notebooks/part_1_demo.ipynb``) construct your model with
    this signature, so keep it working:

        Wind(mean_speed, beta, semantics=..., sigma_slow=..., seed=...)

    Parameters
    ----------
    mean_speed : mean wind speed [m/s].
    beta : direction [rad] in NED (0 = North, pi/2 = East).
    semantics : ``"from"`` (default, the usual meteorological convention —
        "wind from south" blows northward) or ``"towards"``.
    sigma_slow : standard deviation of the slowly-varying wind speed
        component [m/s] (required in Part 1; 0 disables it).
    tau_slow : time constant of the slow variation [s].
    seed : random seed for the slow component, so runs are reproducible.
    """

    def __init__(self, mean_speed: float = 0.0, beta: float = 0.0, *,
                 semantics: str = "from", sigma_slow: float = 0.0,
                 tau_slow: float = 120.0, seed: int | None = None):
        # TODO: Store and use the parameters above in step().
        self.mean_speed = float(mean_speed) # Mean wind speed in m/s
        self.beta = float(beta) # Wind direction in NED frame (0 = North, pi/2 = East)
        self.semantics = semantics  # "from" or "towards" semantics for wind direction
        self.sigma_slow = float(sigma_slow) # Standard deviation of the slowly-varying wind speed component in m/s
        self.tau_slow = float(tau_slow) # Time constant of the slow variation in seconds
        self.seed = seed # Random seed for reproducibility of the slow component

        self.alpha_deg_table, self.C6_table = load_wind_coefficients() # Load wind coefficient table from CSV file

        # Initialize random number generator for the slow wind component
        self.rng = np.random.default_rng(self.seed)
        self.V_slow = 0.0
        self.V_max = WindConfig().V_max  # Maximum wind speed [m/s] from configuration 

    def step(
            self,
            t: float,
            dt: float,
            eta: np.ndarray,
            nu: np.ndarray,
        ) -> Tuple[np.ndarray, Dict[str, float]]:
            
            # Slowly varying wind speed (exact discrete-time Ornstein-Uhlenbeck / Gauss-Markov)
            if self.sigma_slow > 0.0 and self.tau_slow > 0.0:
                a = np.exp(-dt / self.tau_slow)
                z = self.rng.normal(0.0, 1.0)
                self.V_slow = a * self.V_slow + self.sigma_slow * np.sqrt(1.0 - a**2) * z
            else:
                self.V_slow = 0.0

            # Total ambient wind speed (bounded)
            V_w = np.clip(self.mean_speed + self.V_slow, 0.0, self.V_max)

            # Ambient wind direction in NED (convert "from" -> "towards")
            if self.semantics == "from":
                beta_ned = (self.beta + np.pi) % (2.0 * np.pi)
            else:
                beta_ned = self.beta % (2.0 * np.pi)

            # Ambient wind velocity components in BODY frame
            psi = eta[5]
            u_w = V_w * np.cos(beta_ned - psi)
            v_w = V_w * np.sin(beta_ned - psi)

            # Relative wind velocity in BODY frame (V_wind - V_vessel)
            u_rw = u_w - nu[0]
            v_rw = v_w - nu[1]
            
            U_rw = np.hypot(u_rw, v_rw)

            # Relative wind angle (Direction TOWARDS which relative wind blows in BODY frame)
            alpha_body_rad = np.arctan2(v_rw, u_rw)
            alpha_body_deg = np.degrees(alpha_body_rad) % 360.0

            # Interpolate wind coefficients for the current relative angle
            C_alpha = np.empty(6)
            for i in range(6):
                C_alpha[i] = np.interp(
                    alpha_body_deg, 
                    self.alpha_deg_table, 
                    self.C6_table[:, i], 
                    period=360.0
                )

            # Compute final wind loads in BODY frame
            tau_w6 = (U_rw**2) * C_alpha

            # Logging info
            info = {
                "U": float(V_w),
                "beta_ned": float(beta_ned),
                "alpha_body": float(alpha_body_rad)
            }
            
            return tau_w6, info