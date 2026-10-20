"""
Wind template (Project Part 2)

Part 2 extends the Part 1 wind model.  In addition to the mean and slowly
varying wind speed, the wind field must now contain

* a fluctuating **gust** component of the wind speed, generated from a wind
  spectrum of your choice (Harris, Davenport, NORSOK, ...), and
* a **slowly varying wind direction**, without a gust component, bounded to
  at most +/- 5 degrees around the mean direction.

You built such a wind generator in Assignment 2 -- reuse it.  The coefficient
lookup (relative wind -> loads) is unchanged from Part 1; only the ambient
wind field (speed and direction as functions of time) becomes richer.

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
             {"U": ambient speed (total), "beta_ned": direction (towards, rad),
              "alpha_body": relative wind angle in BODY (rad),
              "U_mean": mean + slowly varying speed, "U_gust": gust speed}
             The Part 2 checks read "U_mean", "U_gust" and "beta_ned" to
             verify your wind field (gust present, direction within limit),
             so fill them in.
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
from part_2.config import WindConfig

_WIND_COEFF_FILE = Path(__file__).resolve().parent.parent / "data" / "wind_coeff.csv"


def load_wind_coefficients() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the vessel wind coefficient table.

    Returns
    -------
    alpha_deg : (M,) ndarray
        Relative wind angle grid [deg], from 0 to 360.
    C6 : (M, 6) ndarray
        Coefficients [Cx, Cy, Cz, Cphi, Ctheta, Cpsi] at each angle.
    """
    table = np.loadtxt(_WIND_COEFF_FILE, delimiter=",", skiprows=1)
    return table[:, 0], table[:, 1:]

def wrap_pi(angle: float) -> float:
    """Angle [-pi, pi]."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi

class Wind:
    """Template for the Part 2 wind model.

    Constructor contract -- the automated checks (``python check.py``,
    ``pytest``) and the Part 2 runner construct your model with this
    signature, so keep it working.  It is a superset of the Part 1 contract:
    your Part 1 model can be pasted in and extended.

        Wind(mean_speed, beta, semantics=..., sigma_slow=..., tau_slow=...,
             gust=..., gust_params=..., sigma_dir=..., tau_dir=...,
             dir_limit=..., seed=...)

    Parameters
    ----------
    mean_speed : mean wind speed [m/s].
    beta : mean direction [rad] in NED (0 = North, pi/2 = East).
    semantics : ``"from"`` (default, the usual meteorological convention --
        "wind from south" blows northward) or ``"towards"``.
    sigma_slow : standard deviation of the slowly varying wind speed
        component [m/s] (Part 1; 0 disables it).
    tau_slow : time constant of the slow speed variation [s].
    gust : switch the fluctuating (gust) speed component on.  The gust must
        be zero-mean and follow a wind spectrum; the spectrum and its
        parameters are your choice (see Assignment 2).  ``False`` gives the
        Part 1 wind field, which the checks use to isolate the gust.
    gust_params : dict with the parameters of *your* gust model (spectrum
        type, elevation, turbulence intensity, number of harmonics, ...).
        ``None`` means your own defaults.
    sigma_dir : standard deviation of the slowly varying wind direction
        [rad] (0 disables it).  The direction has NO gust component.
    tau_dir : time constant of the slow direction variation [s].
    dir_limit : hard bound on the direction excursion |beta(t) - beta| [rad];
        the project requires at most 5 degrees.
    seed : random seed for all stochastic components, so runs are
        reproducible.
    """

    def __init__(self, mean_speed: float = 0.0, beta: float = 0.0, *,
                 semantics: str = "from", sigma_slow: float = 0.0,
                 tau_slow: float = 120.0, gust: bool = False,
                 gust_params: Dict | None = None, sigma_dir: float = 0.0,
                 tau_dir: float = 300.0, dir_limit: float = np.deg2rad(5.0),
                 seed: int | None = None):
        # TODO: Store and use the parameters above in step().
        self.mean_speed = float(mean_speed)
        self.beta = float(beta)
        self.semantics = semantics
        self.sigma_slow = float(sigma_slow)
        self.tau_slow = float(tau_slow)
        self.gust = bool(gust)
        self.gust_params = dict(gust_params or {})
        self.sigma_dir = float(sigma_dir)
        self.tau_dir = float(tau_dir)
        self.dir_limit = float(dir_limit)
        self.seed = seed

        self.alpha_deg_table, self.C6_table = load_wind_coefficients() # Load wind coefficient table from CSV file

        # Initialize random number generator for the slow wind component
        self.rng = np.random.default_rng(self.seed)
        self.V_slow = 0.0
        self.beta_slow = 0.0

        self.V_max = WindConfig().V_max  # Maximum wind speed [m/s] from configuration 

        if self.gust:
            z = self.gust_params.get('z', 3.0)
            U_10 = self.mean_speed if self.mean_speed > 0 else 1e-3
            n = 0.468
            
            w_min = self.gust_params.get('w_min', 0.01)
            w_max = self.gust_params.get('w_max', 3.0)
            
            self.N_gust = self.gust_params.get('N_gust', 50)
            
            # Inicializace polí před použitím
            self.gust_omegas = np.zeros(self.N_gust)
            self.gust_phases = np.zeros(self.N_gust)
            self.gust_amps = np.zeros(self.N_gust)
            
            dw = (w_max - w_min) / self.N_gust
            df = dw / (2.0 * np.pi)
            
            for i in range(self.N_gust):
                omega = w_min + i * dw
                self.gust_omegas[i] = omega
                self.gust_phases[i] = self.rng.uniform(0, 2.0 * np.pi)
                
                f = omega / (2.0 * np.pi)
                x = 172.0 * f * ((z / 10.0)**(2/3)) * ((U_10 / 10.0)**(-0.75))
                S_f = 320.0 * ((U_10 / 10.0)**2) * ((z / 10.0)**0.45) * ((1.0 + x**n)**(-5.0 / (3.0 * n)))
                
                self.gust_amps[i] = np.sqrt(2 * S_f * df)

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

            # Slowly varying wind direction (exact discrete-time Ornstein-Uhlenbeck)
            if self.sigma_dir > 0.0 and self.tau_dir > 0.0:
                a_dir = np.exp(-dt / self.tau_dir)
                z_dir = self.rng.normal(0.0, 1.0)
                self.beta_slow = a_dir * self.beta_slow + self.sigma_dir * np.sqrt(1.0 - a_dir**2) * z_dir
                # Hard bound on the direction excursion
                self.beta_slow = np.clip(self.beta_slow, -self.dir_limit, self.dir_limit)
            else:
                self.beta_slow = 0.0

            # Gust component
            U_gust = 0.0
            if self.gust:
                U_gust = np.sum(self.gust_amps * np.cos(self.gust_omegas * t + self.gust_phases))

            # Total ambient wind speed (bounded)
            V_w = np.clip(self.mean_speed + self.V_slow + U_gust, 0.0, self.V_max)

            # Ambient wind direction in NED (convert "from" -> "towards")
            if self.semantics == "from":
                beta_ned = wrap_pi(self.beta + self.beta_slow + np.pi)
            else:
                beta_ned = wrap_pi(self.beta + self.beta_slow)

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
            info = {"U": float(V_w), "beta_ned": float(beta_ned), "alpha_body": float(alpha_body_rad),
                    "U_mean": float(self.mean_speed + self.V_slow), "U_gust": float(U_gust)}
            return tau_w6, info
