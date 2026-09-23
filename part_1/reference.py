"""
Reference template

Students should filter or shape commanded setpoints before they are sent to
the controller. The simulator calls, once per step:

    ref.step(t, dt, eta_cmd) -> (eta_ref, nu_ref, acc_ref)

All generalized vectors are 6-DOF, ordered [surge, sway, heave, roll, pitch,
yaw]. The 3-DOF model uses indices [0, 1, 5]; leave the rest zero.

Inputs:
    t       : current simulation time [s]
    dt      : time step [s]
    eta_cmd : (6,) commanded setpoint
              (use N_cmd = eta_cmd[0], E_cmd = eta_cmd[1], psi_cmd = eta_cmd[5])

Outputs (all NED-frame, (6,) each):
    eta_ref : filtered reference
              (fill in N_ref = [0], E_ref = [1], psi_ref = [5])
    nu_ref  : reference velocities
              (fill in Ndot_ref = [0], Edot_ref = [1], psidot_ref = [5])
    acc_ref : reference accelerations
              (fill in Nddot_ref = [0], Eddot_ref = [1], psiddot_ref = [5])

The simulator forwards all three to the controller, so a smooth reference
model here directly enables velocity/acceleration feedforward there.
"""
from typing import Tuple
import numpy as np

# Per-axis tuning parameters live with the rest of the Part 1 configuration.
from part_1.config import RefAxisConfig


class ReferenceModel:
    """
    Template for student reference model.

    The default implementation is pass-through, so eta_ref = eta_cmd and the
    reference velocities/accelerations are zero.
    """

    def __init__(
        self,
        dt: float,
        cfg_xy: RefAxisConfig | None = None,
        cfg_psi: RefAxisConfig | None = None,
    ):
        self.dt = float(dt)
        self.cfg_xy = cfg_xy if cfg_xy is not None else RefAxisConfig()
        self.cfg_psi = cfg_psi if cfg_psi is not None else RefAxisConfig()
        self.eta_ref = np.zeros(6)
        self.nu_ref = np.zeros(6)
        self.acc_ref = np.zeros(6)

    def reset(self, eta0: np.ndarray) -> None:
        """Initialize the reference at the vessel's current (6,) state."""
        self.eta_ref = np.asarray(eta0, dtype=float).reshape(6).copy()
        self.nu_ref = np.zeros(6)
        self.acc_ref = np.zeros(6)

    @staticmethod

    def _wrap_to_pi(angle :float) -> float:                          
        """Wrapping the boundaries of psi from [0,2pi) to (-pi, pi]"""
        return (angle + np.pi) % (2*np.pi) - np.pi

    @staticmethod

    def _second_order_accerleration(cfg: RefAxisConfig, x1, x2, r): 
        """Solving the second-order low-pass filter for desired accelerations. x1 = eta_desired, x2 = eta_dot_desired, r = eta_cmd"""

        wn, zeta= cfg.wn, cfg.zeta

        return wn**2*(r-x1) - 2*zeta*wn*x2
    
    @staticmethod

    def third_order_jerk(cfg: RefAxisConfig, x1, x2, x3, r):
        wn, zeta = cfg.wn, cfg.zeta
        return (wn**3)*(r-x1) - (2*zeta+1)*wn*x3 - (2*zeta+1)*wn**2*x2

    def step(
        self, t: float, dt: float, eta_cmd: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # TODO: Replace this pass-through placeholder with your reference model.
        N_cmd, E_cmd, psi_cmd = eta_cmd[0], eta_cmd[1], eta_cmd[5]

        # N, E 
        for idx, r in ((0,N_cmd), (1,E_cmd)):
            x1, x2, x3 = self.eta_ref[idx], self.nu_ref[idx], self.acc_ref[idx]
            x3_dot = self.third_order_jerk(self.cfg_xy, x1, x2, x3, r)
            self.eta_ref[idx] = x1 + x2*dt
            self.nu_ref[idx] = x2 + x3*dt
            self.acc_ref[idx] = x3 + x3_dot*dt

        # psi - yaw
        psi_error = self._wrap_to_pi(psi_cmd - self.eta_ref[5])
        r = self.eta_ref[5] + psi_error
        x1, x2, x3 = self.eta_ref[5], self.nu_ref[5], self.acc_ref[5]
        x3_dot = self.third_order_jerk(self.cfg_psi, x1, x2, x3, r)
        self.eta_ref[5] = self._wrap_to_pi(x1 + x2*dt)
        self.nu_ref[5] = x2 + x3*dt
        self.acc_ref[5] = x3 + x3_dot*dt
    
        return self.eta_ref, self.nu_ref, self.acc_ref
