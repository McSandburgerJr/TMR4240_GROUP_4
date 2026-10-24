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

# Per-axis tuning parameters live with the rest of the Part 2 configuration.
from part_2.config import RefAxisConfig
from simulation.utils import wrap_angle_pi


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


    def third_order_jerk(self, cfg: RefAxisConfig, x1: float, x2: float,
                         x3: float, r: float) -> float:
        """Desired jerk from the third-order reference model.

        Low-pass filter, per DOF:
            x1_ddd + (2 zeta + 1) wn x1_dd + (2 zeta + 1) wn^2 x1_d + wn^3 x1 = wn^3 r

        Args:
            cfg : RefAxisConfig with natural frequency wn [rad/s] and damping ratio zeta
            x1  : desired position [m] or heading [rad]
            x2  : desired velocity [m/s] or [rad/s]
            x3  : desired acceleration [m/s^2] or [rad/s^2]
            r   : commanded setpoint

        Returns:
            x3_dot : desired jerk [m/s^3] or [rad/s^3]
        """
        wn, zeta = cfg.wn, cfg.zeta
        return wn**3 * (r - x1) - (2*zeta + 1) * wn * x3 - (2*zeta + 1) * wn**2 * x2

    def step(self, t: float, dt: float,
             eta_cmd: np.ndarray) -> tuple[np.ndarray, ...]:
        """One step of the reference model for N, E and psi.

        Filters the commanded setpoint through the third-order reference
        model to give smooth desired position, velocity and acceleration.
        Integrated with forward Euler.

        Steps:
            1. N and E: filter each axis towards its setpoint (cfg_xy).
            2. psi: use the shortest angle to the setpoint, so the vessel
               turns the short way, then filter with cfg_psi and wrap.

        Args:
            t       : current time [s] (unused)
            dt      : time step [s]
            eta_cmd : (6,) commanded pose [N, E, z, phi, theta, psi] in NED

        Returns:
            eta_ref : (6,) desired pose in NED
            nu_ref  : (6,) desired velocity, time derivative of eta_ref (NED)
            acc_ref : (6,) desired acceleration (NED)
        """
        N_cmd, E_cmd, psi_cmd = eta_cmd[0], eta_cmd[1], eta_cmd[5]

        # 1. N and E
        for idx, r in ((0, N_cmd), (1, E_cmd)):
            x1, x2, x3 = self.eta_ref[idx], self.nu_ref[idx], self.acc_ref[idx]
            x3_dot = self.third_order_jerk(self.cfg_xy, x1, x2, x3, r)
            self.eta_ref[idx] = x1 + dt * x2
            self.nu_ref[idx] = x2 + dt * x3
            self.acc_ref[idx] = x3 + dt * x3_dot

        # 2. Yaw: setpoint as current reference + shortest angle error,
        #    so the filter never sees a 2*pi jump
        x1, x2, x3 = self.eta_ref[5], self.nu_ref[5], self.acc_ref[5]
        r = x1 + wrap_angle_pi(psi_cmd - x1)
        x3_dot = self.third_order_jerk(self.cfg_psi, x1, x2, x3, r)
        self.eta_ref[5] = wrap_angle_pi(x1 + dt * x2)
        self.nu_ref[5] = x2 + dt * x3
        self.acc_ref[5] = x3 + dt * x3_dot

        return self.eta_ref.copy(), self.nu_ref.copy(), self.acc_ref.copy()