# part_2/config.py
# -----------------------------------------------------------------------------
# TMR4240 Marine Control Systems I
# Project – Design of Dynamic Positioning System
#
# Copyright (C) 2026: NTNU, Trondheim
# License: GPL-3.0-or-later
# -----------------------------------------------------------------------------
"""
Project Part 2 configuration — all tunable parameters in one place.

Like ``part_1/config.py``, this file belongs to you: put your own
configuration dataclasses here (observer parameters, wave-model settings,
controller gains, ...) so that every Part 2 simulation can be reconfigured
from this file and ``run_case_part_2.py`` without touching the engine.
"""
from dataclasses import dataclass, field
from math import pi
from typing import Any, Dict, Optional
from models.thruster_dynamics import ThrusterConfig
from part_2.control_plant_model import ControlPlantModel
import numpy as np


@dataclass
class RefAxisConfig:
    """Reference-model configuration for one axis (see part_2/reference.py).

    Tune these per simulation and justify the values in the report.

    Note: the automated checks build the default ``ReferenceModel(dt)``, which
    uses the field defaults below — so keep your final tuned values as the
    defaults here (overriding them only in ``run_case_part_2.py`` will not
    reach the checks).
    """
    # TODO (students): none of the three values below is a tuned value — they
    # are placeholders. Choose the natural frequency, the damping ratio and
    # the rate limit yourself, and justify each of them in the report.
    wn: float = 0.05                    # natural frequency [rad/s] (placeholder)
    zeta: float = 1.0                   # damping ratio [-] (placeholder)
    rate_limit: Optional[float] = None  # max |x_dot| (m/s or rad/s); None = off


@dataclass
class Part2SimConfig:
    """Part 2-only clock and switches; it does not inherit Part 1 settings.

    ``use_sensor_noise`` switches measurement noise on for the optional
    sensor-noise part of the project.  The noise levels are fixed course
    parameters in ``models/sensors.py`` — the switch is the only sensor
    setting you select per simulation.  The mandatory simulations run with
    ideal measurements (``False``, the default).

    ``use_controller=False`` switches the DP system off: the controller is
    not called, the desired wrench is zero and the vessel drifts freely under
    the environmental loads (Simulation 1).  ``use_observer`` still works in
    that mode, so an observer can be watched on a drifting vessel.

    ``dt`` must be 0.1 or 0.01 s (the engine rejects other steps; the Part 1
    default of 0.05 s does not carry over).  ``observer_kwargs`` are passed to
    ``select_observer(observer_type, **observer_kwargs)`` when the engine
    builds the observer, so a differently tuned observer can be selected from
    the configuration (e.g. ``{"cfg": MyObserverConfig(...)}``).
    """
    dt: float = 0.01
    T: float = 1000.0
    method: str = "Euler"
    use_controller: bool = True
    use_reference: bool = True
    use_observer: bool = False
    # Which observer the engine builds when none is injected.  This is only a
    # plumbing default so the field is never empty, NOT a recommendation:
    # every preset that needs an observer names the type explicitly, and which
    # one you adopt is your Part 2, Simulation 4 decision.
    observer_type: str = "nonlinear_passive"
    use_sensor_noise: bool = False
    thruster_dynamics: bool = True
    bypass_actuators: bool = False
    observer_kwargs: Dict[str, Any] = field(default_factory=dict)


def default_thrusters_part_2() -> list[ThrusterConfig]:
    """Same Gunnerus thruster fit as Project Part 1.

    Part 2 extends Part 1: the vessel and its three thrusters are unchanged.
    What is new in Part 2 is that the thruster *constraints* (max thrust,
    ramp rate ``u_rate`` and azimuth rotation speed) are enforced by the
    thruster dynamics and must be respected by your thrust allocation.
    The azimuth rotation limit is 2 rpm = 0.20944 rad/s.  Angles are in the
    BODY frame, where the fixed tunnel thruster points at ``+pi/2``.
    """
    rotation_speed = 2.0 * 2.0 * pi / 60.0
    return [
        ThrusterConfig("Tunnel_Bow", "tunnel", x=12.0, y=0.0,
                       u_max=32_000.0, u_rate=4_000.0,
                       rot_speed=0.0, alpha0=pi / 2.0),
        ThrusterConfig("Azimuth_1", "azimuth", x=-13.0, y=3.0,
                       u_max=80_000.0, u_rate=10_000.0,
                       rot_speed=rotation_speed, alpha0=0.0),
        ThrusterConfig("Azimuth_2", "azimuth", x=-13.0, y=-3.0,
                       u_max=80_000.0, u_rate=10_000.0,
                       rot_speed=rotation_speed, alpha0=0.0),
    ]



@dataclass
class ControllerConfig:
    """Controller selection and feed-forward options."""

    use_lqr: bool = True           # True -> LQR with integral effect, False -> PID
    use_feed_forward: bool = True  # True -> Using feed forward
    use_coriolis_ff: bool = True   # include C(nu_d) nu_d in the feed forward

@dataclass  
class PIDGains:
    """PID gains from pole placement, per DOF (surge, sway, yaw).

        K_p = omega_n^2 m
        K_d = 2 zeta omega_n m - d
        K_i = omega_n / 10 K_p
    """
    ref : RefAxisConfig = field(default_factory=RefAxisConfig)
    model : ControlPlantModel = field(default_factory=ControlPlantModel)

    zeta: float = 1.0              # closed-loop relative damping
    omega_ratio: float = 4.0       # omega_n = ratio * reference model frequency

    @property
    def m(self) -> np.ndarray:
        """(3,) mass/inertia per DOF, diagonal of M."""
        return np.diag(self.model.M)

    @property
    def d(self) -> np.ndarray:
        """(3,) linear damping per DOF, diagonal of D_l."""
        return np.diag(self.model.D_l)

    @property
    def omega_n(self) -> float:
        """Closed-loop natural frequency [rad/s]."""
        return self.omega_ratio * self.ref.wn

    @property
    def Kp(self) -> np.ndarray:
        """(3, 3) proportional gain."""
        return np.diag(self.omega_n**2 * self.m)

    @property
    def Kd(self) -> np.ndarray:
        """(3, 3) derivative gain."""
        return np.diag(2 * self.zeta * self.omega_n * self.m - self.d)

    @property
    def Ki(self) -> np.ndarray:
        """(3, 3) integral gain."""
        return self.omega_n / 10 * self.Kp

@dataclass
class LQRWeights:
    """LQR weights from Bryson's rule.

    Each state and input is weighted by the inverse square of its
    largest acceptable value:
        Q = diag(1 / x_max^2),   R = diag(1 / u_max^2)

    State vector [z, e, nu_err] (9,), where z is the integral of the
    position/heading error.
    """
    max_int: np.ndarray = field(default_factory=lambda: np.array([50.0, 50.0, np.deg2rad(100.0)]))  # integral states [m s, m s, rad s]
    max_err: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, np.deg2rad(2.0)]))      # position/heading error [m, m, rad]
    max_vel: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5, np.deg2rad(1.0)]))      # velocity error [m/s, m/s, rad/s]
    max_tau: np.ndarray = field(default_factory=lambda: np.array([50e3, 50e3, 500e3]))              # thrust [N, N, Nm]

    @property
    def Q(self) -> np.ndarray:
        """(9, 9) state weight, order [z, e, nu_err]."""
        return np.diag(np.r_[1 / self.max_int**2,
                             1 / self.max_err**2,
                             1 / self.max_vel**2])

    @property
    def R(self) -> np.ndarray:
        """(3, 3) input weight."""
        return np.diag(1 / self.max_tau**2)
@dataclass
class WaveConfig:
    """Linear second-order wave model per DOF (surge, sway, yaw).

    Model:
        xi_dot = A_w xi + K_w w_w
        eta_w  = C_w xi

    with xi = [xi_1, xi_2] (6,), where xi_2 is the wave-induced motion.
    
    """
    lam : np.ndarray = field(default_factory=lambda: np.full(3, 0.1))               # wave damping coeff, used in K_w
    omega_0 : np.ndarray = field(default_factory=lambda: np.full(3, 2*np.pi/ 8.0))  # wave peak frequency [rad/s]
    sigma : np.ndarray = field(default_factory=lambda: np.ones(3))                  # wave intensity
    zeta: np.ndarray = field(default_factory= lambda: np.full(3, 0.1))              # wave damping ratio, used in A_w

    @property
    def A_w(self) -> np.ndarray:
        """(6, 6) wave system matrix, [[0, I], [-Omega^2, -2 Lambda Omega]]."""
        Z, I = np.zeros((3,3)), np.eye(3)
        Omega = self.omega_0 * I  
        Lambda = self.zeta * I
        return np.block([[Z, I],[-Omega @ Omega, -2 * Lambda @ Omega]])
    
    @property
    def C_w(self) -> np.ndarray:
        """(3, 6) wave output matrix, [0, I]: picks out the wave-induced motion."""
        Z, I = np.zeros((3, 3)), np.eye(3)
        return np.block([Z, I])

    @property
    def K_w(self) -> np.ndarray:
        """(3, 3) wave noise gain, diag(2 lambda omega_0 sigma)."""
        return np.diag(2*self.lam*self.omega_0*self.sigma)

@dataclass
class BiasConfig:
    """First-order bias model: b_dot = -T_b^-1 b + w_b."""

    T_b : float = 1000

    @property
    def T_inv(self) -> np.ndarray:
        """(3, 3) inverse bias time constant matrix."""
        return np.eye(3) / self.T_b
  


@dataclass
class NPOConfig:
    """Models and gains for the nonlinear passive observer.

    K_1, K_2 are chosen from the wave model so the wave filter has
    notch damping zeta_n and cut-off frequency omega_c.
    K_3 (velocity) and K_4 (bias) are scaled from the vessel mass.

    """
    wave: WaveConfig = field(default_factory=WaveConfig)
    bias: BiasConfig = field(default_factory=BiasConfig)
    p2sim: Part2SimConfig = field(default_factory=Part2SimConfig)

    
    # Wave filter design
    cutoff_ratio : float = 1.25
    zeta_n: np.ndarray = field(default_factory= lambda: np.full(3, 1.0))

    #Velocity and bias gains
    nu_gain_scale: float = 0.1          # K_3 = scale * mass
    bias_ratio: float = 0.1             # K_4 = ratio * K_3

    @property
    def omega_c(self) -> np.ndarray:
        """(3,) filter cut-off frequencies [rad/s]."""
        return self.cutoff_ratio * self.wave.omega_0


    @property
    def K_1(self) -> np.ndarray:
        """(6, 3) wave state injection gain."""
        k1 = 2* self.omega_c/self.wave.omega_0 * (self.wave.zeta - self.zeta_n)
        k2 = 2*self.wave.omega_0 *(self.zeta_n - self.wave.zeta)
        return np.vstack([np.diag(k1), np.diag(k2)])

    @property
    def K_2(self) -> np.ndarray:
        """(3, 3) position injection gain, diag(omega_c)."""
        k3 = self.omega_c
        return np.diag(k3)

    def K_3(self, M: np.ndarray) -> np.ndarray :
        """(3, 3) velocity injection gain, scaled from the vessel mass.

        Args:
            M : (3, 3) mass matrix; surge/sway use the mean mass, yaw the inertia.
        """
        m = np.diag(M)
        m_xy = 0.5* (m[0]+m[1])
        return self.nu_gain_scale*np.diag([m_xy,m_xy,m[2]])

    def K_4(self,M: np.ndarray) -> np.ndarray:
        """(3, 3) bias injection gain, K_4 = bias_ratio * K_3."""

        return self.bias_ratio * self.K_3(M)
    

@dataclass
class EKFConfig:
    wave: WaveConfig = field(default_factory=WaveConfig)
    bias: BiasConfig = field(default_factory=BiasConfig)
    p2sim: Part2SimConfig = field(default_factory=Part2SimConfig)
    
    # Process noise, wave states
    q_w_N : float = 3
    q_w_E : float = 3
    q_w_psi : float = 4e-3

    # Process noise, bias states
    q_b_N : float = 1e4
    q_b_E : float = 1e4
    q_b_psi : float = 1e7
    
    # Measurement noise standard deviations [m, m, rad]
    sigma_N : float = 0.5
    sigma_E : float = 0.5
    sigma_psi : float = 8.7e-3 # ~0.5 deg

    # Initial covariance (diagonal)
    p0_xi: float = 1.0
    p0_eta: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 0.01]))
    p0_nu: np.ndarray = field(default_factory=lambda: np.array([0.1, 0.1, 0.01]))
    p0_b: np.ndarray = field(default_factory=lambda: np.array([1e8, 1e8, 1e10]))


    @property
    def Q_d(self) -> np.ndarray:
        """(6, 6) discrete process noise covariance, Q / dt."""
        Q = np.diag([self.q_w_N, self.q_w_E, self.q_w_psi, self.q_b_N, self.q_b_E, self.q_b_psi])
        return Q/self.p2sim.dt

    @property
    def R_d(self) -> np.ndarray:
        """(3, 3) measurement noise covariance, diag(sigma^2)."""
        return np.diag([self.sigma_N**2, self.sigma_E**2, self.sigma_psi**2])

    @property
    def P_0(self) -> np.ndarray:
        """(15, 15) initial error covariance for [xi, eta, nu, b]."""
        return np.diag(np.concatenate([np.full(6, self.p0_xi),
                                    self.p0_eta, self.p0_nu, self.p0_b]))
    

    
@dataclass
class WindConfig:
    V_max: float = 25.0        # maximum wind speed [m/s]
