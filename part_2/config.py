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
    wn: float = 1.0                     # natural frequency [rad/s] (placeholder)
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
    dt: float = 0.1
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
class PIDGains:
    Kp: np.ndarray = field(default_factory=lambda: np.array([9.61e4, 1.13e5, 8.73e6]))
    Ki: np.ndarray = field(default_factory=lambda: np.array([3.84e3, 4.52e3, 3.49e5]))
    Kd: np.ndarray = field(default_factory=lambda: np.array([4.58e5, 4.54e5, 3.39e7]))

@dataclass
class LQRWeights: 
    #Using Bryons rule
    max_int: np.ndarray = field(default_factory=lambda: np.array([50.0, 50.0, np.deg2rad(100.0)]))  # integral states [m*s, m*s, rad*s]
    max_err: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, np.deg2rad(2.0)]))      # position/heading error [m, m, rad]
    max_vel: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5, np.deg2rad(1.0)]))      # velocity error [m/s, m/s, rad/s]
    max_tau: np.ndarray = field(default_factory=lambda: np.array([50e3, 50e3, 500e3]))              # thrust [N, N, Nm]
    
    @property
    def Q(self) -> np.ndarray:
        """9x9 state weight, order [z, e, nu_err]"""
        return np.diag(np.r_[1 / self.max_int**2,
                             1 / self.max_err**2,
                             1 / self.max_vel**2])

    @property
    def R(self) -> np.ndarray:
        """3x3 input weight"""
        return np.diag(1 / self.max_tau**2)

@dataclass
class ControllerConfig:
    use_lqr: bool = True           # True -> LQR with integral effect, False -> PID
    use_feed_forward: bool = True  # True -> Using feed forward
    use_coriolis_ff: bool = True   # include C(nu_d) nu_d in the feed forward


@dataclass
class NPOConfig:
    omega_0 : np.ndarray = field(default_factory=lambda: np.full(3, 2*np.pi/ 8.0))
    zeta: np.ndarray = field(default_factory= lambda: np.full(3, 0.1))
    
    # Wave filter design
    cutoff_ratio : float = 1.25
    zeta_n: np.ndarray = field(default_factory= lambda: np.full(3, 1.0))
    
    # Bias Model
    T_b : float = 1000

    #Velocity and bias gains
    nu_gain_scale: float = 0.1          # K_3 = scale * mass
    bias_ratio: float = 0.1             # K_4 = ratio * K_3

    @property
    def omega_c(self):
        return self.cutoff_ratio * self.omega_0

    @property
    def T_inv(self):
        return np.eye(3)/self.T_b
    
    @property
    def A_w(self):
        Z, I = np.zeros((3,3)), np.eye(3)
        Omega = self.omega_0 * I  
        Lambda = self.zeta * I
        return np.block([[Z, I],[-Omega @ Omega, -2 * Lambda @ Omega]])
    
    @property
    def C_w(self):
        Z, I = np.zeros((3, 3)), np.eye(3)
        return np.block([Z, I])

    @property
    def K_1(self):
        k1 = 2* self.omega_c/self.omega_0 * (self.zeta - self.zeta_n)
        k2 = 2*self.omega_0 *(self.zeta_n - self.zeta)
        return np.vstack([np.diag(k1), np.diag(k2)])

    @property
    def K_2(self):
        k3 = self.omega_c
        return np.diag(k3)

    def K_3(self, M):
        m = np.diag(M)
        m_xy = 0.5* (m[0]+m[1])
        return self.nu_gain_scale*np.diag([m_xy,m_xy,m[2]])

    def K_4(self,M):
        return self.bias_ratio * self.K_3(M)
    

