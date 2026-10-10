"""Student observer templates and their common simulator interface.

The simulator calls, once per step (pose measurement only, no velocity):

    observer.step(t, dt, eta_measured, tau_est) -> ObserverEstimate(eta, nu, bias)

    eta_measured : (6,) measured NED pose [N, E, z, phi, theta, psi]
    tau_est      : (6,) desired controller wrench (before thruster dynamics)
                   of the PREVIOUS step (zero at the first step)
    eta, nu      : (6,) low-frequency NED pose and BODY velocity estimates
    bias         : (6,) slowly varying bias estimate (NED force) — may be zeros

A plain ``(eta, nu, bias)`` tuple is accepted as well.  Before every run the
engine calls ``reset(eta0)`` with the TRUE initial pose, so an observer can
start from the vessel's actual position.  Select an implementation with
``Part2SimConfig(observer_type=...)`` (Simulations 4-7 use the
``SELECTED_OBSERVER`` constant of ``run_case_part_2.py``); extra constructor
arguments go through ``Part2SimConfig(observer_kwargs=...)``.  Both classes
intentionally start as pass-through placeholders; students must implement and
tune them, keeping the tuned parameters as constructor defaults because the
checks call ``select_observer(kind)`` without arguments.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from simulation.utils import Rz, wrap_angle_pi
from part_2.control_plant_model import ControlPlantModel 
from part_2.config import NPOConfig

DOF = [0, 1 , 5]
@dataclass
class ObserverEstimate:
    eta: np.ndarray
    nu: np.ndarray
    bias: np.ndarray


class Observer:
    """Interface shared by all Part 2 observers."""

    name = "observer"

    def reset(self, eta0: np.ndarray | None = None):
        pass

    def step(
        self, t: float, dt: float, eta_measured: np.ndarray, tau_est: np.ndarray
    ) -> ObserverEstimate:
        raise NotImplementedError

    @staticmethod
    def _placeholder(eta_measured: np.ndarray) -> ObserverEstimate:
        return ObserverEstimate(
            eta=np.asarray(eta_measured, dtype=float).reshape(6).copy(),
            nu=np.zeros(6),
            bias=np.zeros(6),
        )


class NonlinearPassiveObserver(Observer):
    """Template for a nonlinear passive wave-filtering observer.

    Suggested states are wave-frequency motion, low-frequency pose, BODY
    velocity, and slowly-varying bias.  Implement the correction and model
    propagation in :meth:`step`.
    """

    name = "nonlinear_passive"

    def __init__(self, conf=None, model=None, **kwargs):
        conf = conf or NPOConfig(**kwargs)
        model = model or ControlPlantModel()

        self.A_w, self.C_w = conf.A_w, conf.C_w
        self.M_inv = np.linalg.inv(model.M)
        self.D = model.D_l
        self.T_inv = conf.T_inv
        self.K_1, self.K_2 = conf.K_1, conf.K_2
        self.K_3, self.K_4 = conf.K_3(model.M), conf.K_4(model.M)

        self.reset()


    def reset(self, eta0=None):
        self.xi_hat = np.zeros(6)
        self.nu_hat = np.zeros(3)
        self.b_hat = np.zeros(3)
        if eta0 is None:
            self.eta_hat = None
        else:
            self.eta_hat = np.asarray(eta0, dtype=float).reshape(6)[DOF]
            self.eta_hat[2] = wrap_angle_pi(self.eta_hat[2])


    def estimate_derivatives(self, y_tilde, tau, psi_m):
        """
        Args:
        
        
        Returns:
        """          
        R = Rz(psi_m)

        xi_hat_dot = self.A_w @ self.xi_hat + self.K_1 @ y_tilde
        eta_hat_dot = R @ self.nu_hat + self.K_2 @ y_tilde
        nu_hat_dot = -self.M_inv @ self.D @ self.nu_hat + self.M_inv @ R.T @ self.b_hat + self.M_inv @ tau + self.M_inv @ R.T @ self.K_3 @ y_tilde
        b_hat_dot = -self.T_inv @ self.b_hat + self.K_4 @ y_tilde
        
        return xi_hat_dot, eta_hat_dot, nu_hat_dot, b_hat_dot

    def step(self, t, dt, eta_measured, tau_est):
        """
        Args:
        
        
        Returns:
        
        """
        # TODO: implement the nonlinear passive observer.
        if self.eta_hat is None:
            self.eta_hat = np.asarray(eta_measured, dtype=float).reshape(6)[DOF]
            self.eta_hat[2] = wrap_angle_pi(self.eta_hat[2])

        y = np.asarray(eta_measured, dtype=float).reshape(6)[DOF]
        tau = np.asarray(tau_est, dtype=float).reshape(6)[DOF]
        psi_m = y[2]
        y_tilde = y - self.eta_hat - self.C_w @ self.xi_hat
        y_tilde[2] = wrap_angle_pi(y_tilde[2])

        xi_hat_dot, eta_hat_dot, nu_hat_dot, b_hat_dot = self.estimate_derivatives(y_tilde, tau, psi_m)

        self.xi_hat += xi_hat_dot * dt 
        self.eta_hat += eta_hat_dot * dt
        self.eta_hat[2] = wrap_angle_pi(self.eta_hat[2])
        self.nu_hat += nu_hat_dot * dt
        self.b_hat += b_hat_dot * dt
        
        estimates = np.zeros((3, 6))
        estimates[:, DOF] = np.stack([self.eta_hat, self.nu_hat, self.b_hat])
        
        return ObserverEstimate(eta = estimates[0], nu = estimates[1], bias = estimates[2])


class KalmanFilterObserver(Observer):
    """Template for a discrete KF/EKF observer.

    Students define the process/measurement models and tune Q, R and P0.
    The input ``tau_est`` is the desired controller wrench before thruster
    dynamics.
    """

    name = "kalman"

    def __init__(self, *args, **kwargs):
        pass

    def step(self, t, dt, eta_measured, tau_est):
        # TODO: implement the Kalman filter or extended Kalman filter.
        return self._placeholder(eta_measured)


def select_observer(kind: str, **kwargs) -> Observer:
    """Construct one observer without changing simulator code."""
    choices = {
        "nonlinear_passive": NonlinearPassiveObserver,
        "kalman": KalmanFilterObserver,
    }
    try:
        return choices[kind](**kwargs)
    except KeyError as exc:
        raise ValueError(f"observer_type must be one of {tuple(choices)}") from exc

