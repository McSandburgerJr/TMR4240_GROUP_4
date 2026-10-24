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
from part_2.config import NPOConfig, EKFConfig

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
        conf = conf or NPOConfig()
        model = model or ControlPlantModel()

        self.A_w, self.C_w = conf.wave.A_w, conf.wave.C_w
        self.M_inv = np.linalg.inv(model.M)
        self.D = model.D_l
        self.T_inv = conf.bias.T_inv
        self.K_1, self.K_2 = conf.K_1, conf.K_2
        self.K_3, self.K_4 = conf.K_3(model.M), conf.K_4(model.M)

        self.reset()


    def reset(self, eta0 : np.ndarray | None = None) -> None:
        """Resetting the observer states.

        Steps:
            1. Set wave states, velocities and bias to zero
            2. Set position estimates from eta0 if given, otherwise leave it 
               as None and let  step() take it from the first measurement.

        Args:
            eta0 : (6,) initial position/attitude, 3-DOF part used.
                   If None, initialized from the first measurement.
        """
        self.xi_hat = np.zeros(6)
        self.nu_hat = np.zeros(3)
        self.b_hat = np.zeros(3)
        if eta0 is None:
            self.eta_hat = None
        else:
            self.eta_hat = np.asarray(eta0, dtype=float).reshape(6)[DOF]
            self.eta_hat[2] = wrap_angle_pi(self.eta_hat[2])


    def estimate_derivatives(self, y_tilde : np.ndarray, tau : np.ndarray, psi_m : float) -> tuple[np.ndarray, ...]:
        """ Time derivatives of the state estimates for the nonlinear passive observer

        Args:
            y_tilde     : (3,) output estimation error, y - (eta_hat + C_w xi_hat)
            tau         : (3,) control forces/moment [X,Y,N] in BODY
            psi_m       : measured heading [rad]
        
        Returns:
            xi_hat_dot  : (6,) wave-frequency state derivatives
            eta_hat_dot : (3,) position and heading derivatives
            nu_hat_dot  : (3,) body-fixed acceleration
            b_hat_dot   : (3,) bias derivatives
        """          
        R = Rz(psi_m)

        xi_hat_dot = self.A_w @ self.xi_hat + self.K_1 @ y_tilde
        eta_hat_dot = R @ self.nu_hat + self.K_2 @ y_tilde
        nu_hat_dot = -self.M_inv @ self.D @ self.nu_hat + self.M_inv @ R.T @ self.b_hat + self.M_inv @ tau + self.M_inv @ R.T @ self.K_3 @ y_tilde
        b_hat_dot = -self.T_inv @ self.b_hat + self.K_4 @ y_tilde
        
        return xi_hat_dot, eta_hat_dot, nu_hat_dot, b_hat_dot

    def step(self, t : float, dt : float, eta_measured : np.ndarray, tau_est : np.ndarray) -> ObserverEstimate:
        """One iteration of the nonlinear passive observer

        Steps:
            1. Extract the 3-DOF measurement y and control input tau.
            2. Initialize : set eta_hat to the first estimate.
            3. Compute the output estimate error : y - (eta_hat + C_w xi_hat)
            4. Store the current estimates for output
            5. Compute the state derivatives and integrate with forward Euler
        
        Args:
            t               : current time [s] (unused)
            dt              : time step [s]
            eta_measured    : (6,) measured position/attitude, 3-DOF part used
            tau_est         : (6,) control forces/moments, 3-DOF part used
        
        Returns:
            ObserverEstimate with estimates eta, nu and bias at time t
        
        """
        # 1. measurement and input, 3-DOF (surge, sway and yaw)
        y = np.asarray(eta_measured, dtype=float).reshape(6)[DOF]
        tau = np.asarray(tau_est, dtype=float).reshape(6)[DOF]
        psi_m = y[2]

        # 2. Initialize position estimates with the first measurement
        if self.eta_hat is None:
            self.eta_hat = np.asarray(eta_measured, dtype=float).reshape(6)[DOF]
            self.eta_hat[2] = wrap_angle_pi(self.eta_hat[2])

        # 3. Output estimation error
        y_tilde = y - self.eta_hat - self.C_w @ self.xi_hat
        y_tilde[2] = wrap_angle_pi(y_tilde[2])

        # 4. Current estimates (time t), in 6-DOF format
        estimates = np.zeros((3, 6))
        estimates[:, DOF] = np.stack([self.eta_hat, self.nu_hat, self.b_hat])

        # 5. Observer dynamics, forward Euler
        xi_hat_dot, eta_hat_dot, nu_hat_dot, b_hat_dot = self.estimate_derivatives(y_tilde, tau, psi_m)

        self.xi_hat += xi_hat_dot * dt 
        self.eta_hat += eta_hat_dot * dt
        self.eta_hat[2] = wrap_angle_pi(self.eta_hat[2])
        self.nu_hat += nu_hat_dot * dt
        self.b_hat += b_hat_dot * dt

        
        return ObserverEstimate(eta = estimates[0], nu = estimates[1], bias = estimates[2])


class KalmanFilterObserver(Observer):
    """Template for a discrete KF/EKF observer.

    Students define the process/measurement models and tune Q, R and P0.
    The input ``tau_est`` is the desired controller wrench before thruster
    dynamics.
    """

    name = "kalman"

    def __init__(self, conf : "EKFConfig | None"= None, model: "ControlPlantModel | None" = None, **kwargs) -> None:
        """Extended Kalman filter for 3-DOF DP with wave filtering and bias estimation.

        State vector x = [xi, eta, nu, b] (15,), model:
            x_dot = f(x, tau) + E w
            y     = C x + epsilon     (y = eta + wave-frequency motion)

        Args:
            conf  : filter configuration (noise covariances, wave and bias
                    models, initial covariance). Defaults to EKFConfig().
            model : vessel model providing M and D. Defaults to ControlPlantModel().
        """
        conf = conf or EKFConfig()
        model = model or ControlPlantModel()

        # Vessel model
        self.M_inv = np.linalg.inv(model.M)
        self.D = model.D_l

        # Wave and bias model
        self.A_w = conf.wave.A_w
        self.T_inv = conf.bias.T_inv

        Z, I = np.zeros((3,3)), np.eye(3)    

        # measurement matrix (3x15): y = eta + wave-frequency position (xi[3:6])  
        C_w = conf.wave.C_w 
        self.C_d = np.hstack([C_w, I, Z, Z])

        # Discrete noise input matrix (15x6), E_d = dt * E
        self.E_d = conf.p2sim.dt *np.block([[Z,Z],
                                            [conf.wave.K_w, Z],
                                            [Z,Z],
                                            [Z,Z],
                                            [Z,I]])

        # Noise covariances
        self.R_d = conf.R_d
        self.Q_d = conf.Q_d

        # Initial covariance
        self.P_0 = conf.P_0

        self.reset()

    def reset(self, eta0 : np.ndarray | None = None) -> None:
        """Reset the filter to its initial state and covariance
        Steps:
            1. Set wave states, velocities and bias to zero.
            2. Set position states from eta0 if given; otherwise leave them
            at zero and let step() take them from the first measurement.
            3. Set the a priori state x_bar and covariance P_bar = P_0.

        Args:
            eta0 : (6,) initial position/attitude, 3-DOF part used.
                   If None, initialized from the first measurement.
        
        """
        xi_hat = np.zeros(6)
        nu_hat = np.zeros(3)
        b_hat = np.zeros(3)
        if eta0 is None:
            eta_hat = np.zeros(3)
            self.initialized = False      # eta taken from the first measurement in step()
        else:
            eta_hat = np.asarray(eta0, dtype=float).reshape(6)[DOF]
            eta_hat[2] = wrap_angle_pi(eta_hat[2])
            self.initialized = True

        self.x_bar = np.concatenate([xi_hat, eta_hat, nu_hat, b_hat])
        self.P_bar = self.P_0.copy()


    def calculate_A_d(self, x : np.ndarray, h : float) -> np.ndarray:
        """Discrete-time state transition matrix for the EKF covariance predictor.

        Linearizes f(x,tau) about x and discretizes with forward Euler:
            A   = df/dx evaluated at x
            A_d = I + h * A
        
        State vector x : [xi, eta, nu, b]:
            xi  : (6,) wave-frequency states
            eta : (3,) position and heading [N,E,psi] in NED
            nu  : (3,) body fixed velocities [u,v,r]
            b   : (3,) bias forces/moment in NED
        Args:
            x   : (15,)
            h   : time step [s]

        Returns:
            A_d : (15,15) discrete state transition matrix
        """
        psi = x[8]
        u,v = x[9:11]
        b_N, b_E = x[12:14]

        Z, I = np.zeros((3,3)), np.eye(3)
        Z_6_3 = np.zeros((6,3))
        Z_3_6 = np.zeros((3,6))
        
        s = np.sin(psi)
        c = np.cos(psi)
        R = Rz(psi)      
        J_eta = np.array([[0,0,-u*s - v*c],
                          [0,0, u*c - v*s],
                          [0,0,0]])
        J_b = np.array([[0,0, -s*b_N + c*b_E],
                        [0,0, -c*b_N -s*b_E],
                        [0,0,0]])  
        A_d = np.eye(15) + h * np.block([
                                    [self.A_w, Z_6_3, Z_6_3, Z_6_3],
                                    [Z_3_6, J_eta, R, Z],
                                    [Z_3_6, self.M_inv@J_b, -self.M_inv@self.D, self.M_inv @ R.T],
                                    [Z_3_6, Z, Z, -self.T_inv]
                                    ])
        return A_d

    def calculate_h(self, x : np.ndarray) -> np.ndarray:
        """Noise free measurement function h(x), from y = h(x) + epsilon

        Measured output is total position and heading:
            y = eta + C_w xi (low-frequency + wave-frequency motion)

        Args:
            x       : (15,) state vector [xi, eta, nu, b]
        
        Returns: 
            h(x)    : (3,) predicted measurement [N, E, psi]
        
        """
        return self.C_d @ x

    def calculate_f(self, x : np.ndarray, tau : np.ndarray) -> np.ndarray:
        """Calculating the noise free part of the system function, x_dot = f(x,tau) + Ew

        State vector x : [xi, eta, nu, b]:
            xi  : (6,) wave-frequency states
            eta : (3,) position and heading [N,E,psi] in NED
            nu  : (3,) body fixed velocities [u,v,r]
            b   : (3,) bias forces/moment in NED
    
        Args:
            x   : (15,) state vector
            tau : (3,)  control forces/moment [X,Y,N] in BODY 
        
        Returns : 
            f   : (15,) time derivative of x with w=0
        """
        xi, eta, nu, b = x[:6], x[6:9], x[9:12], x[12:15]
        R = Rz(eta[2])

        f_1 = self.A_w @ xi
        f_2 = R @ nu
        f_3 = self.M_inv @(-self.D @ nu + R.T @ b + tau)
        f_4 = -self.T_inv @ b 

        return np.concatenate([f_1,f_2,f_3,f_4])
       

    
    def step(self, t : float, dt : float, eta_measured : np.ndarray, tau_est : np.ndarray) -> ObserverEstimate:
        """One EKF iteration: correct with the new measurement, then predict.

        Model : 
            x_dot = f(x,tau) + Ew
            y     = h(x) + epsilon
        
        Steps:
            1. Extract the 3-DOF measurements y and control input tau
            2. Initialize : set the position states to the first measurement
            3. Corrector : compute the output estimation error y - h(x_bar), the Kalman gain K,
               and the a posteriori estimate x_hat and covariance P_hat.
            4. Predictor : propagate x_hat with forward Euler through f, and 
               P_hat through the linearized dynamics A_d, giving x_bar and P_bar
               for the next step.
            5. Return the a posteriori estimates of eta, nu and bias. 
        
        Args:
            t               : current time [s] (unused)
            dt              : time step [s]
            eta_measured    : (6,) measured position, 3-DOF part used 
            tau_est         : (6,) control forces/moments, 3-DOF part used
        
        Returns: 
            ObserverEstimate with posteriori estimates of eta, nu and bias
        """
        # 1. measurement and input, 3-DOF (surge, sway and yaw)
        y = np.asarray(eta_measured, dtype=float).reshape(6)[DOF]
        tau = np.asarray(tau_est, dtype=float).reshape(6)[DOF]

        # 2. Initalize position states with the first measurement
        if not self.initialized:
            self.x_bar[6:9] = y
            self.x_bar[8] = wrap_angle_pi(self.x_bar[8])
            self.initialized = True

        h = dt
        I = np.eye(15)

        # 3. Corrector
        y_bar = self.calculate_h(self.x_bar)                                                        # predicted measurement
        y_tilde =  y - y_bar                                                                        # output estimation error
        y_tilde[2] = wrap_angle_pi(y_tilde[2])

        S = self.C_d @ self.P_bar @ self.C_d.T + self.R_d                                           # output error covariance
        K = np.linalg.solve(S, self.C_d @ self.P_bar).T                                             # Kalman Gain

        x_hat = self.x_bar + K @ y_tilde                                                            # state corrector
        x_hat[8] = wrap_angle_pi(x_hat[8])

        P_hat = (I - K @ self.C_d)@ self.P_bar @ (I - K @ self.C_d).T + K @ self.R_d @ K.T          # covariance corrector


        # 4. Predictor
        A_d = self.calculate_A_d(x_hat, h)                                                          # linearized, discretized dynamics
        f = self.calculate_f(x_hat, tau)
        
        self.x_bar = x_hat + h * f                                                                  # state predictor (Forward Euler)
        self.x_bar[8] = wrap_angle_pi(self.x_bar[8])                                                

        self.P_bar = A_d @ P_hat @ A_d.T + self.E_d @ self.Q_d @ self.E_d.T                         # covariance predictor

        # 5. Return a posteriori estimates in 6-DOF format
        estimates = np.zeros((3,6))
        estimates[:,DOF] = np.stack([x_hat[6:9],x_hat[9:12],x_hat[12:15]])

        return ObserverEstimate(eta = estimates[0], nu = estimates[1], bias = estimates[2])


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

