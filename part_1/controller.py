"""
Controller template

Students should implement a controller that maps the vessel state and the
full reference to a body-frame wrench. The simulator calls, once per step:

    controller.compute(t, dt, eta, nu, eta_ref, nu_ref, acc_ref) -> tau_d

All generalized vectors are 6-DOF, ordered [surge, sway, heave, roll, pitch,
yaw]. The 3-DOF model uses indices [0, 1, 5]; the remaining components are
zero on input and ignored on output.

Inputs (full loop state and full reference):
    t       : current simulation time [s]
    dt      : time step [s]
    eta     : (6,) vessel NED state [N, E, z, phi, theta, psi]
              (use N = eta[0], E = eta[1], psi = eta[5])
    nu      : (6,) vessel BODY velocities [u, v, w, p, q, r]
              (use u = nu[0], v = nu[1], r = nu[5])
    eta_ref : (6,) NED reference state
              (use N_d = eta_ref[0], E_d = eta_ref[1], psi_d = eta_ref[5])
    nu_ref  : (6,) NED-frame reference velocities
              (use Ndot_d = nu_ref[0], Edot_d = nu_ref[1], psidot_d = nu_ref[5])
    acc_ref : (6,) NED-frame reference accelerations, same layout as nu_ref
              (use for model-based / inertia feedforward)

Output:
    tau_d   : (6,) desired BODY wrench [Fx, Fy, Fz, Mx, My, Mz] (N, Nm)
              (fill in Fx = tau_d[0], Fy = tau_d[1], Mz = tau_d[5];
               leave the other components zero)

Optional hooks the simulator will use IF you define them (safe to omit):
    reset()                                  — called before each run
    apply_external_aw(tau_applied, psi, dt)  — anti-windup with the (6,)
                                               wrench actually applied after
                                               allocation and the actuator
                                               model (ideal in Part 1)
    last_pid_body  : {"P","I","D"} -> (6,) BODY components   (logged)
    int_ned (2,), int_psi (float)            — integrator states (logged)

Constructor contract — the automated checks (``python check.py``, ``pytest``,
``notebooks/part_1_demo.ipynb``) construct your controller as
``DPController()`` with NO arguments, so your final tuned gains must be the
constructor defaults. Tuning only inside ``run_case_part1.py`` will pass your
own runs but fail the checks.
"""
import numpy as np
from scipy.linalg import solve_continuous_are  #Ricatti solver
from part_1.config import PIDGains, LQRWeights, ControllerConfig

DOF = [0,1,5]

class DPController:
    """
    Template for student DP controller.

    Students may implement any type of controller (PID, LQR, backstepping,
    ...). Only compute() is required; everything else is optional.
    """

    def __init__(self, *args, **kwargs):
        self.Kp = np.asarray(PIDGains().Kp, dtype=float)
        self.Ki = np.asarray(PIDGains().Ki, dtype=float)
        self.Kd = np.asarray(PIDGains().Kd, dtype=float)
        self.K_aw = np.where(self.Kp > 0, 1.0/self.Kp, 0.0)

        M_RB = np.diag([5.741e5, 5.741e5, 4.124e7])
        M_A = np.array([[2.66e4, 0, 0],[0,1.326e5, -4.733e5], [0, -5.712e5, 1.332e7]])

        self.M = M_RB + M_A
        self.D = np.diag([2.235e4, 1.114e5, 9.748e6])

        self.K_lqr = self.design_lqr(LQRWeights())
        self.use_lqr = ControllerConfig.use_lqr
        self.use_coriolis_ff = ControllerConfig.use_coriolis_ff
        self.reset()

    def reset(self) -> None:
        """Optional: reset internal states (integrators, filters) before a run."""
        self.integral = np.zeros(3)               # NED integral [N, E, psi]
        self._u_cmd = np.zeros(3)
        self.sat_log = []
        self.last_pid_body = {k: np.zeros(6) for k in ("P", "I", "D")}
        

    @staticmethod

    def Rz(psi: float)-> float:
        """Rotation matrix about z : body frame -> NED frame"""
        c, s = np.cos(psi), np.sin(psi)

        return np.array([[c, -s, 0],
                         [s, c, 0],
                         [0, 0, 1]])
    
    def PID(self, error : np.array, error_dot : np.array, dt: float) -> np.array:

        # Proportional
        P = self.Kp* error

        #Integral 
        self.integral += error*dt
        I = self.Ki * self.integral

        #Derivative
        D = self.Kd * error_dot

        #adding anti-windup ? 
        
        return P, I, D

    @staticmethod 
    def Cor3(nu, M):
        u, v, r = nu
        m23 = 0.5 * (M[1,2] + M[2,1])
        return np.array([[0, 0, -M[1,1]*v -m23 *r],
                         [0, 0, M[0,0] *u],
                         [M[1,1] * v +m23*r, -M[0,0] *u, 0.0]])

    def FF(self, eta_d, eta_dot_d, eta_ddot_d):
        psi_d = eta_d[5]
        r_d = eta_dot_d[2]
        R_d = self.Rz(psi_d)

        S = np.array([[0, -r_d, 0],[r_d, 0, 0],[0,0,0]])
        nu_d = R_d.T @ eta_dot_d
        nu_dot_d = R_d.T @ eta_ddot_d - S @ nu_d
        tau_ff = self.M @ nu_dot_d + self.D @ nu_d
        tau_ff += self.Cor3(nu_d, self.M) @ nu_d
        return tau_ff
    
    def design_lqr(self, weights: LQRWeights) -> np.ndarray:
        Q, R = weights.Q , weights.R
        Z, I = np.zeros((3,3)), np.eye(3)  # Building block for matrices
        M_inv = np.linalg.inv(self.M)
        A_lin = np.block([[Z, I, Z],
                          [Z, Z, I],
                          [Z, Z, -M_inv@self.D]])
        B_lin = np.vstack([Z, Z, M_inv])
        P = solve_continuous_are(A_lin, B_lin, Q, R)   # solves A^T P + P A - P B R^-1 B^T P + Q = 0
        K = np.linalg.solve(R, B_lin.T @ P)         # K = R^-1 B^T P
        return K

    def LQR(self, eta, nu, eta_ref, eta_dot_d, dt):

        psi = eta[5]
        R = self.Rz(psi)

        e_ned = eta_ref[DOF]-eta[DOF]
        e_ned[2] = np.arctan2(np.sin(e_ned[2]), np.cos(e_ned[2]))

        self.integral += e_ned *dt

        z = R.T @ self.integral
        e = R.T @ e_ned
        nu_err = R.T @ eta_dot_d - nu[DOF]
        x = np.hstack([z, e, nu_err])

        tau = self.K_lqr @ x
        return tau

    def compute(
        self,
        t: float,
        dt: float,
        eta: np.ndarray,
        nu: np.ndarray,
        eta_ref: np.ndarray,
        nu_ref: np.ndarray | None = None,
        acc_ref: np.ndarray | None = None,
    ) -> np.ndarray:
        # TODO: Replace this placeholder with your DP controller.
        # Return the (6,) desired BODY wrench — fill in tau_d[0] = Fx,
        # tau_d[1] = Fy, tau_d[5] = Mz and leave the rest zero.
        psi = eta[5]
        r = nu[5]

        R = self.Rz(psi)
        error = eta_ref[DOF] - eta[DOF]
        error[2] = np.arctan2(np.sin(error[2]), np.cos(error[2]))

        eta_dot = R @ nu[DOF]
        eta_dot_d = np.zeros(3) if nu_ref is None else nu_ref[DOF]
        eta_ddot_d = np.zeros(3) if acc_ref is None else acc_ref[DOF]
        error_dot = eta_dot_d - eta_dot

        

        tau_d = np.zeros(6)

        if self.use_lqr:
            tau_d[DOF] = self.LQR(eta, nu, eta_ref, eta_dot_d, dt)
        else:
        #Rotate error to error_BF
            P, I, D = self.PID(error, error_dot, dt)
    
            P, I, D = R.T @ P, R.T @ I, R.T @ D 
            tau_d[DOF] = P + I + D
            for k, v in (("P", P), ("I", I), ("D", D)):
                self.last_pid_body[k] = np.zeros(6)
                self.last_pid_body[k][DOF] = v
            if self.use_coriolis_ff:
                tau_d[DOF] += self.FF(eta_ref, eta_dot_d, eta_ddot_d)  #Feed-Forward


        self._u_cmd = tau_d[DOF].copy()
        
        return tau_d

    def apply_external_aw(self, tau_applied, psi, dt):
        """Anti Windup with backcalculations : Integral -= K_aw(u-u_a) dt, with (u-u_a) rotated to NED"""
        u_a = tau_applied[DOF]
        du_body = self._u_cmd - u_a
        self.sat_log.append(bool(np.any(np.abs(du_body) > 1e-6 * (np.abs(self._u_cmd) + 1.0))))
        du_ned = self.Rz(psi) @ du_body
        self.integral -= self.K_aw * du_ned * dt
     