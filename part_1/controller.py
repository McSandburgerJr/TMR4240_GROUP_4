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
from part_1.config import PIDGains

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
    
    @staticmethod 

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
        R = self.Rz(psi)

        error = eta_ref[DOF] - eta[DOF]
        error[2] = np.arctan2(np.sin(error[2]), np.cos(error[2]))

        eta_dot = R @ nu[DOF]
        eta_dot_ref = np.zeros(3) if nu_ref is None else nu_ref[DOF]
        error_dot = eta_dot_ref - eta_dot

        #Rotate error to error_BF
        P, I, D = self.PID(self, error, error_dot, dt)

        P, I, D = R.T @ P, R.T @ I, R.T @ D 
        tau_d = np.zeros(6)
        tau_d[DOF] = P + I + D
        self._u_cmd = tau_d[DOF].copy()


        for k, v in (("P", P), ("I", I), ("D", D)):
            self.last_pid_body[k] = np.zeros(6)
            self.last_pid_body[k][DOF] = v

        return tau_d

    def apply_external_aw(self, tau_applied, psi, dt):
        """Anti Windup with backcalculations : Integral -= K_aw(u-u_a) dt, with (u-u_a) rotated to NED"""
        u_a = tau_applied[DOF]
        du_body = self._u_cmd - u_a
        self.sat_log.append(bool(np.any(np.abs(du_body) > 1e-6 * (np.abs(self._u_cmd) + 1.0))))
        du_ned = self.Rz(psi) @ du_body
        self.integral -= self.K_aw * du_ned * dt
     