"""
Thrust Allocation template

Students should implement an algorithm that maps the desired body-frame
wrench to individual thruster commands. The simulator calls, once per step:

    allocator.allocate(t, dt, tau_d, u_now, alpha_now) -> (u_cmd, alpha_cmd)

Inputs (full actuator state — use what your algorithm needs):
    t         : current simulation time [s]
    dt        : time step [s]              (rate-aware/dynamic allocation)
    tau_d     : (6,) desired BODY wrench [Fx, Fy, Fz, Mx, My, Mz]
                (the 3-DOF wrench to allocate is tau_d[[0, 1, 5]]
                 = [Fx, Fy, Mz]; the other components are zero)
    u_now     : current actual thrusts [N]     (rate-aware allocation)
    alpha_now : current thruster angles [rad]  (minimize azimuth slewing)

Outputs:
    u_cmd     : signed thrust command for each thruster [N]
    alpha_cmd : thruster angle command for each thruster [rad]

Students may implement, for example:
    - pseudo-inverse allocation,
    - weighted least-squares allocation,
    - optimization-based allocation,
    - power-minimizing allocation.

Contract with the checks, exactly as the public checks test it: ``allocate()``
must never command more thrust than a thruster can deliver, so for an
infeasible demand (the check uses 400 kN of sway) every |u_cmd| <= u_max.
*How* you handle an infeasible demand beyond that limit is your design
decision, not a checked criterion — state your rule and justify it in the
report.  The check also calls ``allocate()`` repeatedly from the initial
actuator state, applies each command through IDEAL actuators and feeds the
resulting actuator state back as ``u_now``/``alpha_now``; after 30 s of such
calls the achieved wrench must equal the requested one.  A steady-state
allocator therefore passes on its first call; a rate-aware allocator
(optional extension, and optional in the project text too) passes once it has
converged.  In the simulation the transient (thrust ramps, 2 rpm azimuth
slews) is shaped by the actuator model in ``models/thruster_dynamics.py``
whatever you command, so rate limiting your own commands is optional, not
required.  ``u_now``/``alpha_now`` are passed in because a thruster produces
the same force as (u, alpha) or as (-u, alpha + pi): which of the two you
command, and on what basis, is part of your allocation design.
"""
from typing import List, Optional, Tuple
import numpy as np

from models.thruster_dynamics import ThrusterConfig


class ThrustAllocator:
    """Template for student thrust allocation."""

    def __init__(self, thrusters: List[ThrusterConfig]):
        self.thrusters = thrusters

    def allocate(
        self,
        t: float,
        dt: float,
        tau_d: np.ndarray,
        u_now: Optional[np.ndarray] = None,
        alpha_now: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = len(self.thrusters)

        # TODO: Replace this placeholder with your thrust allocation algorithm.
        # The placeholder commands zero thrust and alpha for all thrusters.
        u_cmd = np.zeros(n)
        alpha_cmd = np.zeros(n)
                
        if alpha_now is None: # alpha_now[i] in line 131-132 crashes if alpha_now is None, which it is by default
            alpha_now = np.array([th.alpha0 for th in self.thrusters])
                
                # Weighted least-squares method 
        B = np.zeros((3,n)) # Overview of each thrusters contribution to the body wrench
        tau_3d = tau_d[[0, 1, 5]]
        for i in range(n):
            th = self.thrusters[i]
            B[0, i] = np.cos(th.alpha0) # F_x contribution
            B[1, i] = np.sin(th.alpha0) # F_y contribution
            B[2, i] = th.x * np.sin(th.alpha0) - th.y * np.cos(th.alpha0) # M_z contribution
        # Bm = np.linalg.solve(B, tau_3d) # F_x, F_y, M_z = B * u_cmd
                
        # Implementing the B_e matrix
        B_e = np.zeros((3,5))
        tunnel = self.thrusters[0]
        azimuth1 = self.thrusters[1]
        azimuth2 = self.thrusters[2]
                
        B_e[:,0] = B[:,0] # B_T
        B_e[:,1] = [1,0,-azimuth1.y] # B_Fx_A1
        B_e[:,2] = [0, 1, azimuth1.x] # B_Fy_A1
        B_e[:,3] = [1, 0, -azimuth2.y] # B_Fx_A2
        B_e[:,4] = [0, 1, azimuth2.x] # B_Fy_A2
                
        # Implementing weight matrix and choosing weights
        W_T = 1/tunnel.u_max**2 # Weight for tunnel thruster
        W_Fx_A1 = 1/azimuth1.u_max**2 # Weight for azimuth thruster
        W_Fy_A1 = 1/azimuth1.u_max**2 # Weight for azimuth thruster
        W_Fx_A2 = 1/azimuth2.u_max**2 # Weight for azimuth thruster
        W_Fy_A2 = 1/azimuth2.u_max**2 # Weight for azimuth thruster
                
        W = np.diag([W_T, W_Fx_A1, W_Fy_A1, W_Fx_A2, W_Fy_A2]) # Weighting matrix
                
        def z(tau):
            W_inv = np.linalg.inv(W)
            lambda1 = np.linalg.solve(B_e @ W_inv @ B_e.T, tau)
            return W_inv @ B_e.T @ lambda1
                
        Z = z(tau_3d)
                
        # Saturation
        u_T = abs(Z[0])
        u_A1 = np.hypot(Z[1], Z[2])
        u_A2 = np.hypot(Z[3], Z[4])
                
        if u_T > tunnel.u_max or u_A1 > azimuth1.u_max or u_A2 > azimuth2.u_max:
            scale1 = np.abs(tunnel.u_max / max(u_T,1e-9))
            scale2 = np.abs(azimuth1.u_max / max(u_A1,1e-9))
            scale3 = np.abs(azimuth2.u_max / max(u_A2,1e-9))
            scale = min(scale1, scale2, scale3)
                        
            Z *= scale
                
                
        u_cmd[0] = Z[0]
        alpha_cmd[0] = tunnel.alpha0
                
        def wrap(a): # From section 3.3, enshures heading (-pi, pi]
            return np.arctan2(np.sin(a), np.cos(a))

        for i, (Fx, Fy) in [(1, (Z[1], Z[2])), (2, (Z[3], Z[4]))]:
            u = np.hypot(Fx, Fy)
                    
            # If theres no force, the angle stays the same
            if u < 1:
                u_cmd[i] = 0
                alpha_cmd[i] = alpha_now[i]
                continue
                    
                    
            # Angle ambiguity
            # To solve angle ambiguity, the closest angle to alpha_now is chosen
            # Positive thrust at alpha and negative thrust at alpha + pi produce the same force
            alpha1 = np.arctan2(Fy, Fx)
            alpha2 = wrap(alpha1 + np.pi)
                    
            a1 = abs(wrap(alpha1 - alpha_now[i]))
            a2 = abs(wrap(alpha2 - alpha_now[i]))
                    
                    
            if a1 <= a2:
                u_cmd[i] = u
                alpha_cmd[i] = alpha1
            else:
                u_cmd[i] = -u
                alpha_cmd[i] = alpha2
        
        # Thrust- and angle rate limits: max change per step is rate * dt (eq. (1) and (3))
        u_prev = u_now if u_now is not None else np.zeros(n)
        alpha_prev = alpha_now if alpha_now is not None else np.zeros(n)
        for i, th in enumerate(self.thrusters):

            if th.kind == "azimuth":
                angle_difference = wrap(alpha_cmd[i] - alpha_prev[i]) # If the differnce in commanded and actual angles is too large, the thrust output has to be reduced to avoid overshoot.
                #u_cmd[i] *= np.abs(np.cos(angle_difference))
                dalpha_max = th.rot_speed * dt
                dalpha = np.clip(angle_difference, -dalpha_max, dalpha_max)
                alpha_cmd[i] = wrap(alpha_prev[i] + dalpha)
                
                    
            du_max = th.u_rate * dt
            u_cmd[i] = u_prev[i] + np.clip(u_cmd[i] - u_prev[i], -du_max, du_max)
        
        return u_cmd, alpha_cmd
