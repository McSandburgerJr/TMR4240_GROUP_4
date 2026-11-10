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
from scipy.optimize import minimize
import quadprog

from models.thruster_dynamics import ThrusterConfig


class ThrustAllocator:
    """Template for student thrust allocation."""

    def __init__(self, thrusters: List[ThrusterConfig]):
        self.thrusters = thrusters
        self.rate_aware = True 
        # To save last commanded thrust and angles
        self.u_cmd_prev = np.zeros(len(thrusters))
        self.prev_sign = np.zeros(len(thrusters))
        #Saturation methods, if false, QP method is used
        self.part1_saturation = False
        self.part2_QP_NonRotAzimuth = False
        self.part2_QP = True

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
        
        # Previous thrust and angle values, used for rate limiting
        u_prev = u_now if u_now is not None else np.zeros(n)
        alpha_prev = alpha_now if alpha_now is not None else np.zeros(n)
        margin = 1 # To avoid thrusters changing direction of thrust and/or angle at small time steps
        
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
        W_T = 1/max(tunnel.u_max**2, 1e-9) # Weight for tunnel thruster
        W_Fx_A1 = 1/max(azimuth1.u_max**2, 1e-9) # Weight for azimuth thruster
        W_Fy_A1 = 1/max(azimuth1.u_max**2, 1e-9) # Weight for azimuth thruster
        W_Fx_A2 = 1/max(azimuth2.u_max**2, 1e-9) # Weight for azimuth thruster
        W_Fy_A2 = 1/max(azimuth2.u_max**2, 1e-9) # Weight for azimuth thruster
                
        W = np.diag([W_T, W_Fx_A1, W_Fy_A1, W_Fx_A2, W_Fy_A2]) # Weighting matrix
                
        def z(tau):
            W_inv = np.linalg.inv(W)
            lambda1 = np.linalg.solve(B_e @ W_inv @ B_e.T, tau)
            return W_inv @ B_e.T @ lambda1
        
        def wrap(a): # From section 3.3, enshures heading (-pi, pi]
            return np.arctan2(np.sin(a), np.cos(a))
        
        # Part 1 Satuation method:
        if self.part1_saturation:
            Z = z(tau_3d)
            u_T = abs(Z[0])
            u_A1 = np.hypot(Z[1], Z[2])
            u_A2 = np.hypot(Z[3], Z[4])
                
            if u_T > tunnel.u_max or u_A1 > azimuth1.u_max or u_A2 > azimuth2.u_max:
                scale1 = np.abs(tunnel.u_max / max(u_T,1e-9))
                scale2 = np.abs(azimuth1.u_max / max(u_A1,1e-9))
                scale3 = np.abs(azimuth2.u_max / max(u_A2,1e-9))
                scale = min(scale1, scale2, scale3)
                        
                Z *= scale
        # Part 2 QP method with iterative solutions using Linear Programming
        if self.part2_QP_NonRotAzimuth: 
            tau_kN = tau_3d / 1000 # Convert to kN
            W_kN = W * 1e6 # Convert to kN, W depends on u_max^2
            # Q-matrix (Q >> W > 0)
            Q = np.diag([1, 1, 0.1])*100 # Weighting matrix for the QP problem
            
            # J
            def J(z):
                f, s = z[:5], z[5:]
                return f @ W_kN @ f + s @ Q @ s #  Eq. 12.255, p. 406 Fossen 2011
            
            # B_e f = tau_3d + s
            eq = {"type": "eq", "fun": lambda z: B_e @ z[:5] - tau_kN - z[5:]} # Constraint for Eq. 12.255, p. 406 Fossen 2011
            # B_e f - tau_kN - s = 0 
            
            
            # Iterative solutions using Linear Programming
            # These are the saturtaion constraints for the azimuth thrusters
            # u_max_A1^2 - F_x_A1^2 - F_y_A1^2 >= 0
            # u_max_A2^2 - F_x_A2^2 - F_y_A2^2 >= 0
            z_a = {"type": "ineq", "fun": lambda x: np.array([(azimuth1.u_max/1000)**2 - x[1]**2 - x[2]**2,   (azimuth2.u_max/1000)**2 - x[3]**2 - x[4]**2])}
            
            # minimize:
            # Takes the cost function, J (that should be minimized)
            # Initial guess (np.zeros(8))
            # Bounds for the tunnel, for x and y forces of both azimuth thrusters, and for slack variables (slack variables are unlimited)
            # Method: SLSQP(Sequential Least Squares Programming) solves the nonlinear optimization problem with constraints
            result = minimize(J, np.zeros(8), bounds = [(-tunnel.u_max/1000, tunnel.u_max/1000)] + [(-azimuth1.u_max/1000, azimuth1.u_max/1000)] * 4 + [(None, None)] * 3, constraints = [eq, z_a], method = "SLSQP")
            f_res = result.x[:5]
            
            Z = f_res * 1000 # Convert back to N
        
        # QP method with iterative solutions using Quadratic Programming
        def QP():
            tau_kN = tau_3d / 1000 # Convert to kN
            Z = z(tau_kN)
            u_max = np.array([tunnel.u_max, azimuth1.u_max, azimuth2.u_max]) / 1000 # Convert to kN
            u_rate = np.array([tunnel.u_rate, azimuth1.u_rate, azimuth2.u_rate]) / 1000 # Convert to kN/s
            
            
            a1 = alpha_now[1]
            a2 = alpha_now[2]
            # Define the configuration matrix for the QP problem which depends on angle
            B_e_QP = np.zeros((3,3))
            
            B_e_QP[:,0] = [0,1,tunnel.x] # B_T
            B_e_QP[:,1] = [np.cos(a1),np.sin(a1),azimuth1.x*np.sin(a1)-azimuth1.y*np.cos(a1)] # B_A1
            B_e_QP[:,2] = [np.cos(a2), np.sin(a2), azimuth2.x*np.sin(a2)-azimuth2.y*np.cos(a2)] # B_A2
            
            # Define the derivative requirement for the QP problem, constraint in eq. 12.263, p. 410 Fossen 2011
            u0 = u_prev / 1000 if u_prev is not None else np.zeros(5) # Convert to kN
            df = np.zeros((3,2))
            df[:,0] = u0[1] * np.array([-np.sin(a1), np.cos(a1), azimuth1.x*np.cos(a1)+azimuth1.y*np.sin(a1)]) # df/da for A1
            df[:,1] = u0[2] * np.array([-np.sin(a2), np.cos(a2), azimuth2.x*np.cos(a2)+azimuth2.y*np.sin(a2)]) # df/da for A2
            
            # Weight matrix for the QP problem
            W_kN = np.diag([1/max(tunnel.u_max**2, 1e-9), 1/max(azimuth1.u_max**2, 1e-9), 1/max(azimuth2.u_max**2, 1e-9)]) * 1e6 # Convert to kN
            
            # Q-matrix for the QP problem
            Q_QP = np.diag([1, 1, 0.01])*100 
            omega = np.diag([1, 1]) # Weighting factor for the angle change
            #----
            """
            # Cost function for the QP problem
            def J_QP(x):
                # x = [delta_u_T, delta_u_A1, delta_u_A2, delta_alpha_A1, delta_alpha_A2, s1, s2, s3]
                # x includes the change in thrust and angle for each thruster, as well as slack variables for the constraints
                u = u0 + x[:3]
                return u @ W_kN @ u + x[5:8] @ Q_QP @ x[5:8] + x[3:5] @ omega @ x[3:5] # Eq. 12.263, p. 410 Fossen 2011
            
            # Constraints in Eq. 12.263, p. 410 Fossen 2011 
            eq = {"type": "eq", "fun": lambda x: x[5:8] + B_e_QP @ x[0:3] + df @ x[3:5] - (tau_kN-B_e_QP @ u0)} 
            
            # Limits for the QP problem
            th_less0 = np.maximum(-u_max - u0, -u_rate*dt)
            th_greater0 = np.minimum(u_max - u0, u_rate*dt)
            alpha_rate_limits = np.array([azimuth1.rot_speed, azimuth2.rot_speed])
            
            # Result
            z0 = np.zeros(8)
            result = minimize(J_QP, z0, bounds = [(th_less0[i], th_greater0[i]) for i in range(3)] + [(-alpha_rate_limits[k]*dt, alpha_rate_limits[k]*dt) for k in range(2)] + [(None, None)]*3, constraints = [eq], method = "SLSQP")
            Z = result.x 
            
            u_cmd = np.clip(u0 + Z[0:3], -u_max, u_max) * 1000 # Convert back to N, the clip ensures that the commanded thrust does not exceed the maximum thrust of the thrusters
            alpha_cmd = alpha_now.copy()
            alpha_cmd[1] = wrap(a1 + Z[3]) # Wrap ensures that the commanded angle is within the range (-pi, pi]
            alpha_cmd[2] = wrap(a2 + Z[4])
            """
            # Med quadprog:
            H = np.zeros((8, 8))
            H[0:3, 0:3] = 2*W_kN
            H[3:5, 3:5] = 2*omega
            H[5:8, 5:8] = 2*Q_QP
            q = np.concatenate([2*W_kN @ u0,np.zeros(2),np.zeros(3)])
            
            A_eq = np.hstack([B_e_QP, df, np.eye(3)])   # 3×8
            b_eq = tau_kN - B_e_QP @ u0                 # 3×1
            
            # Limits for the QP problem
            th_less0 = np.maximum(-u_max - u0, -u_rate*dt)
            th_greater0 = np.minimum(u_max - u0, u_rate*dt)
            alpha_rate_limits = np.array([azimuth1.rot_speed, azimuth2.rot_speed])
            lb = np.concatenate([th_less0, -alpha_rate_limits*dt])
            ub = np.concatenate([th_greater0, alpha_rate_limits*dt])
            n_b = len(lb) # Only the first 5 variables are bounded
            I_b = np.hstack([np.eye(n_b), np.zeros((n_b, 3))])
            C = np.vstack([A_eq, I_b, -I_b]).T
            b = np.concatenate([b_eq, lb, -ub])
            
            Z = quadprog.solve_qp(H, -q, C, b, meq=3)[0]
            
            u_cmd = np.clip(u0 + Z[0:3], -u_max, u_max) * 1000 # Convert back to N, the clip ensures that the commanded thrust does not exceed the maximum thrust of the thrusters
            alpha_cmd = alpha_now.copy()
            alpha_cmd[1] = wrap(a1 + Z[3]) # Wrap ensures that the commanded angle is within the range (-pi, pi]
            alpha_cmd[2] = wrap(a2 + Z[4])
            
            return u_cmd, alpha_cmd
        
        
        if self.part2_QP:
            return QP()
        
        # Rest of code is used when self.part2_QP is not used
        u_cmd[0] = Z[0]
        alpha_cmd[0] = tunnel.alpha0

        for i, (Fx, Fy) in [(1, (Z[1], Z[2])), (2, (Z[3], Z[4]))]:
            u = np.hypot(Fx, Fy)
            th = self.thrusters[i]
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
            
            if self.rate_aware:
                angle_difference1 = wrap(alpha1 - alpha_now[i])
                angle_difference2 = wrap(alpha2 - alpha_now[i])
                time_angle_pos = abs(angle_difference1) / th.rot_speed
                time_thrust_pos = abs(u - u_prev[i]) / th.u_rate
                max_time_pos = max(time_angle_pos, time_thrust_pos)
                time_angle_neg = abs(angle_difference2) / th.rot_speed
                time_thrust_neg = abs(-u - u_prev[i]) / th.u_rate
                max_time_neg = max(time_angle_neg, time_thrust_neg)
                
                if self.prev_sign[i] == 0:
                    if max_time_pos <= max_time_neg:
                        u_cmd[i] = u
                        alpha_cmd[i] = alpha1
                    else:
                        u_cmd[i] = -u
                        alpha_cmd[i] = alpha2
                elif self.prev_sign[i] < 0: # If the previous thrust was negative, the thruster should continue with negative thrust
                    if max_time_pos <= max_time_neg - margin:
                        u_cmd[i] = u
                        alpha_cmd[i] = alpha1
                    else:
                        u_cmd[i] = -u
                        alpha_cmd[i] = alpha2
                
                elif self.prev_sign[i] > 0: # If the previous thrust was positive, the thruster should continue with positive thrust
                    if max_time_neg <= max_time_pos - margin:
                        u_cmd[i] = -u
                        alpha_cmd[i] = alpha2
                    else:
                        u_cmd[i] = u
                        alpha_cmd[i] = alpha1        
                self.prev_sign[i] = np.sign(u_cmd[i])
                
                
            if self.rate_aware == False:
                if a1 <= a2:
                    u_cmd[i] = u
                    alpha_cmd[i] = alpha1
                else:
                    u_cmd[i] = -u
                    alpha_cmd[i] = alpha2
            
        
        # Thrust- and angle rate limits: max change per step is rate * dt (eq. (1) and (3))
        if self.rate_aware:
            
            for i, th in enumerate(self.thrusters):

                if th.kind == "azimuth":
                    angle_difference = wrap(alpha_cmd[i] - alpha_prev[i]) # If the differnce in commanded and actual angles is too large, the thrust output has to be reduced to avoid overshoot.
                    
                    dalpha_max = th.rot_speed * dt
                    dalpha = np.clip(angle_difference, -dalpha_max, dalpha_max)
                    alpha_cmd[i] = wrap(alpha_prev[i] + dalpha)
                    
                    
                du_max = th.u_rate * dt
                u_cmd[i] = u_prev[i] + np.clip(u_cmd[i] - u_prev[i], -du_max, du_max)
        self.u_cmd_prev = u_cmd
        
        return u_cmd, alpha_cmd
