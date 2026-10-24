import pickle
from importlib.resources import files
from IPython.display import Math, display


class ControlPlantModel:
    """3-DOF control plant model of R/V Gunnerus (surge, sway, yaw).

    Loads the vessel parameters from the MCSimPython Gunnerus data:
        M = M_RB + M_A   (3, 3) rigid-body + added mass
        D_l              (3, 3) linear damping
        D_u, D_v, D_r    nonlinear damping terms

    Low-frequency model used by the observers:
        M nu_dot + D_l nu = tau + R^T(psi) b
    """

    def __init__(self, *args, **kwargs):
        pkl = files("mcsimpy.vessel_data.gunnerus") / "parV_RVG3DOF.pkl"
        with open(str(pkl), "rb") as f:
            data = pickle.load(f)

        M_RB, M_A = data["Mrb"], data["Ma"]
        self.D_l, self.D_u, self.D_v, self.D_r = data["Dl"], data["Du"], data["Dv"], data["Dr"]
        self.M = M_RB + M_A

        