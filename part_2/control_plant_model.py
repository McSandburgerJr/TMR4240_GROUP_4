import pickle
from importlib.resources import files
from IPython.display import Math, display


class ControlPlantModel:

    def __init__(self, *args, **kwargs):
        pkl = files("mcsimpy.vessel_data.gunnerus") / "parV_RVG3DOF.pkl"
        with open(str(pkl), "rb") as f:
            data = pickle.load(f)

        M_RB, M_A = data["Mrb"], data["Ma"]
        self.D_l, self.D_u, self.D_v, self.D_r = data["Dl"], data["Du"], data["Dv"], data["Dr"]
        self.M = M_RB + M_A
