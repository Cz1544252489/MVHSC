import os

os.environ["OMP_NUM_THREADS"] = "1"

from MVHSC_Aux import create_instances


settings = {"learning_rate": 0.01, "lambda_r": 1,
            "max_ll_epochs": 30, "max_ul_epochs": 20, "orth1": False,
            "orth2": True}
DI, IN, CL, EV, IT = create_instances(settings,0)

data = EV.use_result({}, "load")
EV.plot_result(data, ["nmi","grad","val"])


print("aa")