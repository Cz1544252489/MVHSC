


import torch
from MVHSC_Aux import data_importation, initialization, evaluation, clustering, iteration

DI = data_importation()
IN = initialization(DI)
CL = clustering()
EV = evaluation(DI, CL)

settings = {"learning_rate": 0.01, "lambda_r": 1, "epsilon": 0.05, "update_learning_rate": True,
            "max_ll_epochs": 30, "max_ul_epochs": 20, "orth1": False,
            "orth2": True, "update_lambda_r": False, "use_proj": True,
            "plot_vline": True, "grad_method":"man"}

IT = iteration(EV, IN, settings)

data = EV.use_result({}, "load")
EV.plot_result(data, [], ["nmi","grad","val","acc","ari"])
