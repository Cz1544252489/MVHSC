import torch
from MVHSC_Aux import data_importation, initialization, evaluation, clustering, iteration

DI = data_importation(seed=42)
IN = initialization(DI)
CL = clustering()
EV = evaluation(DI, CL)

settings = {"learning_rate": 0.01, "lambda_r": 1, "epsilon": 0.05, "update_learning_rate": True,
             "max_ll_epochs": 90, "max_ul_epochs": 90,
             "update_lambda_r": False, "use_proj": True,
             "plot_vline": True, "grad_method":"auto"}

IT = iteration(EV, IN, settings)

IT.inner_loop()
IT.outer_loop()

EV.use_result(IT.result,'dump')

data = EV.use_result({}, "load")
EV.plot_result(data, [], ["grad","val","nmi","acc","ari"])


print("aa")