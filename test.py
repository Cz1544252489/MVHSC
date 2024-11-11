import torch
from MVHSC_Aux import data_importation, initialization, evaluation, clustering, iteration

DI = data_importation()
IN = initialization(DI)
CL = clustering()
EV = evaluation(DI, CL)

settings = {"learning_rate": 0.01, "lambda_r": 1, "epsilon": 0.05, "update_learning_rate": True,
             "max_ll_epochs": 30, "max_ul_epochs": 20, "orth1": False,
             "orth2": True, "update_lambda_r": False, "use_proj": True,
             "plot_vline": True, "grad_method":"auto"}

IT = iteration(EV, IN, settings)

Epochs = 10
for _ in range(Epochs):
    ll_val = IT.LL()
    grad_ll_y = torch.autograd.grad(ll_val, IT.LL.y, retain_graph=True)[0].clone()

    y = IT.update_value(IT.LL.y, grad_ll_y)
    with torch.no_grad():
        IT.LL.y.copy_(y)

    norm_grad_ll = torch.linalg.norm(grad_ll_y, ord=2)
    EV.record(IT.result,"LL", val=ll_val, grad=norm_grad_ll)

EV.use_result(IT.result,'dump')

EV.use_result(IT.result, "load")
EV.output_type(IT.result,["val","grad"])
print("aa")