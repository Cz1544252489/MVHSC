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
Epochs = 400
optimizer = torch.optim.Adam([IT.LL.y], lr=0.0001)
for i in range(Epochs):
    LL_val = IT.LL(IT.x)
    match IT.settings["grad_method"]:
        case "auto":
            if i>0:
                IT.LL.y.grad.zero_()
            (LL_val).backward()
            grad_y_LL = IT.LL.y.grad.clone()
            # optimizer.step()
            IT.update_value(IT.y, grad_y_LL)
        case "man":
            Theta_LL = IT.Theta + IT.settings["lambda_r"] * IT.x @ IT.x.T
            if IT.settings["use_proj"]:
                Proj_LL = torch.eye(IT.y.shape[0]) - IT.y @ IT.y.T
                grad_y_LL = 2 * Proj_LL @ Theta_LL @ IT.y
            else:
                grad_y_LL = 2 * Theta_LL @ IT.y
            IT.update_value(IT.y, grad_y_LL)
#
    norm_grad_ll = torch.linalg.norm(grad_y_LL, ord=2)
    ll_acc, ll_nmi, ll_ari = EV.assess(IT.y)
    IT.EV.record(IT.result, "LL", val=LL_val.item(), nmi=ll_nmi, grad=norm_grad_ll.item(), acc=ll_acc, ari=ll_ari)
    # print(f"val={LL_val.item()}, nmi={ll_nmi}, grad={norm_grad_ll.item()}, acc={ll_acc}, ari={ll_ari}")

opt_ul = torch.optim.Adam([IT.UL.x], lr=0.001)

for i in range(Epochs):
    UL_val = IT.UL(IT.y)
    match IT.settings["grad_method"]:
        case "auto":
            if i>0:
                IT.UL.x.grad.zero_()
            UL_val.backward()
            grad_x_UL = IT.UL.x.grad.clone()
            # opt_ul.step()
            IT.update_value(IT.x, grad_x_UL)
        case "man":
            Theta_UL = IT.settings["lambda_r"] * IT.y @ IT.y.T
            Proj_UL = torch.eye(IT.y.shape[0]) - IT.x @ IT.x.T
            grad_x_UL = 2 * Proj_UL @ Theta_UL @ IT.x
            IT.update_value(IT.x, grad_x_UL)

    norm_grad_ul = torch.linalg.norm(grad_x_UL, ord=2)
    ul_acc, ul_nmi, ul_ari = EV.assess(IT.x)
    IT.EV.record(IT.result, "UL", val=UL_val.item(), nmi=ul_nmi, grad=norm_grad_ul.item(), acc=ul_acc, ari=ul_ari)
    # print(f"val={UL_val.item()}, nmi={ul_nmi}, grad={norm_grad_ul.item()}, acc={ul_acc}, ari={ul_ari}")

EV.use_result(IT.result, "dump")


data = EV.use_result({}, "load")
EV.plot_result(data, [], ["nmi","grad","val","acc","ari"])
# print(f"best_ul_nmi:{IT.result["best_ul_nmi"]}, best_ll_nmi:{IT.result["best_ll_nmi"]}")

print("aa")