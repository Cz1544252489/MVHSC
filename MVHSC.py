import torch
from MVHSC_Aux import data_importation, initialization, evaluation, clustering, iteration

DI = data_importation(seed=729)
IN = initialization(DI)
CL = clustering()
EV = evaluation(DI, CL)

settings = {"learning_rate": 0.01, "lambda_r": 1, "epsilon": 0.05, "update_learning_rate": True,
             "max_ll_epochs": 30, "max_ul_epochs": 20, "orth1": False,
             "orth2": True, "update_lambda_r": False, "use_proj": True,
             "plot_vline": True, "grad_method":"auto"}

IT = iteration(EV, IN, settings)

# UL = IT.upper_level(IT.x, IT.y, IT.settings["lambda_r"])
norm1 = torch.linalg.norm(torch.ones((250,6), dtype=torch.float64), ord=2)
IT.x = torch.ones((250,6), dtype=torch.float64)/norm1
IT.y = torch.ones((250,6), dtype=torch.float64)/norm1

LL = IT.lower_level(IT.x, IT.y, IT.settings["lambda_r"], IT.Theta)

Epochs = 20
for i in range(Epochs):
    ll_val = LL()

    grad_ll_y_auto1 = torch.autograd.grad(ll_val, LL.y, retain_graph=True)[0]
    Proj_LL = torch.eye(LL.y.shape[0]) - LL.y @ LL.y.T
    grad_ll_y_auto = Proj_LL @ grad_ll_y_auto1
    # IT.syn()
    grad_ll_y_man = IT.get_grad_y_ll_man(LL.x, LL.y)
    match IT.settings["grad_method"]:
        case "auto":
            grad_ll_y = grad_ll_y_auto
        case "man":
            grad_ll_y = grad_ll_y_man

    y = IT.update_value(LL.y, grad_ll_y)
    with torch.no_grad():
        LL.y.copy_(y)

    norm_grad_ll = torch.linalg.norm(grad_ll_y, ord=2)
    EV.record(IT.result,"LL", val=ll_val.item(), grad=norm_grad_ll.item())

EV.use_result(IT.result,'dump')

data = EV.use_result({}, "load")
EV.plot_result(data, [], ["grad","val"])

# 遍历 z 的计算图
node = ll_val.grad_fn
while node is not None:
    print(node)
    node = node.next_functions[0][0] if node.next_functions else None

print("aa")