from MVHSC_Aux import create_instances
import torch


settings = {"learning_rate": 0.01, "lambda_r": 1, "epsilon": 0.05, "update_learning_rate": True,
             "max_ll_epochs": 1, "max_ul_epochs": 1,
             "update_lambda_r": False, "use_proj": True,
             "plot_vline": True, "grad_method":"auto",
            "s_u": 1, "s_l": 1, "mu": 0,
            "alpha": 1, "beta":1 ,"theta":0.1
            }


DI, IN, CL, EV, IT = create_instances(settings,view2=2, seed_num= 42)


for _ in range(1):
    IT.inner_loop()

    ul_val = IT.UL()
    grad_ul_x = torch.autograd.grad(ul_val, IT.UL.x, retain_graph = True)[0]


file_name = "result.json"
EV.use_result(IT.result,'dump', file_name)

data = EV.use_result({}, "load", file_name)
EV.plot_result(data, [], ["grad","val","acc"])


print("aa")