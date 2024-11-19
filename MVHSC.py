from MVHSC_Aux import create_instances
import torch


settings = {"learning_rate": 0.01, "lambda_r": 1, "epsilon": 0.05, "update_learning_rate": True,
             "max_ll_epochs": 100, "max_ul_epochs": 1,
             "update_lambda_r": False, "use_proj": True,
             "plot_vline": True, "grad_method":"auto",
            "s_u": 1, "s_l": 1, "mu": 0.5,
            "theta":0.1
            }


DI, IN, CL, EV, IT = create_instances(settings,view2=0, seed_num= 42)


for _ in range(100):
    IT.inner_loop()
    IT.outer_loop()


file_name = "result.json"
EV.use_result(IT.result,'dump', file_name)

data = EV.use_result({}, "load", file_name)
EV.plot_result(data, [], ["grad","val","acc"],"show", picname=f"Mu{settings["mu"]}.png")


print("aa")