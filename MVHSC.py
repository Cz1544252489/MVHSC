from MVHSC_Aux import create_instances


settings = {"learning_rate": 0.01, "lambda_r": 1, "epsilon": 0.05, "update_learning_rate": True,
             "max_ll_epochs": 300, "max_ul_epochs": 300,
             "update_lambda_r": False, "use_proj": True,
             "plot_vline": True, "grad_method":"auto"}


DI, IN, CL, EV, IT = create_instances(settings,view2=4, seed_num= 42)

for _ in range(10):
    IT.inner_loop()
    IT.outer_loop()

file_name = "result4.json"
EV.use_result(IT.result,'dump', file_name)

data = EV.use_result({}, "load", file_name)
EV.plot_result(data, [], ["grad","val","nmi","acc","ari"])


print("aa")