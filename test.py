#this file aims to process the algorithm step by step

# import part


import os

os.environ["OMP_NUM_THREADS"] = "1"
import torch
from MVHSC_Aux import create_instances
import matplotlib.pyplot as plt


settings = {"learning_rate": 0.01, "lambda_r": 1,
            "max_ll_epochs": 30, "max_ul_epochs": 20, "orth": True}
DI, IN, CL, EV, IT = create_instances(settings,0)

Epochs = 10
epsilon = 0.05

for _ in range(Epochs):
    # 优化下层函数
    IT.inner_loop()
    # 优化上层变量
    IT.outer_loop()

    val = torch.trace(IT.F["UL"].T @ (torch.eye(IT.F["UL"].shape[0]) - IT.F["LL"] @ IT.F["LL"].T) @ IT.F["UL"])
    if val <= epsilon:
        IT.lambda_r = IT.lambda_r /2
    print(f"val:{val.item()}")

EV.use_result(IT.result, "dump")

data = EV.use_result({}, "load")
EV.plot_result(data, ["nmi"])
print(f"best_ul_nmi:{IT.result["best_ul_nmi"]}, best_ll_nmi:{IT.result["best_ll_nmi"]}")

print("aa")