# This file aims to solve the problem referred to as Multi-View Hypergraph Spectral Clustering(MVHSC)


import os

os.environ["OMP_NUM_THREADS"] = "1"
import torch
from MVHSC_Aux import create_instances
import matplotlib.pyplot as plt


settings = {"learning_rate": 0.01, "lambda_r": 1,"epsilon": 0.05,
            "max_ll_epochs": 30, "max_ul_epochs": 20, "orth1": True,
            "orth2": True, "update_lambda": False}
DI, IN, CL, EV, IT = create_instances(settings,0)

Epochs = 200

for _ in range(Epochs):
    # 优化下层函数
    IT.inner_loop()
    # 优化上层变量
    IT.outer_loop()
    IT.update_lambda()

EV.use_result(IT.result, "dump")

data = EV.use_result({}, "load")
EV.plot_result(data, ["nmi"])
print(f"best_ul_nmi:{IT.result["best_ul_nmi"]}, best_ll_nmi:{IT.result["best_ll_nmi"]}")

print("aaa")


