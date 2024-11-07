# This file aims to solve the problem referred to as Multi-View Hypergraph Spectral Clustering(MVHSC)

import os

os.environ["OMP_NUM_THREADS"] = "1"
import torch
from MVHSC_Aux import create_instances
import matplotlib.pyplot as plt


settings = {}
DI, IN, CL, EV, IT = create_instances(settings,0)

Epochs = 200
IT.result["list"] = []
for i in range(Epochs):
    # 优化下层函数
    IT.inner_loop()
    # 优化上层变量
    IT.outer_loop()
    flag = IT.update_lambda_r()
    if flag:
        IT.result["list"].append(i)

EV.use_result(IT.result, "dump")

data = EV.use_result({}, "load")
EV.plot_result(data, IT.result["list"], ["nmi","grad","val"])
print(f"best_ul_nmi:{IT.result["best_ul_nmi"]}, best_ll_nmi:{IT.result["best_ll_nmi"]}")

print("aaa")


