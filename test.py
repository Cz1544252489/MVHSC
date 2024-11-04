#this file aims to process the algorithm step by step

# import part


import os

os.environ["OMP_NUM_THREADS"] = "1"
import torch
from MVHSC_Aux import data_importation, initialization, clustering, evaluation, iteration, test_part
import matplotlib.pyplot as plt

DI = data_importation(view2=0) # 载入数据
IN = initialization(DI)     # 初始化数据
CL = clustering()           # 用于聚类
EV = evaluation(DI, CL)     # 用于评价
#
learning_rate = 0.01
lambda_r = 1

data = DI.data
Theta, F = IN.initial()
settings = {"learning_rate": learning_rate, "lambda_r": lambda_r,
            "max_ll_epochs": 30, "max_ul_epochs": 20, "orth": True}
IT = iteration(EV, F, settings)

Epochs = 100
epsilon = 0.5

for _ in range(Epochs):
    F = IT.inner_loop(F, Theta)

    # 优化上层变量
    F = IT.outer_loop(F)

    val = torch.trace(F["UL"].T @ (torch.eye(F["UL"].shape[0]) - F["LL"] @ F["LL"].T) @ F["UL"])
    if val <= epsilon:
        IT.lambda_r = IT.lambda_r /2
    print(f"val:{val.item()}")

EV.use_result(IT.result, "dump")

print(f"best_ul_nmi:{IT.result["best_ul_nmi"]}, best_ll_nmi:{IT.result["best_ll_nmi"]}")

print("aa")