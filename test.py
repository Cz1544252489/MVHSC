#this file aims to process the algorithm step by step

# import part


import os

os.environ["OMP_NUM_THREADS"] = "1"
import torch
from MVHSC_Aux import data_importation, initialization, clustering, evaluation, lower_level, upper_level, iteration, test_part
from torch.optim import Adam, SGD
import matplotlib.pyplot as plt

DI = data_importation(view2=0) # 载入数据
IN = initialization(DI)     # 初始化数据
CL = clustering()           # 用于聚类
EV = evaluation(DI, CL)     # 用于评价
#
learning_rate = 0.01
lambda_r = 1

data = DI.get_data()
Theta, F = IN.initial()
UL = upper_level(F["UL"])
ULOP = SGD(UL.parameters(), lr = learning_rate)
LL = lower_level(F["LL"])
LLOP = Adam(LL.parameters(), lr = learning_rate)

IT = iteration(UL, LL, EV, learning_rate, lambda_r)

Epochs = 20
outer_epochs = 20
inner_epochs = 50

for _ in range(Epochs):
    for _ in range(Epochs):
        F = IT.inner_loop(F, Theta)
        # 优化上层变量
    F = IT.outer_loop(F)

EV.use_result(IT.result, "dump")

print(f"best_ul_nmi:{IT.result["best_ul_nmi"]}, best_ll_nmi:{IT.result["best_ll_nmi"]}")

print("aa")