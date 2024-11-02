#this file aims to process the algorithm step by step

# import part


import os
from typing import Literal

os.environ["OMP_NUM_THREADS"] = "1"
import torch
from MVHSC_Aux import data_importation, initialization, clustering, evaluation, lower_level, upper_level, iteration, test_part
from torch.optim import Adam, SGD
import matplotlib.pyplot as plt




DI = data_importation(view2=0)
data = DI.get_data()

IN = initialization(DI)
Theta, F = IN.initial()
CL = clustering()
EV = evaluation(DI, CL)
IT = iteration()
#
learning_rate = 0.01
lambda_r = 1

# 建立下层模型
LL = lower_level(F["LL"])
LLOP = Adam(LL.parameters(), lr = learning_rate)

UL = upper_level(F["UL"])
ULOP = SGD(UL.parameters(), lr = learning_rate)




epochs = 300
result = {"ll_nmi":[], "norm_grad_ll": []}
grad_method = "man"
for _ in range(1):

    for epoch in range(epochs):
        # 优化下层变量
        LL_val = LL(F["UL"],Theta["LL"], lambda_r)
        match grad_method:
            case "man":
                grad_ll = 2* (Theta["LL"]@F["UL"] + lambda_r * F["UL"] @ F["UL"].T @ F["LL"])
            case "auto":
                (-LL_val).backward()
                grad_ll = LL.F_ll.grad.clone()

        F["LL"] = IT.update_value(F["LL"], grad_ll, learning_rate/(epoch+1), False)
        nmi, _ = EV.assess(F["LL"])
        norm_grad_ll = torch.linalg.norm(grad_ll, ord =2).item()
        EV.record(epoch, result, nmi, norm_grad_ll)



    # for epoch in range(epochs):
    #
        # 优化上层变量
        # UL_val = UL(F["LL"], lambda_r)
        # UL_val.backward()
        # grad_UL = UL.F_UL.grad.clone()
        # if epoch < epochs-1:
        #     F["UL"] = update_value(F["UL"], grad_UL, learning_rate, False)
        # else:
        #     F["UL"] = update_value(F["UL"], grad_UL, learning_rate)
        #
        # print(torch.linalg.norm(grad_UL).item())


EV.plot_result(result["ll_nmi"])
# EV.plot_result(result["norm_grad_ll"])

print("aa")