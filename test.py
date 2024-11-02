#this file aims to process the algorithm step by step

# import part


import os

os.environ["OMP_NUM_THREADS"] = "1"
import torch
from MVHSC_Aux import data_importation, initialization, clustering, evaluation, lower_level, upper_level, iteration, test_part
from torch.optim import Adam, SGD
import matplotlib.pyplot as plt




DI = data_importation(view2=4)
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

Epochs = 20
epochs = 40
result = {"ll_nmi":[], "norm_grad_ll": [], "ll_val":[],
          "ul_nmi": [], "norm_grad_ul": [], "ul_val":[],
          "best_ll_nmi":0, "best_ul_nmi": 0 }
grad_method = "man"
for _ in range(Epochs):

    for epoch in range(epochs):
        # 优化下层变量
        LL_val = LL(F["UL"],Theta["LL"], lambda_r)
        match grad_method:
            case "man":
                Theta_ = Theta["LL"] + lambda_r * F["UL"] @ F["UL"].T
                # Proj_ = torch.eye(F["LL"].shape[0]) - F["LL"] @ F["LL"].T
                grad_ll = 2 * Theta_ @ F["LL"]
            case "auto":
                (-LL_val).backward()
                grad_ll = LL.F_ll.grad.clone()

        F["LL"] = IT.update_value(F["LL"], grad_ll, learning_rate, True)
        ll_nmi, _ = EV.assess(F["LL"])
        if ll_nmi > result["best_ll_nmi"]:
            result["best_ll_nmi"] = ll_nmi
            result["best_F_ll"] = F["LL"].tolist()
        norm_grad_ll = torch.linalg.norm(grad_ll, ord =2).item()
        EV.record(epoch, result, LL_val.item(), ll_nmi, norm_grad_ll,"LL")

    for epoch in range(epochs):
        # 优化上层变量
        UL_val = UL(F["LL"], lambda_r)
        match grad_method:
            case "man":
                Theta_ = lambda_r * F["LL"] @ F["LL"].T
                # Proj_ = torch.eye(F["LL"].shape[0]) #  - F["UL"] @ F["UL"].T
                grad_ul = 2 * Theta_ @ F["UL"]
            case "auto":
                UL_val.backward()
                grad_ul = UL.F_ul.grad.clone()

        F["UL"] = IT.update_value(F["UL"], grad_ul, learning_rate, True)
        ul_nmi, _ = EV.assess(F["UL"])
        if ul_nmi > result["best_ul_nmi"]:
            result["best_ul_nmi"] = ul_nmi
            result["best_F_ul"] = F["UL"].tolist()
        norm_grad_ul = torch.linalg.norm(grad_ul, ord =2).item()
        EV.record(epoch, result, UL_val.item(), ul_nmi, norm_grad_ul, "UL")


EV.use_result(result, "dump")

print(f"best_ul_nmi:{result["best_ul_nmi"]}, best_ll_nmi:{result["best_ll_nmi"]}")

print("aa")