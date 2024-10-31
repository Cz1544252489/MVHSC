#this file aims to process the algorithm step by step

# import part


import os
from typing import Literal

from pymanopt.autodiff.backends import pytorch

os.environ["OMP_NUM_THREADS"] = "1"
import torch
from MVHSC_Aux import data_importation, initialization, clustering, evaluation, lower_level, upper_level
from torch.optim import Adam, SGD





DI = data_importation(view2=0)
data = DI.get_data()

IN = initialization(DI)
Theta, F = IN.initial()
CL = clustering()
EV = evaluation(DI, CL)
#
learning_rate = 0.01
lambda_r = 1

# 建立下层模型
LL = lower_level(F["LL"])
LLOP = Adam(LL.parameters(), lr = learning_rate)

UL = upper_level(F["UL"])
ULOP = SGD(UL.parameters(), lr = learning_rate)

# 测试结果
def do_assess(F_ul, F_ll, type:Literal["UL","LL","both"]):
    if type in ["UL", "both"]:
        nmi, _ = EV.assess(F_ul)
        print(f"ul_nmi:{nmi}")

    if type in ["LL", "both"]:
        nmi, _ = EV.assess(F_ll)
        print(f"ll_nmi:{nmi}")

def update_value(F, grad_F, learning_rate = 0.01, method:bool=True):
    F -= learning_rate * grad_F
    if method:
        F,_ = torch.linalg.qr(F, mode= "reduced")
    return F

def judge_orth(F):
    FTF = F.T @ F
    norm2 = torch.linalg.norm(FTF, ord = 2)
    return norm2

def judge_orths(F_UL, F_LL):
    norm_LL = judge_orth(F_LL)
    norm_UL = judge_orth(F_UL)
    print(f"norm_UL:{norm_UL},norm_LL:{norm_LL}")

do_assess(F["UL"],F["LL"],"both")
judge_orths(F["UL"], F["LL"])

epochs = 60

grad_method = "man"
for _ in range(1):

    for epoch in range(epochs):
        # 优化下层变量
        LL_val = LL(F["UL"],Theta["LL"], lambda_r)
        match grad_method:
            case "man":
                grad_ll = 2*(torch.eye(F["LL"].shape[0])-F["LL"]@F["LL"].T)@ Theta["LL"]@F["UL"]
            case "auto":
                (-LL_val).backward()
                grad_ll = LL.F_ll.grad.clone()
        if epoch < epochs-1:
            F["LL"] = update_value(F["LL"], grad_ll, learning_rate/(epoch+1), False)
            F[f"LL{epoch}"] = F["LL"]
            F[f"grad_LL{epoch}"] = grad_ll
            print("更新F['LL]，未使用正交化")
        else:
            F["LL"] = update_value(F["LL"], grad_ll, learning_rate/(epoch+1))
            F[f"LL{epoch}"] = F["LL"]
            F[f"grad_LL{epoch}"] = grad_ll
            print("更新F['LL]，使用了正交化")


        do_assess(F["UL"],F["LL"],"LL")
        print(torch.linalg.norm(grad_ll, ord =2).item())
        judge_orths(F["UL"], F["LL"])


    print("-"*50)

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
        # do_assess(F["UL"],F["LL"],"LL")

    # judge_orths(F["UL"], F["LL"])
    print("-"*50)



print("aa")