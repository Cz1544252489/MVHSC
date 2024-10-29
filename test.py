#this file aims to process the algorithm step by step

# import part


import os

os.environ["OMP_NUM_THREADS"] = "1"
import torch
from MVHSC_Aux import data_importation, initialization, clustering, evaluation, lower_level, upper_level
from torch.optim import Adam, SGD


DI = data_importation(view2=4)
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

nmi, ari = EV.assess(F["LL"])
print(f"ll_nmi:{nmi} ari:{ari}")

nmi, ari = EV.assess(F["UL"])
print(f"ul_nmi:{nmi} air:{ari}")










print("aa")