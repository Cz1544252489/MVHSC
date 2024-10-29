# This file aims to solve the problem referred to as Multi-View Hypergraph Spectral Clustering(MVHSC)

import os

os.environ["OMP_NUM_THREADS"] = "1"
import torch
from MVHSC_Aux import data_importation, initialization, clustering, evaluation, lower_level, upper_level
from torch.optim import Adam, SGD

view_num = 2
DI = data_importation(view_num)
data = DI.get_data()
IN = initialization(DI)
Theta, F = IN.initial()
CL = clustering()
EV = evaluation()
#
learning_rate = 0.01
lambda_r = 1

# 建立下层模型
LL = lower_level(F["var"])
LLOP = Adam(LL.parameters(), lr = learning_rate)

UL = upper_level(F["star"])
ULOP = Adam(UL.parameters(), lr = learning_rate)

labels_pred = CL.cluster(F["var"],6)
nmi = EV.calculate_nmi(data["labels_true_bbc_guardian"],labels_pred)
print(f"ll_nmi:{nmi}")


for epoch in range(4):
    for i in range(20):

        LL_val = LL(F["star"],Theta["var"], lambda_r)
        LL_val.backward()
        LLOP.step()

        if i%5 ==0 :
            labels_pred = CL.cluster(F["var"],6)
            LL_nmi = EV.calculate_nmi(data["labels_true_bbc_guardian"],labels_pred)
            print(f"ll_nmi:{LL_nmi}")

    labels_pred = CL.cluster(F["star"],6)
    nmi = EV.calculate_nmi(data["labels_true_bbc_guardian"],labels_pred)
    print(f"ul_nmi:{nmi}")

    UL_val = UL(F["var"], lambda_r)
    UL_val.backward()
    ULOP.step()

    labels_pred = CL.cluster(F["star"],6)
    nmi = EV.calculate_nmi(data["labels_true_bbc_guardian"],labels_pred)
    print(f"ul_nmi:{nmi}")

    print("*"*40)


print("aa")


