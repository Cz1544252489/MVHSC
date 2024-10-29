# This file aims to process the algorithm step by step

# import part

import os

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from MVHSC_Aux import data_importation, initialization, clustering, evaluation, test_part

# 测试部分

# 测试基本聚类的结果


DI = data_importation()
CL = clustering()
EV = evaluation()
data = DI.data
IN = initialization(data, DI.device)
TP = test_part(DI,CL,EV,IN)

data1 = DI.get_data()
# data1 = {}
# sources = DI.sources
# i,j = 0,1
# data1[f"docs_{sources[i]}_{sources[j]}"] = np.intersect1d(data[f"{sources[i]}_docs"], data[f"{sources[j]}_docs"])
# labels_true = DI.get_labels_from_sample(data1[f"docs_{sources[i]}_{sources[j]}"])
# temp = [np.where(data[f"{sources[i]}_docs"]== value)[0].item() for value in data1[f"docs_{sources[0]}_{sources[1]}"]]
# data1[f"{sources[i]}_mtx_{sources[i]}_{sources[j]}"] = data["bbc_mtx"].T[temp]
# labels_pred = CL.cluster(data1[f"{sources[i]}_mtx_{sources[i]}_{sources[j]}"], 6)
# nmi = EV.calculate_nmi(labels_true, labels_pred)
# TP.cluster_and_evaluation(3,"file")

# with open("output.txt","w") as file:
#     for view in DI.sources:
#         labels_pred = CL.cluster(data[f'{view}_mtx'],6, method="spectral")
#         nmi = EV.calculate_nmi(data[f"{view}_labels_true"], labels_pred)
#         print(f"SC_{view}:{nmi}", file=file)
#
#     for view in DI.sources:
#         _, F = IN.get_Theta_and_F(data[f'{view}_mtx'],6)
#         labels_pred = CL.cluster(F,6, method="spectral")
#         nmi = EV.calculate_nmi(data[f"{view}_labels_true"], labels_pred)
#         print(f"HSC_{view}:{nmi}", file=file)

print("aa")