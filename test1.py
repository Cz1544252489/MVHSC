import os

os.environ["OMP_NUM_THREADS"] = "1"
import torch
from MVHSC_Aux import data_importation, initialization, clustering, evaluation, lower_level, upper_level
import json


DI = data_importation(view2=0)
data = DI.get_data()

IN = initialization(DI)
Theta, F = IN.initial()
CL = clustering()
EV = evaluation(DI, CL)

result = EV.use_result({}, "load")

output = {"ll_nmi":result["ll_nmi"],"ul_nmi":result["ul_nmi"]}
      #    "ll_val":result["ll_val"],"ul_val":result["ul_val"]
EV.plot_result(output)


print("aa")