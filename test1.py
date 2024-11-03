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

def output_type(flag):
    output = {}
    if "nmi" in flag :
        output["ll_nmi"] = result["ll_nmi"]
        output["ul_nmi"] = result["ul_nmi"]
    if "val" in flag:
        output["ll_val"] =result["ll_val"]
        output["ul_val"] =result["ul_val"]
    return output

EV.plot_result(output_type(["nmi","val"]))


print("aa")