import os

os.environ["OMP_NUM_THREADS"] = "1"
import torch
from MVHSC_Aux import data_importation, initialization, clustering, evaluation, lower_level, upper_level
from torch.optim import Adam, SGD


DI = data_importation(view2=4)
data = DI.get_data()




print("aa")