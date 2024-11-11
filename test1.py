import torch
import torch.nn as nn
from MVHSC_Aux import data_importation, initialization, evaluation, clustering, iteration

DI = data_importation()
IN = initialization(DI)
CL = clustering()
EV = evaluation(DI, CL)

settings = {"learning_rate": 0.01, "lambda_r": 1, "epsilon": 0.05, "update_learning_rate": True,
             "max_ll_epochs": 30, "max_ul_epochs": 20, "orth1": False,
             "orth2": True, "update_lambda_r": False, "use_proj": True,
             "plot_vline": True, "grad_method":"auto"}

IT = iteration(EV, IN, settings)

class AA(nn.Module):

    def __init__(self, x, y):
        super().__init__()
        self.x = nn.Parameter(x)
        self.y = nn.Parameter(y)

    def forward(self):
        z = self.x * self.y
        return z


f = AA(torch.tensor(10.1),torch.tensor(20.0))
val = f()

grad_x = torch.autograd.grad(val,f.x,retain_graph=True)
grad_y = torch.autograd.grad(val,f.y, retain_graph=True)


print("aa")