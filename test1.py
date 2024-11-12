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

class BB(nn.Module):
    def __init__(self,x,y):
        super().__init__()
        self.x = nn.Parameter(x)
        self.y = nn.Parameter(y)

    def forward(self, t):
        z = self.x * self.y * t
        return z

i = torch.tensor(1.0)
j = torch.tensor(2.0)
k = torch.tensor(3.0)


f = AA(i,j)
g = BB(i,j)
val1 = f()
val2 = g(k)
grad_x1 = torch.autograd.grad(val1,f.x,retain_graph=True)
grad_y1 = torch.autograd.grad(val1,f.y, retain_graph=True)

grad_x2 = torch.autograd.grad(val2,g.x,retain_graph=True)
grad_y2 = torch.autograd.grad(val2,g.y, retain_graph=True)

print("aa")