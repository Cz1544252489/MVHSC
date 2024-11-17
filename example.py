import torch.linalg
import torch.nn as nn

class upper_level(nn.Module):

    def __init__(self,x ,y):
        super().__init__()
        self.x = nn.Parameter(x)
        self.y = nn.Parameter(y)
        self.e = torch.ones((1,100), dtype=torch.float64)

    def forward(self, z):
        term = torch.linalg.norm(self.x-z, ord=4) + torch.linalg.norm(self.y - self.e, ord=4)
        return term

class lower_level(nn.Module):

    def __init__(self, y, z):
        super().__init__()
        self.y = nn.Parameter(y)
        self.z = nn.Parameter(z)

    def forward(self, x):
        term = torch.linalg.norm(self.y, ord=2) - x @ y.T
        return term
torch.random.seed()
x = torch.rand((1,100), dtype=torch.float64)
y = torch.rand((1,100), dtype=torch.float64)
z = torch.rand((1,100), dtype=torch.float64)

UL = upper_level(x ,y)
LL = lower_level(y, z)

for _ in range(5):
    ll_val = LL(x)
    grad_ll_y = torch.autograd.grad(ll_val, LL.y)[0]
    y = LL.y - 0.1* grad_ll_y
    norm_grad = torch.linalg.norm(grad_ll_y, ord=2)
    print(f"val:{ll_val.item()}, norm_of_grad:{norm_grad}")
    with torch.no_grad():
        LL.y.copy_(y)
# ll_val.backward()


print("aa")