import torch
from torch import nn
import numpy as np


batch_size = 10
hidden_dim = 20
input_dim = 3
output_dim = 2 

model = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, output_dim)).double()
x = torch.rand(batch_size, input_dim, requires_grad=True, dtype=torch.float64) #(batch_size, input_dim)
y = model(x) #y: (batch_size, output_dim) 

#using torch.autograd.grad
dydx1 = torch.autograd.grad(y, x, retain_graph=True, grad_outputs=torch.ones_like(y))
dydx1 = dydx1[0]  #dydx1: (batch_size, input_dim)
print(f' using grad dydx1: {dydx1.shape}')

#using torch.autograd.functional.jacobian
j = torch.autograd.functional.jacobian(lambda t: model(t), x) #j: (batch_size, output_dim, batch_size, input_dim)

#the off-diagonal elements of 0th and 2nd dimension are all zero. So we remove them
dydx2 = torch.diagonal(j, offset=0, dim1=0, dim2=2) #dydx2: (output_dim, input_dim, batch_size)
dydx2 = dydx2.permute(2, 0, 1) #dydx2: (batch_size, output_dim, input_dim)
print(f' using jacobian dydx2: {dydx2.shape}')

#round to 14 decimal digits to avoid noise 
print(np.round((dydx2.sum(dim=1)).numpy(), 14) == np.round(dydx1.numpy(), 14))




print("IS BATCHED TESTING")
x = torch.randn(2, 2, requires_grad=True)

# Scalar outputs
out = x.sum()  # Size([])
batched_grad = torch.arange(3)  # Size([3])
grad, = torch.autograd.grad(out, (x,), (batched_grad,), is_grads_batched=True)
print("is batched", grad)

# loop approach
grads = torch.stack(([torch.autograd.grad(out, x, torch.tensor(a))[0] for a in range(3)]))
print("looped ", grads)



print("MY OWN")
x = torch.randn(3, 2, requires_grad=True)
a = torch.tensor([[1,2],[3,4],[5,6]])
y = torch.sum(x*a, dim=-1)
print("x / y shape", x.shape, y.shape)
grad1 = torch.autograd.grad(torch.sum(y), x)[0]
print("grad1", grad1)
x = torch.randn(3, 2, requires_grad=True)
y = torch.sum(x*a, dim=-1)
grad2 = torch.autograd.grad(y, x, create_graph=False, grad_outputs=torch.ones_like(y))[0]
print("grad2", grad2)
grad3 = torch.autograd.grad(y[0], x)[0]
print("grad3", grad3)

