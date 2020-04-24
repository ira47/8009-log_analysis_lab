import torch
from torch.autograd import Variable
# requires_grad 表示是否对其求梯度，默认是False
x = Variable(torch.Tensor([3]), requires_grad=True)
y = Variable(torch.Tensor([5]), requires_grad=True)
z = 2 * x * x + y * y + 4
# 对 x 和 y 分别求导
z.backward()
# x 的导数和 y 的导数
#x =3*2*2
#y = 5*2
print('dz/dx: {}'.format(x.grad.data))
print('dz/dy: {}'.format(y.grad.data))
