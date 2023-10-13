if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from dezero import Variable
import dezero.functions as F

x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y = F.reshape(x, (6,))
# y = F.reshape(x, 6)
y.backward(retain_grad=True)
# y.backward()
print(y, x.grad)

x = Variable(np.random.randn(1, 2, 3))
y = x.reshape((2, 3))
y = x.reshape([2, 3])
y = x.reshape(2, 3)
print(x)

x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y = F.transpose(x)
y.backward()
print(y, x.grad)

x = Variable(np.random.rand(2, 3))
y = x.T
print(x, y)

A, B, C, D = 1, 2, 3, 4
x = np.random.rand(A, B, C, D)
y = x.transpose(1, 0, 3, 2)
print(x, y)
