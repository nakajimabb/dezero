import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creater = None

    def set_creater(self, func):
        self.creater = func

    def backward(self):
        f = self.creater
        if f is not None:
            x = f.input
            x.grad = f.backward(self.grad)
            x.backward()

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creater(self)
        self.input = input
        self.output = output
        return output

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx


def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

assert y.creater == C
assert y.creater.input == b
assert y.creater.input.creater == B
assert y.creater.input.creater.input == a
assert y.creater.input.creater.input.creater == A
assert y.creater.input.creater.input.creater.input == x

y.grad = np.array(1.0)
y.backward()
print(x.grad)
