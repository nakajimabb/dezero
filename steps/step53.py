if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import dezero
from dezero import optimizers
import dezero.functions as F
from dezero.models import MLP
from dezero import DataLoader

np.random.seed(0)

max_epoch = 5
batch_size = 100

train_set = dezero.datasets.MNIST(train=True)
train_loader = DataLoader(train_set, batch_size)
model = MLP((1000, 10))
optimizer = optimizers.SGD().setup(model)

# パラメータの読み込み
if os.path.exists("my_mlp.npz"):
    model.load_weights("my_mlp.npz")
    print("weights data loaded")

for epoch in range(max_epoch):
    sum_loss = 0

    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        sum_loss += float(loss.data) * len(t)

    print("epoch: {}".format(epoch + 1))
    print("train loss: {:.4f}".format(sum_loss / len(train_set)))

model.save_weights("my_mlp.npz")
