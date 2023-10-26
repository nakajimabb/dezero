if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import math
import numpy as np
import matplotlib.pyplot as plt
import dezero
from dezero import optimizers
import dezero.functions as F
from dezero.models import MLP
from dezero.datasets import Spiral
from dezero import DataLoader

# batch_size = 10
# max_epoch = 1

# train_set = Spiral(train=True)
# test_set = Spiral(train=False)
# train_loader = DataLoader(train_set, batch_size)
# test_loader = DataLoader(test_set, batch_size, shuffle=False)

# for epoch in range(max_epoch):
#     for x, t in train_loader:
#         print(x.shape, t.shape)
#         break

#     for x, t in test_loader:
#         print(x.shape, t.shape)
#         break

# y = np.array([[0.2, 0.8, 0], [0.1, 0.9, 0], [0.8, 0.1, 0.1]])
# t = np.array([1, 2, 0])
# acc = F.accuracy(y, t)
# print(acc)

max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

train_set = Spiral(train=True)
test_set = Spiral(train=False)
train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

model = MLP((hidden_size, 3))
optimizer = optimizers.SGD(lr).setup(model)

losses = []
test_losses = []
accs = []
test_accs = []

for epoch in range(max_epoch):
    sum_loss, sum_acc = 0, 0

    for x, t in train_loader:  # ①訓練用のミニバッチデータ
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)  # ②訓練データの認識精度
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc.data) * len(t)

    print("epoch: {}".format(epoch + 1))
    print(
        "train loss: {:.4f}, accuracy: {:.4f}".format(
            sum_loss / len(train_set), sum_acc / len(train_set)
        )
    )
    losses.append(sum_loss / len(train_set))
    accs.append(sum_acc / len(train_set))

    sum_loss, sum_acc = 0, 0
    with dezero.no_grad():  # ③勾配不要モード
        for x, t in test_loader:  # 　④訓練用のミニバッチデータ
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)  # ⑤テストデータの認識精度
            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)

    print(
        "test loss: {:.4f}, accuracy: {:.4f}".format(
            sum_loss / len(test_set), sum_acc / len(test_set)
        )
    )
    test_losses.append(sum_loss / len(test_set))
    test_accs.append(sum_acc / len(test_set))


plt.plot(np.arange(max_epoch), losses, label="train")
plt.plot(np.arange(max_epoch), test_losses, label="test")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.show()

plt.plot(np.arange(max_epoch), accs, label="train")
plt.plot(np.arange(max_epoch), test_accs, label="test")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend()
plt.show()
