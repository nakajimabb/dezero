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

# x, t = dezero.datasets.get_spiral(train=True)
# print(x.shape)
# print(t.shape)

# print(x[10], t[10])
# print(x[110], t[110])

# markers = ["o", "x", "^"]
# colors = ["orange", "blue", "green"]
# for i in range(len(x)):
#     c = t[i]
#     plt.scatter(x[i][0], x[i][1], s=40, marker=markers[c], c=colors[c])
# plt.show()

# ①ハイパーパラメータの設定
max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

# ②データの読み込み / モデル・オプティマイザの生成
x, t = dezero.datasets.get_spiral(train=True)
model = MLP((hidden_size, 3))
optimizer = optimizers.SGD(lr).setup(model)

data_size = len(x)
max_iter = math.ceil(data_size / batch_size)
losses = []

for epoch in range(max_epoch):
    # ③データセットのインデックスのシャッフル
    index = np.random.permutation(data_size)
    sum_loss = 0

    for i in range(max_iter):
        # ④ミニバッチの生成
        batch_index = index[i * batch_size : (i + 1) * batch_size]
        batch_x = x[batch_index]
        batch_t = t[batch_index]
        # ⑤勾配の算出 / パラメータの更新
        y = model(batch_x)
        loss = F.softmax_cross_entropy(y, batch_t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(batch_t)

    # ⑥エポックごとに学習経過を出力
    avg_loss = sum_loss / data_size
    losses.append(avg_loss)
    print("epoch %d, loss %.2f" % (epoch + 1, avg_loss))

plt.plot(np.arange(max_epoch), losses)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

# Plot boundary area the model predict
h = 0.001
x_min, x_max = x[:, 0].min() - 0.1, x[:, 0].max() + 0.1
y_min, y_max = x[:, 1].min() - 0.1, x[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
X = np.c_[xx.ravel(), yy.ravel()]

with dezero.no_grad():
    score = model(X)
predict_cls = np.argmax(score.data, axis=1)
Z = predict_cls.reshape(xx.shape)
plt.contourf(xx, yy, Z)

# Plot data points of the dataset
markers = ["o", "x", "^"]
colors = ["orange", "blue", "green"]
for i in range(len(x)):
    c = t[i]
    plt.scatter(x[i][0], x[i][1], s=40, marker=markers[c], c=colors[c])
plt.show()
