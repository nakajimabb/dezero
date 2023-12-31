if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import dezero
from dezero.models import VGG16
from PIL import Image

url = (
    "https://github.com/oreilly-japan/deep-learning-from-scratch-3/raw/images/zebra.jpg"
)
img_path = dezero.utils.get_file(url)
img = Image.open(img_path)

x = VGG16.preprocess(img)
x = x[np.newaxis]

model = VGG16(pretrained=True)
with dezero.test_mode():
    y = model(x)
predict_id = np.argmax(y.data)

model.plot(x, to_file="vgg.pdf")
labels = dezero.datasets.ImageNet.labels()
print(labels[predict_id])
