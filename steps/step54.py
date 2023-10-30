if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from dezero import test_mode
import dezero.functions as F

np.random.seed(0)

x = np.ones(5)
print(x)

# train mode
y = F.dropout(x)
print(y)

# test mode(for predict)
with test_mode():
    y = F.dropout(x)
    print(y)
