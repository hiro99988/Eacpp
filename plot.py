import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt('out/data/result.txt')
x = data[:, 0]
y = data[:, 1]

plt.scatter(x, y, s=1)
plt.show()