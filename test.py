import numpy as np

a = np.array([[1],[2],[3]])
print(a.shape)
a = np.append(a,[4])
print(a.shape)
a = a.reshape(len(a),1)
print(a.shape)
print(a)