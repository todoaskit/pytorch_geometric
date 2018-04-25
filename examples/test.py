import numpy as np

a = np.array([[5, 3, 1],
              [3, 4, 6],
              [1, 6, 7]])

b = np.array([[1, 5, 3],
              [5, 2, 4],
              [3, 4, 3]])

w, v = np.linalg.eig(a)
print(w)
print(v)

w, v = np.linalg.eig(b)
print(w)
print(v)

c = np.array([[5, 3, 1, 0, 0, 0],
              [3, 4, 6, 0, 0, 0],
              [1, 6, 7, 0, 0, 0],
              [0, 0, 0, 1, 5, 3],
              [0, 0, 0, 5, 2, 4],
              [0, 0, 0, 3, 4, 3]])

w, v = np.linalg.eig(c)
print(w)
print(v)