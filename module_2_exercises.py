# Week 3 exercises - Markov Chains

import numpy as np

# 2
v0 = np.array([1/2, 1/2, 1/2, 1/2])
v1 = np.array([1/2, 1/2, -1/2, -1/2])
v2 = np.array([1/2, -1/2, 1/2, -1/2])
v3 = np.array([1/2, -1/2, -1/2, 1/2])

print(np.dot(v0, v3))
print(np.dot(v1, v3))
print(np.dot(v2, v3))

# 3
y = np.array([0, 1, 1, 2])
A = np.matrix([v0, v1, v2, v3])
print(np.dot(np.transpose(A), y))

# 4
A1 = np.matrix([y, v0, v1, v2])
A2 = np.matrix([y, v0, v2, v3])
A3 = np.matrix([y, v1, v2, v3 - 2*v1])
A4 = np.matrix([y, v1, v2, v3])

print(np.linalg.det(A1))
print(np.linalg.det(A2))
print(np.linalg.det(A3))
print(np.linalg.det(A4))


# 6
P1 = np.matrix([[0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [1, 0, 0, 0]
                ])
print(P1**4)