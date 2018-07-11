# module 4 part 2 exercises

import numpy as np


def delta(n):
    """
    Discrete delta functional
    :param n: sample
    :return: 1 if n=0, 0 otherwise
    """
    return n == 0


def x(n):
    return delta(n) + 1/2*delta(n-1)

N = 51     # The number of samples
y = np.zeros(N)
y[0] = x(1) - 1/2*x(0)    # Initial condition for y

for i in np.arange(1, N):
    y[i] = -2*y[i-1] + x(i+1) - 1/2*x(i)

print(y)
