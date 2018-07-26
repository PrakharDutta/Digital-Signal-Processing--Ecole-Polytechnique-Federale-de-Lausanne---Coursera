# module 5 exercises

import numpy as np
import matplotlib.pyplot as plt


# 10
def lagrange_poly(k, N, t):
    res = 1
    for i in np.arange(-N, N+1, 1):
        if i != k:
            res *= (t - i)/(k - i)
    return res

def lag_interp(s, k, N, t):
    res = 0
    for ss, kk in zip(s, k):
        res += ss*lagrange_poly(kk, 1, t)
    return res

if __name__ == "__main__":
    s = np.array([4, -2, -3])
    k = np.array([-1, 0, 1])
    res = lag_interp(s, k, 1, 1/4)
    print(res)

    x = np.linspace(-1, 1, 50)
    plt.plot(x, lag_interp(s, k, 1, x))
    plt.show()
