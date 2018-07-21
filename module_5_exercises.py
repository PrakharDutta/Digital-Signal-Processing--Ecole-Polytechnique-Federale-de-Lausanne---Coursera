# module 5 exercises

import numpy as np
import matplotlib.pyplot as plt

# 13

Fs = 8000  # Hertz
f0 = 0     # Hertz
f1 = 10    # Hertz
t1 = 2     # seconds

t = np.linspace(0, 2, Fs*t1)

alpha = (f1-f0)/t1

x = np.cos(2*np.pi*f0*t + alpha*np.pi*t**2)
plt.plot(t, x)
plt.show()