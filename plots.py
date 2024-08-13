import numpy as np
import matplotlib.pyplot as plt

array_loaded = np.load('P_t_all.npy')
n=20
time = np.linspace(0, n,n)

plt.plot(time,array_loaded)
plt.xlabel('Time[s]')
plt.ylabel('$P_t(beta=1)$')
plt.title('Probability Distributions for Different Prediction Horizons')
plt.legend()
plt.grid(True)
plt.xticks(np.arange(0, 6, 1))
plt.show()