import matplotlib.pyplot as plt
import numpy as np

x1 = np.random.random(10)
y1 = np.random.random(10)
x2 = np.random.random(10)
y2 = np.random.random(10)

symbolsdic = {0: 'o', 1: '^', 2: 's', 3: '*', 4: 'D', 5: '+', 6: '8', 7: 'd', 8: 'H', 9: 'v'}

for i in range(10):
    plt.plot(x1[i], y1[i], symbolsdic[i], label=str(i), color='red')
    plt.plot(x2[i], y2[i], symbolsdic[i], color='blue')

plt.legend()

plt.show()
