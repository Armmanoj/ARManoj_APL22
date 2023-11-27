import numpy as np
from matplotlib import pyplot as plt

data = np.loadtxt('dataset1.txt')
x = data[:,0]
y = data[:,1]
Ey=np.mean(y)
Ex=np.mean(x)
covxy=np.cov(x,y)[1,0]
varx=np.var(x)
slope=covxy/varx
xintercept=Ey-slope*Ex
print(xintercept)
print(slope)
yerr=np.std(y)*((1-np.corrcoef(x,y)[0][1]**2)**0.5)

plt.plot(x,y,label="noisy signal")
plt.plot(x,[xintercept+i*slope for i in x],label="estimated")
plt.errorbar(x[::25], y[::25], yerr=np.array([yerr for i in range(len(x[::25]))]),fmt='o',markersize=4, label='Error Bars')
plt.legend(loc="upper left")
plt.savefig("Leastsq.png")
