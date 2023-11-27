import numpy as np
import scipy
from matplotlib import pyplot as plt

stefans=5.670374419*0.00000001
x=np.loadtxt('dataset3.txt')[:,0]
y=np.loadtxt('dataset3.txt')[:,1]
# Moving average
for i in range(2989):
    y[i+5]=0.1*sum(y[i:i+10])

L=scipy.integrate.trapezoid(y,dx=x[9]-x[10])
# Integrating plancks intensity distribution gives Stefan's law
# https://www.spectralcalc.com/blackbody/integrate_planck.html
print('T= ',(L*np.pi/stefans)**0.25,'K')

