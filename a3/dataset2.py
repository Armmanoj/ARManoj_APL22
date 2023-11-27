import numpy as np
import scipy
from matplotlib import pyplot as plt

def sinecoeff(x,y,T,k):
    # k is the number of coefficients we want to calculate
    f=1/T
    step=x[1]-x[0]
    StepsInT=int(T//step)
    alpha=np.zeros(k)
    M=len(x)
    for j in range(k):
        # Add over 2 time periods to get the average of coefficients from both cycles
        sin2pifx=np.sin(2*np.pi*f*j*x[M//2-StepsInT:M//2+StepsInT]) # sin function from -T to +T
        alpha[j]=np.dot(sin2pifx,y[M//2-StepsInT:M//2+StepsInT])
    return T*alpha/M # As 2 timeperiods are used, division is by M/T, not 2M/T 
    
x=np.loadtxt('dataset2.txt')[:,0]
y=np.loadtxt('dataset2.txt')[:,1]
N=len(y)
print(N)

# Moving average to smoothen data and attenuate high frequency component,
for i in range(N-10):
    y[i+5]=0.1*sum(y[i:i+10])
    
# The signal is assuumed to be periodic. Hence, I will apply autocorrelation to extract the time period of the signal.
zauto=np.zeros(N) # z is the autocorrelation, it's length is 2N but is symmetric about the middle, and so only 1st half is calculated
for i in range(N):
    zauto[i]=np.dot(y[0:i+1],y[N-i-1:N])

plt.plot([i-3 for i in x],zauto)
plt.xlabel("x")
plt.ylabel("Autocorr[y]")
plt.savefig("Atocorr.png")
# By visual inspection, the x-values of the peaks are;
# Peaks are +ve:(0.52,3), -ve:(1.77,-0.75), the average peak-to-peak distance is 2.5. Hence, time period is 2.5
T=2.5 # time period
f=0.4 # Fundamental freqency is 0.4

# It is given that the signal is a sum of sine waves, also, the signal has value 0 at x=0

print(sinecoeff(x,y,T,10))  #output is shown below, clearly, f,3f,5f are the important frequencies 
# [ 0.00000000e+00  4.99834857e+00  1.28442540e-03  1.61355402e+00
#  9.18092881e-03  7.53817115e-01 -6.41446306e-03  3.79051631e-03
# -1.67601659e-02 -9.52726423e-03]
sincoeffs=sinecoeff(x,y,T,10)
bestfitsine=sincoeffs[1]*np.sin(2*np.pi*f*x)+sincoeffs[3]*np.sin(2*3*np.pi*f*x)+sincoeffs[5]*np.sin(2*5*np.pi*f*x)
# The discrete sine series coefficients are also the coefficients for best fit curve
plt.plot(x,y,label="noisy signal")
plt.plot(x,bestfitsine,label="cleaned signal")
plt.title("Noisy and cleaned signals")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(loc="center left")
plt.savefig("sig2.png")

# Using scipy
def scineapprox(x,a,b,c):
    return a*np.sin(0.8*np.pi*x)+b*np.sin(2.4*np.pi*x)+c*np.sin(4*np.pi*x)

initial_guess = [1.0, 1.0, 1.0]  # Initial guess for parameters (a, b, c)
params, covariance = scipy.optimize.curve_fit(scineapprox, x, y, p0=initial_guess)

# Extract optimized parameters
a,b,c = params
bestfitsine=scineapprox(x,a,b,c)
plt.plot(x,y,label="noisy signal")
plt.plot(x,bestfitsine,label="cleaned signal")
plt.title("Noisy and cleaned signals")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(loc="center left")
plt.savefig("sig3.png")
