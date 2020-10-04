from scipy.integrated import odeint
import numpy as np
import matplotlib.pyplot as plt

# population
N = 1000
# init infected, recovered
I, R = 1, 0
# susceptibe (evreyone else)
S = N - I + R

# contact rate beta(b), recovery rate gama (k)
b, k = 0.2, 1/10
# time (days)
t = np.linspace(0, 160, 160)

def derive(y, t, N, b, k):
    S,I,R = y
    dS = -b * S * I/N
    dR = k * I
    dI = b * S * I/N - dR
    return dS, dI, dR

y0 = S, I, R
ret = odeint(derive, y0, t, args=(N,b,k))
S,I,R = ret.T

plt.plot(t, S/N, 'b-', label='Suceptibe')
plt.plot(t, I/N 'r-', label='Infected')
plt.plot(t, R/N, 'g-', label='Recoverd')
plt.legend()
plt.show()
