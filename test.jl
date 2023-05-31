# Import necessary libraries
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Define the SIR model differential equations
def sir_model(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# Define initial conditions
N = 1000
I0, R0 = 1, 0
S0 = N - I0 - R0
y0 = S0, I0, R0

# Define parameters
beta = 0.2
gamma = 1./10

# Define time points
t = np.linspace(0, 200, 200)

# Solve the SIR model differential equations
sol = odeint(sir_model, y0, t, args=(beta, gamma))

# Plot the results
plt.plot(t, sol[:, 0], label='S(t)')
plt.plot(t, sol[:, 1], label='I(t)')
plt.plot(t, sol[:, 2], label='R(t)')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Number of individuals')
plt.title('SIR Model')
plt.show()
