from sympy import *
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import math
from astropy import constants as const
from STMint.STMint import STMIntegrator

#print(dir(STMint))

x, y, z, vx, vy, vz, ux, uy, uz = symbols("x,y,z,vx,vy,vz,ux,uy,uz")
var = [x, y, z, vx, vy, vz, ux, uy, uz]
print(const.GM_earth)
print(const.GM_earth.value * 10**(-9))

#in km and s units
mu = const.GM_earth.value * 10**(-9)
V = mu / sqrt(x**2 + y**2 + z**2)
r = Matrix([x, y, z])
u = Matrix([ux, uy, uz])
vr = Matrix([vx, vy, vz])
zeros = Matrix([0, 0, 0])

dVdr = diff(V, r)

dynamics = Matrix.vstack(vr, dVdr + u, zeros)

mySTMint = STMIntegrator(var, dynamics)

# only need to create from here once
#solf = mySTMint.dyn_int([0,(2*math.pi)], [1,0,0,0,1,0,0,0,0], max_step=.1)

a = 42164.
vCirc = math.sqrt(mu / a)
period = 2. * math.pi * math.sqrt(a**3 / mu)
sol1 = mySTMint.dynVar_int([0, (period)], [a, 0, 0, 0, vCirc, 0, 0, 0, 0],
                           max_step=100)

print(np.reshape(sol1.y[9:, -1], (9, 9)))  # phi for final time step
print(sol1.y[:9, -1])  # y for final time step.
