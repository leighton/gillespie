#coding=utf-8

from sympy import symbols
import numpy as np
import matplotlib.pyplot as plt
from stochastic import integrate

X, Y, Z = symbols('X Y Z')
B, l, mu, nu = symbols('β λ μ ν')

x = [X, Y, Z]
T = [
    {X: -1, Y: 1},
    {X: -1, Y: 1},
    {X:  1},
    {Y: -1}
]
prop = [
  B*X*Y,
  nu*X,
  l,
  mu*Y
]


ics = {
  X: 50,
  Y: 0
}
parm = {
  B: 0.001,
  l: 2,
  mu: 0.05,
  nu: 0.01,
}

print "Start simulation"
y = integrate(x, T, prop, parm, ics, 500)
print "Stop simulation"

t, x1, x2, x3 = np.transpose(y);
plt.plot(t, x1, t, x2, t, x3)
plt.show()



