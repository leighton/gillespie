#coding=utf-8

from sympy import symbols
import numpy as np
import matplotlib.pyplot as plt
from gillespie import integrate
from gillespie import cython_propensity_function

X, Y, Z = symbols('X Y Z', integer=True)
B, l, mu, nu = symbols('β λ μ ν', real=True)

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
  nu: 0.01
}
#import profile

#print "Start simulation"
def run():
  y = list(integrate(x, T, prop, parm, ics, 500))
  return y

x = run()
t, x1, x2, x3 = np.transpose(x);
plt.plot(t, x1, t, x2, t, x3)
plt.show()

x = run()


#profile.run("run(); print")




