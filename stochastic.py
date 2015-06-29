import numpy as np
from random import random
from sympy.utilities.lambdify import lambdify, lambdastr
from sympy.utilities.codegen import codegen
import sympy.mpmath as mp

#def create_c_propensity_function(x, parm, prop):
#  [(c_name, c_code), (h_name, c_header)] = codegen(("propensity", np.array(prop)), "C", "test", header=False, empty=False)
#  print c_code

def create_propensity_function(x, parm, prop):
  args = x+parm.keys()
  return (lambdify(args, prop), args)

"""
 s: symbolic states of the model
 T: vector of sparse vectors of transitions of between states
 p: vector of transitiion probabilities
 ics: intitial conditions
 t: simulation duration
"""
def integrate(x, T, prop, parm, ics, tmax, method="direct"):

  if len(T) != len(prop):
    raise Error("Transition and propensity vectors must be of same size")

  #time!
  t = 0

#  create_c_propensity_function(x, parm, prop)

  #convert sparse state space to dense state space
  xv = dict([(xi, ics.get(xi, 0)) for xi in x])

  #create initial state vector with zeroth realisation
  #TODO: replace with sparse vector
  y = [ [t] + [xv[xi] for xi in x] ]

  model_map = xv.copy()
  model_map.update(parm)

  prop_fn, prop_args = create_propensity_function(x, parm, prop)

  while t < tmax:

    #evaluate propensity vector
    #propv = np.array(map(lambda p: p.evalf(subs=model_map), prop))

    args = [model_map[p] for p in prop_args]
    propv = np.array(prop_fn(*args))

    #calculate total propensity
    propt = np.sum(propv)

    #no more propensity, system is at rest
    if propt == 0:
      raise Exception("No more propensity? "+ str(propv))
      return y

    #determine transition probability
    #TODO: allow for negative values? e.g. abs(propv)/np.sum(abs(propv))
    prob = propv/propt

    r = random()

    #determine future event time
    tau = (1/propt)*np.log(1/r)

    rxn = np.random.choice(len(prop), 1, p=list(prob))[0]

    for state, delta in T[rxn].items():
      xv[state] = xv[state] + delta
      model_map[state] = model_map[state] + delta

    t+=tau

    yi = [t]+[xv[xi] for xi in x]
    y.append(yi)

  return y

