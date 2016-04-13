import numpy as np
from numpy.random import random
from sympy.utilities.lambdify import lambdify, lambdastr
from sympy.utilities.codegen import codegen
import sympy.mpmath as mp
import cython

from sympy.utilities.autowrap import ufuncify
from sympy.printing.theanocode import theano_function

import theano

#def create_c_propensity_function(x, parm, prop):
#  [(c_name, c_code), (h_name, c_header)] = codegen(("propensity", np.array(prop)), "C", "test", header=False, empty=False)
#  print c_code

"""
TODO: look into:
 - https://github.com/Theano/
 - http://docs.scipy.org/doc/scipy/reference/generated/scipy.weave.inline.html
 - cython.inline
"""

def py_propensity_function(x, parm, prop):
  args = x+parm.keys()
  return (lambdify(args, prop, 'numpy'), args)

def cython_propensity_function(x, parm, prop):
  args = x+parm.keys()
  return (theano_function(args, prop), args)
  #return (cython.inline("return "+lambdastr(args, prop, dummify=True)), args)

"""
 s: symbolic states of the model
 T: vector of sparse vectors of transitions of between states
 p: vector of transitiion probabilities
 ics: intitial conditions
 t: simulation duration
"""
def integrate(x, T, prop, parm, ics, tmax, method="direct", propensity_function=py_propensity_function):

  trans_n = len(T)

  if trans_n != len(prop):
    raise Error("Transition and propensity vectors must be of same size")

  #time!
  t = 0 #np.asarray([0])

  #convert sparse state space to dense state space
  xv = dict([(xi, np.array(ics.get(xi, 0))) for xi in x])

  #yield zeroth realisation
  #TODO: replace with sparse vector?
  yield [t] + [xv[xi] for xi in x]

  model_map = xv.copy()
  model_map.update(parm)

  prop_fn, prop_args = propensity_function(x, parm, prop)

  while t < tmax:

    #evaluate propensity vector
    #propv = np.asarray(map(lambda p: p.evalf(subs=model_map), prop))

    args = [model_map[p] for p in prop_args]
    prop_v = np.asarray(prop_fn(*args))

    #calculate total propensity
    prop_t = np.sum(prop_v)

    #no more propensity, system is at rest
    if prop_t == 0:
      #TODO: should return a single last point of zeros(len(x)) at time tmax 
      raise Exception("No more propensity? "+ str(prop_v))

    #TODO: allow for negative values? e.g. abs(propv)/np.sum(abs(propv))
    #determine transition probability
    prob = prop_v/prop_t

    #determine future event time
    tau = (1/prop_t)*np.log(1/random())
    t+=tau

    rxn = np.random.choice(trans_n, 1, p=list(prob))[0]
    
    for state, delta in T[rxn].items():
      xv[state] = xv[state] + delta
      model_map[state] = model_map[state] + delta

    yield [t]+[xv[xi] for xi in x]

  raise StopIteration()


