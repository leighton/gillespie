import numpy as np
import theano
import theano.tensor as tt
from theano.tensor.shared_randomstreams import RandomStreams


"""
p: reaction propensities
"""
def direct_method(var, parm, p, S, seed=233):
    S = np.asarray(S)

    reps = var[0].shape[0]

    rng = RandomStreams(seed=seed) #todo: makes sure it runs on GPU
    r_u = rng.uniform((1,reps))

    p_n = sum(p)
    prob = [p_i/p_n for p_i in p]

    v = tt.stack(prob).reshape((len(p),reps)).T #This happens because of variables as dvectors (rxns, reps)

    #TODO: check prb_f for nans
    prb_f = theano.function(var+parm, v)

    tau_f = theano.function(var+parm, (1/p_n)*tt.log(1/r_u))

    ran_f = theano.function(var+parm, rng.multinomial(n=1, pvals=v))

    def compute(ics, parm, interval):
        x = ics
        reps = ics[0].shape[0]
        time = np.zeros(shape=(1,reps))
        out = np.zeros(shape=(reps, interval, len(var)+1)) #(rep, time, vars+time)

        for idx in range(interval):

            args = [x[0],x[1]]+parm

            time += tau_f(*args)

            ran_i = ran_f(*args)
            incr = np.dot(ran_i, S).T

            x = np.asarray(x)+incr
            out[:,idx,:] = np.concatenate((time, x)).T

        return out

    return compute

_fn_map = {
    'direct' : direct_method
}

def function(*argv, **kwargs):

    fn = direct_method
    if 'method' in kwargs:
        fn = _fn_map[kwargs['method']]
    kwargs.pop("method", None)
    return fn(*argv, **kwargs)
