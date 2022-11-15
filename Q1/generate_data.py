import numpy as np
from numpy.random import multivariate_normal, binomial, shuffle

# the mean and cov matrices
m01 = [5, 0]
m02 = [0, 4]
c01 = [[4, 0], [0, 2]]
c02 = [[1, 0], [0, 3]]
m1 = [3, 2]
c1 = [[2, 0], [0, 2]]


def sample_X_L0(s):
    # the conditional pdf is weighted sum of two gaussian pdfs
    # so it's the same as drawing a sample from each according to a 
    # bernoulli r.v. and we're doing this s times so we can just do a binomial
    n1 = binomial(s, 0.5)
    n1_samps = multivariate_normal(m01, c01, size=n1)
    n2_samps = multivariate_normal(m02, c02, size=s - n1)
    labels = np.zeros((s, 1))
    res = np.concatenate((np.concatenate((n1_samps, n2_samps)).reshape(-1, 2), labels), axis=1)
    shuffle(res)
    return res

def sample_X_L1(s):
    labels = np.ones((s, 1))
    return np.concatenate((multivariate_normal(m1, c1, size= s), labels), axis=1)

def sample_X(s):
    # pdf of X is sum of two conditionals one with 0.6 probability and other with 0.4
    # and we want s samples so we can do a binomial of s with 0.6 probability and generate 
    # one and zero samples according to number of successes and failures (in this case label zero is a success with p = 0.6)
    n0 = binomial(s, 0.6)
    l0_samps = sample_X_L0(n0)
    l1_samps = sample_X_L1(s - n0)
    res = np.concatenate((l0_samps, l1_samps))
    shuffle(res)
    return res

# Now we generate the datasets
np.save("train_100.npy", sample_X(100))
np.save("train_1k.npy", sample_X(1000))
np.save("train_10k.npy", sample_X(10000))
np.save("validation.npy", sample_X(20000))

