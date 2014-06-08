#!/usr/bin/env python
"""
    File    bootstrap.py

    Original Author  Ernesto P. Adorio, PhD.
    UPDEPP(U.P. at Clarkfield)

    http://adorio-research.org/wordpress/?p=12295
"""

import numpy as np
import random


def bootstrap(sample, samplesize=None, nsamples=1000, statfunc=np.mean, conf=0.95):
    """
        Arguments:
            sample - input sample of values
            samplesize - sample size of each generated sample
            nsamples - number of samples to generate
            statfunc- statistical function to apply to each generated sample.

        Returns: bias (deviation of bootstrap from the sample) and confidence interal.
    """
    if samplesize is None:
        samplesize = len(sample)
    n = len(sample)
    X = []
    for i in range(nsamples):
        resample = [random.choice(sample) for i in range(n)]
        x = statfunc(resample)
        X.append(x)
    bias = np.mean(X) - statfunc(sample)

    plower  = (1 - conf) / 2.0
    pupper  = 1 - plower
    symconf = (np.percentile(X, plower * 100), np.percentile(X, pupper * 100))
    return bias, symconf


def Test():
    random.seed(42)
    x = [-0.0802915429072737, -0.053250352200284734, -0.01736675618732031, 0.023235301004853604, 0.11978171918496372, 0.05658880725170068, -0.009028639702335362, 0.044647677002185304, 0.11031884561909511, 0.0681469067959517, 0.005213181783769727, 0.05017491558203524, 0.01865825757134676, -0.040601203753332094, -0.016567084581231573]
    print bootstrap(x)

if __name__ == "__main__":
    Test()

