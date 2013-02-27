#!/usr/bin/env pypy

import sys
import glob

from inference import *
from bootstrap import bootstrap


def perpGain(r1, r2):
    return (r1 - r2) / (r1 - 1)


def llGain(r1, r2):
    return (math.exp(r2 - r1) - 1)


def avg(l):
    s = 0
    n = 0
    for x in l:
        s += x
        n += 1
    return float(s) / n if n else 0


# TESTED_MODEL_PAIRS = ['UBM', 'EB_UBM', 'EB_UBM-IA']
TESTED_MODEL_PAIRS = ['UBM']

if 'RBP' in TESTED_MODEL_PAIRS:
    import scipy
    import scipy.optimize

MODEL_CONSTRUCTORS = {
    'DBN': (lambda: DbnModel((0.9, 0.9, 0.9, 0.9)), lambda: DbnModel((1.0, 0.9, 1.0, 0.9), ignoreIntents=False, ignoreLayout=False)),
    'UBMvsDBN': (UbmModel, lambda: DbnModel((0.9, 0.9, 0.9, 0.9))),
    'UBM': (UbmModel, lambda: UbmModel(ignoreIntents=False, ignoreLayout=False)),
    'EB_UBM': (UbmModel, EbUbmModel, lambda: EbUbmModel(ignoreIntents=False, ignoreLayout=False)),
    'DCM': (DcmModel, lambda: DcmModel(ignoreIntents=False, ignoreLayout=False)),
    'RBP': (SimplifiedRbpModel, lambda: RbpModel(ignoreIntents=False, ignoreLayout=False))
}

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print >>sys.stderr, 'Usage: {0:s} directory_with_files'.format(sys.argv[0])
        sys.exit(1)
    perplexityGains = dict((m, defaultdict(lambda: [])) for m in TESTED_MODEL_PAIRS)
    perplexityGainsPos = [dict((m, defaultdict(lambda: [])) for m in TESTED_MODEL_PAIRS) for pos in xrange(MAX_NUM)]
    llGains = dict((m, defaultdict(lambda: [])) for m in TESTED_MODEL_PAIRS)
    interestingFiles = sorted(glob.glob(sys.argv[1] + '/*'))
    N = len(interestingFiles) // 2
    for fileNumber in xrange(N):
        trainFile = interestingFiles[2 * fileNumber]
        testFile = interestingFiles[2 * fileNumber + 1]
        readInput = InputReader()
        trainSessions = readInput(open(trainFile))
        testSessions = readInput(open(testFile))
        for modelName in TESTED_MODEL_PAIRS:
            res = []
            models = MODEL_CONSTRUCTORS[modelName]
            for idx, model in enumerate(models):
                m = model()
                m.train(trainSessions)
                currentResult = m.test(testSessions, reportPositionPerplexity=True)
                res.append(currentResult)
                print >>sys.stderr, float(fileNumber) / N, modelName, idx, currentResult
                del m
            for i in xrange(len(models)):
                for j in xrange(i + 1, len(models)):
                    perplexityGains[modelName][(i, j)].append(perpGain(res[i][1], res[j][1]))
                    llGains[modelName][(i, j)].append(llGain(res[i][0], res[j][0]))
                    for pos in xrange(MAX_NUM):
                        perplexityGainsPos[pos][modelName][(i, j)].append(perpGain(res[i][2][pos], res[j][2][pos]))

    for t in ['ll', 'perplexity']:
        print t.upper()
        for m in TESTED_MODEL_PAIRS:
            gainsDict = locals()[t + 'Gains'][m]
            for k, gains in gainsDict.iteritems():
                print m, k, gains, bootstrap(gains)[1]
                if t == 'perplexity':
                    print m, 'POSITION PERPLEXITY GAINS:', k, [[f(perplexityGainsPos[pos][m][k]) for f in [avg, lambda l: bootstrap(l)[1]]] for pos in xrange(MAX_NUM)]

