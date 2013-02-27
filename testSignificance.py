#!/usr/bin/env pypy

import sys
import glob
import gc

from inference import *
from bootstrap import bootstrap


def perpGain(r1, r2):
    return (r1 - r2) / ((r1 - 1) if r1 != 1 else 0.000001)


def llGain(r1, r2):
    return (math.exp(r2 - r1) - 1)


def avg(l):
    s = 0
    n = 0
    for x in l:
        s += x
        n += 1
    return float(s) / n if n else 0


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print >>sys.stderr, 'Usage: {0:s} directory_with_files'.format(sys.argv[0])
        sys.exit(1)
    modelName = 'SDBN'
    TESTED_MODEL_PAIRS = [modelName]
    perplexityGains = dict((m, defaultdict(lambda: [])) for m in TESTED_MODEL_PAIRS)
    perplexityGainsPos = [dict((m, defaultdict(lambda: [])) for m in TESTED_MODEL_PAIRS) for pos in xrange(MAX_NUM)]
    llGains = dict((m, defaultdict(lambda: [])) for m in TESTED_MODEL_PAIRS)
    interestingFiles = sorted(glob.glob(sys.argv[1] + '/*'))
    N = len(interestingFiles) - 1
    for fileNumber in xrange(N):
        trainFile = interestingFiles[fileNumber]
        testFile = interestingFiles[fileNumber + 1]
        readInput = InputReader()
        res = []
        m = SimplifiedDbnModel()
        trainFileObj = open(trainFile)
        trainSessions = readInput(trainFileObj)
        trainFileObj.close()
        print 'train read'
        testFileObj = open(testFile)
        # Only needed to compute MAX_QUERY_ID
        readInput(testFileObj)
        testFileObj.close()
        print 'test read'
        m.train(trainSessions)
        del trainSessions
        gc.collect()
        print 'model trained'
        testFileObj = open(testFile)
        testSessions = readInput(testFileObj)
        testFileObj.close()
        currentResult = m.test(testSessions)
        del testSessions
        gc.collect()
        res.append(currentResult)
        print >>sys.stderr, float(fileNumber) / N, modelName, currentResult
        del m
        gc.collect()
        m = SimplifiedDbnModel()
        trainFileObj = open(trainFile)
        trainSessions = readInput(trainFileObj)
        trainSessions = [InputReader.mergeExtraToSessionItem(s) for s in trainSessions]
        trainFileObj.close()
        m.train(trainSessions)
        del trainSessions
        gc.collect()
        testFileObj = open(testFile)
        testSessions = readInput(testFileObj)
        testSessions = [InputReader.mergeExtraToSessionItem(s) for s in testSessions]
        testFileObj.close()
        currentResult = m.test(testSessions)
        del testSessions
        del m
        gc.collect()
        res.append(currentResult)
        print >>sys.stderr, float(fileNumber) / N, modelName, currentResult
        i = 0
        j = 1
        perplexityGains[modelName][(i, j)].append(perpGain(res[i][1], res[j][1]))
        llGains[modelName][(i, j)].append(llGain(res[i][0], res[j][0]))
        for pos in xrange(MAX_NUM):
            perplexityGainsPos[pos][modelName][(i, j)].append(perpGain(res[i][2][pos], res[j][2][pos]))

    for t in ['perplexity']:
        print t.upper()
        for m in [modelName]:
            gainsDict = locals()[t + 'Gains'][m]
            for k, gains in gainsDict.iteritems():
                print m, k, gains, bootstrap(gains)[1]
                if t == 'perplexity':
                    print m, 'POSITION PERPLEXITY GAINS:', k, [[f(perplexityGainsPos[pos][m][k]) for f in [avg, lambda l: bootstrap(l)[1]]] for pos in xrange(MAX_NUM)]

