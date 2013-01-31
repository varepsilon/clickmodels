#!/usr/bin/env pypy
#coding: utf-8

# Input format: hash \t query \t region \t intent_probability \t urls list (json) \t layout (json) \t clicks (json)

import sys
import gc
import simplejson as json
#import json as json
import math

from collections import defaultdict, namedtuple
from datetime import datetime

################################################################################

MAX_ITERATIONS = 40
DEBUG = True
PRETTY_LOG = True

TRAIN_FOR_METRIC = False

USED_MODELS = ['Baseline', 'SDBN', 'UBM', 'UBM-IA', 'EB_UBM', 'EB_UBM-IA', 'DCM', 'DCM-IA', 'DBN', 'DBN-IA']

################################################################################

REL_PRIORS = (0.5, 0.5)

MAX_NUM = 10        # TODO: still hardoded in many places

DEFAULT_REL = REL_PRIORS[1] / sum(REL_PRIORS)

MAX_QUERY_ID = 1000     # some initial value that is changed by InputReader

SessionItem = namedtuple('SessionItem', ['intentWeight', 'query', 'urls', 'layout', 'clicks'])

class ClickModel:

    def __init__(self, ignoreIntents=True, ignoreLayout=True):
        self.ignoreIntents = ignoreIntents
        self.ignoreLayout = ignoreLayout

    def train(self, sessions):
        """
            Set some attributes that will be further used in _getClickProbs function
        """
        pass

    def test(self, sessions, reportPositionPerplexity=False):
        logLikelihood = 0.0
        positionPerplexity = [0.0] * 10
        possibleIntents = [False] if self.ignoreIntents else [False, True]
        for s in sessions:
            iw = s.intentWeight
            intentWeight = {False: 1.0} if self.ignoreIntents else {False: 1 - iw, True: iw}
            clickProbs = self._getClickProbs(s, possibleIntents)
            if DEBUG:
                assert MAX_NUM > 1
                x = sum(clickProbs[i][MAX_NUM // 2] * intentWeight[i] for i in possibleIntents) / sum(clickProbs[i][MAX_NUM // 2 - 1] * intentWeight[i] for i in possibleIntents)
                s.clicks[MAX_NUM // 2] = 1 if s.clicks[MAX_NUM // 2] == 0 else 0
                clickProbs2 = self._getClickProbs(s, possibleIntents)
                y = sum(clickProbs2[i][MAX_NUM // 2] * intentWeight[i] for i in possibleIntents) / sum(clickProbs2[i][MAX_NUM // 2 - 1] * intentWeight[i] for i in possibleIntents)
                assert abs(x + y - 1) < 0.01, (x, y)
            logLikelihood += math.log(sum(clickProbs[i][9] * intentWeight[i] for i in possibleIntents))      # log_e
            for k in xrange(10):
                # P(C_k | C_1, ..., C_{k-1}) = \sum_I P(C_1, ..., C_k | I) P(I) / \sum_I P(C_1, ..., C_{k-1} | I) P(I)
                curClick = dict((i, clickProbs[i][k]) for i in possibleIntents)
                prevClick = dict((i, clickProbs[i][k - 1]) for i in possibleIntents) if k > 0 else dict((i, 1.0) for i in possibleIntents)
                positionPerplexity[k] += math.log(sum(curClick[i] * intentWeight[i] for i in possibleIntents), 2) - math.log(sum(prevClick[i] * intentWeight[i] for i in possibleIntents), 2)
        N = len(sessions)
        positionPerplexity = [2 ** (-x / N) for x in positionPerplexity]
        perplexity = sum(positionPerplexity) / len(positionPerplexity)
        if reportPositionPerplexity:
            return logLikelihood / N / 10, perplexity, positionPerplexity
        else:
            return logLikelihood / N / 10, perplexity

    def _getClickProbs(self, s, possibleIntents):
        """
            Returns clickProbs list
            clickProbs[i][k] = P(C_1, ..., C_k | I=i)
        """
        return dict((i, [0.5 ** (k + 1) for k in xrange(10)]) for i in possibleIntents)


class DbnModel(ClickModel):

    def __init__(self, gammas, ignoreIntents=True, ignoreLayout=True):
        self.gammas = gammas
        ClickModel.__init__(self, ignoreIntents, ignoreLayout)

    def train(self, sessions):
        possibleIntents = [False] if self.ignoreIntents else [False, True]
        # intent -> query -> url -> (a_u, s_u)
        self.urlRelevances = dict((i, [defaultdict(lambda: {'a': DEFAULT_REL, 's': DEFAULT_REL}) for q in xrange(MAX_QUERY_ID)]) for i in possibleIntents)
        # here we store distribution of posterior intent weights given train data
        self.queryIntentsWeights = defaultdict(lambda: [])

        # EM algorithm
        if not PRETTY_LOG:
            print >>sys.stderr, '-' * 80
            print >>sys.stderr, 'Start. Current time is', datetime.now()
        for iteration_count in xrange(MAX_ITERATIONS):
            # urlRelFractions[intent][query][url][r][1] --- coefficient before \log r
            # urlRelFractions[intent][query][url][r][0] --- coefficient before \log (1 - r)
            urlRelFractions = dict((i, [defaultdict(lambda: {'a': [1.0, 1.0], 's': [1.0, 1.0]}) for q in xrange(MAX_QUERY_ID)]) for i in [False, True])
            self.queryIntentsWeights = defaultdict(lambda: [])
            # E step
            for s in sessions:
                positionRelevances = {}
                query = s.query
                for intent in possibleIntents:
                    positionRelevances[intent] = {}
                    for r in ['a', 's']:
                        positionRelevances[intent][r] = [self.urlRelevances[intent][query][url][r] for url in s.urls]
                layout = [False] * 11 if self.ignoreLayout else s.layout
                sessionEstimate = dict((intent, self._getSessionEstimate(positionRelevances[intent], layout, s.clicks, intent)) for intent in possibleIntents)

                # P(I | C, G)
                if self.ignoreIntents:
                    p_I__C_G = {False: 1, True: 0}
                else:
                    a = sessionEstimate[False]['C'] * (1 - s.intentWeight)
                    b = sessionEstimate[True]['C'] * s.intentWeight
                    p_I__C_G = {False: a / (a + b), True: b / (a + b)}
                self.queryIntentsWeights[query].append(p_I__C_G[True])
                for k in xrange(10):
                    url = s.urls[k]
                    for intent in possibleIntents:
                        # update a
                        urlRelFractions[intent][query][url]['a'][1] += sessionEstimate[intent]['a'][k] * p_I__C_G[intent]
                        urlRelFractions[intent][query][url]['a'][0] += (1 - sessionEstimate[intent]['a'][k]) * p_I__C_G[intent]
                        if s.clicks[k] != 0:
                            # Update s
                            urlRelFractions[intent][query][url]['s'][1] += sessionEstimate[intent]['s'][k] * p_I__C_G[intent]
                            urlRelFractions[intent][query][url]['s'][0] += (1 - sessionEstimate[intent]['s'][k]) * p_I__C_G[intent]
            if not PRETTY_LOG:
                sys.stderr.write('E')

            # M step
            # update parameters and record mean square error
            sum_square_displacement = 0.0
            Q_functional = 0.0
            num_points = 0
            for i in possibleIntents:
                for query, d in enumerate(urlRelFractions[i]):
                    if not d:
                        continue
                    for url, relFractions in d.iteritems():
                        a_u_new = relFractions['a'][1] / (relFractions['a'][1] + relFractions['a'][0])
                        sum_square_displacement += (a_u_new - self.urlRelevances[i][query][url]['a']) ** 2
                        num_points += 1
                        self.urlRelevances[i][query][url]['a'] = a_u_new
                        Q_functional += relFractions['a'][1] * math.log(a_u_new) + relFractions['a'][0] * math.log(1 - a_u_new)
                        s_u_new = relFractions['s'][1] / (relFractions['s'][1] + relFractions['s'][0])
                        sum_square_displacement += (s_u_new - self.urlRelevances[i][query][url]['s']) ** 2
                        num_points += 1
                        self.urlRelevances[i][query][url]['s'] = s_u_new
                        Q_functional += relFractions['s'][1] * math.log(s_u_new) + relFractions['s'][0] * math.log(1 - s_u_new)
            if not PRETTY_LOG:
                sys.stderr.write('M\n')
            rmsd = math.sqrt(sum_square_displacement / (num_points if TRAIN_FOR_METRIC else 1.0))
            if PRETTY_LOG:
                sys.stderr.write('%d..' % (iteration_count + 1))
            else:
                print >>sys.stderr, 'Iteration: %d, RMSD: %.10f' % (iteration_count + 1, rmsd)
                print >>sys.stderr, 'Q functional: %f' % Q_functional
        if PRETTY_LOG:
            sys.stderr.write('\n')
        for q, intentWeights in self.queryIntentsWeights.iteritems():
            self.queryIntentsWeights[q] = sum(intentWeights) / len(intentWeights)

    @staticmethod
    def testBackwardForward():
        positionRelevances = {'a': [0.5] * 10, 's': [0.5] * 10}
        gammas = [0.9] * 4
        layout = [False] * 11
        clicks = [0] * 10
        alpha, beta = DbnModel.getForwardBackwardEstimates(positionRelevances, gammas, layout, clicks, False)
        x = alpha[0][0] * beta[0][0] + alpha[0][1] * beta[0][1]
        assert all(abs((a[0] * b[0] + a[1] * b[1]) / x  - 1) < 0.00001 for a, b in zip(alpha, beta))

    @staticmethod
    def getGamma(gammas, k, layout, intent):
        index = 2 * (1 if layout[k + 1] else 0) + (1 if intent else 0)
        return gammas[index]

    @staticmethod
    def getForwardBackwardEstimates(positionRelevances, gammas, layout, clicks, intent):
        alpha = [[0.0, 0.0] for i in xrange(11)]
        beta = [[0.0, 0.0] for i in xrange(11)]
        alpha[0] = [0.0, 1.0]
        beta[10] = [1.0, 1.0]

        # P(E_{k+1} = e, C_k | E_k = e', G, I)
        updateMatrix = [[[0.0 for e1 in [0, 1]] for e in [0, 1]] for i in xrange(10)]
        for k in xrange(10):
            C_k = clicks[k]
            a_u = positionRelevances['a'][k]
            s_u = positionRelevances['s'][k]
            gamma = DbnModel.getGamma(gammas, k, layout, intent)
            if C_k == 0:
                updateMatrix[k][0][0] = 1
                updateMatrix[k][0][1] = (1 - gamma) * (1 - a_u)
                updateMatrix[k][1][0] = 0
                updateMatrix[k][1][1] = gamma * (1 - a_u)
            else:
                updateMatrix[k][0][0] = 0
                updateMatrix[k][0][1] = (s_u + (1 - gamma) * (1 - s_u)) * a_u
                updateMatrix[k][1][0] = 0
                updateMatrix[k][1][1] = gamma * (1 - s_u) * a_u

        for k in xrange(10):
            for e in [0, 1]:
                alpha[k + 1][e] = sum(alpha[k][e1] * updateMatrix[k][e][e1] for e1 in [0, 1])
                beta[9 - k][e] = sum(beta[10 - k][e1] * updateMatrix[9 - k][e1][e] for e1 in [0, 1])

        return alpha, beta

    def _getSessionEstimate(self, positionRelevances, layout, clicks, intent):
        # Returns {'a': P(A_k | I, C, G), 's': P(S_k | I, C, G), 'C': P(C | I, G), 'clicks': P(C_k | C_1, ..., C_{k-1}, I, G)} as a dict
        # sessionEstimate[True]['a'][k] = P(A_k = 1 | I = 'Fresh', C, G), probability of A_k = 0 can be calculated as 1 - p
        sessionEstimate = {'a': [0.0] * 10, 's': [0.0] * 10, 'e': [[0.0, 0.0] for k in xrange(10)], 'C': 0.0, 'clicks': [0.0] * 10}

        alpha, beta = self.getForwardBackwardEstimates(positionRelevances, self.gammas, layout, clicks, intent)
        try:
            varphi = [((a[0] * b[0]) / (a[0] * b[0] + a[1] * b[1]), (a[1] * b[1]) / (a[0] * b[0] + a[1] * b[1])) for a, b in zip(alpha, beta)]
        except ZeroDivisionError:
            print >>sys.stderr, alpha, beta, [(a[0] * b[0] + a[1] * b[1]) for a, b in zip(alpha, beta)], positionRelevances
            sys.exit(1)
        if DEBUG:
            assert all(ph[0] < 0.01 for ph, c in zip(varphi[:10], clicks[:10]) if c != 0), (alpha, beta, varphi, clicks)
        # calculate P(C | I, G) for k = 0
        sessionEstimate['C'] = alpha[0][0] * beta[0][0] + alpha[0][1] * beta[0][1]      # == 0 + 1 * beta[0][1]
        for k in xrange(10):
            C_k = clicks[k]
            a_u = positionRelevances['a'][k]
            s_u = positionRelevances['s'][k]
            gamma = self.getGamma(self.gammas, k, layout, intent)
            # E_k_multiplier --- P(S_k = 0 | C_k) P(C_k | E_k = 1)
            if C_k == 0:
                sessionEstimate['a'][k] = a_u * varphi[k][0]
                sessionEstimate['s'][k] = 0.0
            else:
                sessionEstimate['a'][k] = 1.0
                sessionEstimate['s'][k] = varphi[k + 1][0] * s_u / (s_u + (1 - gamma) * (1 - s_u))
            # P(C_1, ..., C_k | I)
            sessionEstimate['clicks'][k] = sum(alpha[k + 1])

        return sessionEstimate

    def _getClickProbs(self, s, possibleIntents):
        """
            Returns clickProbs list
            clickProbs[i][k] = P(C_1, ..., C_k | I=i)
        """
        positionRelevances = {}
        for intent in possibleIntents:
            positionRelevances[intent] = {}
            for r in ['a', 's']:
                positionRelevances[intent][r] = [self.urlRelevances[intent][s.query][url][r] for url in s.urls]
        layout = [False] * 11 if self.ignoreLayout else s.layout
        return dict((i, self._getSessionEstimate(positionRelevances[i], layout, s.clicks, i)['clicks']) for i in possibleIntents)


class SimplifiedDbnModel(DbnModel):

    def __init__(self, ignoreIntents=True, ignoreLayout=True):
        assert ignoreIntents
        assert ignoreLayout
        DbnModel.__init__(self, (1.0, 1.0, 1.0, 1.0), ignoreIntents, ignoreLayout)

    def train(self, sessions):
        urlRelFractions = [defaultdict(lambda: {'a': [1.0, 1.0], 's': [1.0, 1.0]}) for q in xrange(MAX_QUERY_ID)]
        for s in sessions:
            query = s.query
            lastClickedPos = 9
            for k, c in enumerate(s.clicks[:10]):
                if c != 0:
                    lastClickedPos = k
            for k, (u, c) in enumerate(zip(s.urls, s.clicks[:(lastClickedPos + 1)])):
                if c != 0:
                    urlRelFractions[query][u]['a'][1] += 1
                    if k == lastClickedPos:
                        urlRelFractions[query][u]['s'][1] += 1
                    else:
                        urlRelFractions[query][u]['s'][0] += 1
                else:
                    urlRelFractions[query][u]['a'][0] += 1
        self.urlRelevances = dict((i, [defaultdict(lambda: {'a': DEFAULT_REL, 's': DEFAULT_REL}) for q in xrange(MAX_QUERY_ID)]) for i in [False])
        for query, d in enumerate(urlRelFractions):
            if not d:
                continue
            for url, relFractions in d.iteritems():
                self.urlRelevances[False][query][url]['a'] = relFractions['a'][1] / (relFractions['a'][1] + relFractions['a'][0])
                self.urlRelevances[False][query][url]['s'] = relFractions['s'][1] / (relFractions['s'][1] + relFractions['s'][0])


class UbmModel(ClickModel):

    gammaTypesNum = 4

    def __init__(self, ignoreIntents=True, ignoreLayout=True, explorationBias=False):
        self.explorationBias = explorationBias
        ClickModel.__init__(self, ignoreIntents, ignoreLayout)

    def train(self, sessions):
        possibleIntents = [False] if self.ignoreIntents else [False, True]
        # alpha: intent -> query -> url -> "attractiveness probability"
        self.alpha = dict((i, [defaultdict(lambda: DEFAULT_REL) for q in xrange(MAX_QUERY_ID)]) for i in possibleIntents)
        # gamma: freshness of the current result: gammaType -> rank -> "distance from the last click" - 1 -> examination probability
        self.gamma = [[[0.5 for d in xrange(10)] for r in xrange(10)] for g in xrange(self.gammaTypesNum)]
        if self.explorationBias:
            self.e = [0.5 for p in xrange(10)]
        if not PRETTY_LOG:
            print >>sys.stderr, '-' * 80
            print >>sys.stderr, 'Start. Current time is', datetime.now()
        for iteration_count in xrange(MAX_ITERATIONS):
            self.queryIntentsWeights = defaultdict(lambda: [])
            # not like in DBN! xxxFractions[0] is a numerator while xxxFraction[1] is a denominator
            alphaFractions = dict((i, [defaultdict(lambda: [1.0, 2.0]) for q in xrange(MAX_QUERY_ID)]) for i in possibleIntents)
            gammaFractions = [[[[1.0, 2.0] for d in xrange(10)] for r in xrange(10)] for g in xrange(self.gammaTypesNum)]
            if self.explorationBias:
                eFractions = [[1.0, 2.0] for p in xrange(10)]
            # E-step
            for s in sessions:
                query = s.query
                layout = [False] * 11 if self.ignoreLayout else s.layout
                if self.explorationBias:
                    explorationBiasPossible = any(s.layout[k] and s.clicks[k] for k in xrange(10))      # C_v == 1 in "Beyond 10 blue links..." notation
                    firstVertical = -1 if not any(s.layout[:10]) else [k for k in xrange(10) if s.layout[k]][0]
                if self.ignoreIntents:
                    p_I__C_G = {False: 1.0, True: 0}
                else:
                    a = self._getSessionProb(s) * (1 - s.intentWeight)
                    b = 1 * s.intentWeight
                    p_I__C_G = {False: a / (a + b), True: b / (a + b)}
                self.queryIntentsWeights[query].append(p_I__C_G[True])
                prevClick = -1
                for rank, c in enumerate(s.clicks[:10]):
                    url = s.urls[rank]
                    for intent in possibleIntents:
                        a = self.alpha[intent][query][url]
                        if self.explorationBias and explorationBiasPossible:
                            e = self.e[firstVertical]
                        if c == 0:
                            g = self.getGamma(self.gamma, rank, prevClick, layout, intent)
                            gCorrection = 1
                            if self.explorationBias and explorationBiasPossible and not s.layout[k]:
                                gCorrection = 1 - e
                                g *= gCorrection
                            alphaFractions[intent][query][url][0] += a * (1 - g) / (1 - a * g) * p_I__C_G[intent]
                            self.getGamma(gammaFractions, rank, prevClick, layout, intent)[0] += g / gCorrection * (1 - a) / (1 - a * g) * p_I__C_G[intent]
                            if self.explorationBias and explorationBiasPossible:
                                eFractions[firstVertical][0] += (e if s.layout[k] else e / (1 - a * g)) * p_I__C_G[intent]
                        else:
                            alphaFractions[intent][query][url][0] += 1 * p_I__C_G[intent]
                            self.getGamma(gammaFractions, rank, prevClick, layout, intent)[0] += 1 * p_I__C_G[intent]
                            if self.explorationBias and explorationBiasPossible:
                                eFractions[firstVertical][0] += (e if s.layout[k] else 0) * p_I__C_G[intent]
                        alphaFractions[intent][query][url][1] += 1 * p_I__C_G[intent]
                        self.getGamma(gammaFractions, rank, prevClick, layout, intent)[1] += 1 * p_I__C_G[intent]
                        if self.explorationBias and explorationBiasPossible:
                            eFractions[firstVertical][1] += 1 * p_I__C_G[intent]
                    if c != 0:
                        prevClick = rank
            if not PRETTY_LOG:
                sys.stderr.write('E')
            # M-step
            sum_square_displacement = 0.0
            num_points = 0
            for i in possibleIntents:
                for q in xrange(MAX_QUERY_ID):
                    for url, aF in alphaFractions[i][q].iteritems():
                        new_alpha = aF[0] / aF[1]
                        sum_square_displacement += (self.alpha[i][q][url] - new_alpha) ** 2
                        num_points += 1
                        self.alpha[i][q][url] = new_alpha
            for g in xrange(self.gammaTypesNum):
                for r in xrange(10):
                    for d in xrange(10):
                        gF = gammaFractions[g][r][d]
                        new_gamma = gF[0] / gF[1]
                        sum_square_displacement += (self.gamma[g][r][d] - new_gamma) ** 2
                        num_points += 1
                        self.gamma[g][r][d] = new_gamma
            if self.explorationBias:
                for p in xrange(10):
                    new_e = eFractions[p][0] / eFractions[p][1]
                    sum_square_displacement += (self.e[p] - new_e) ** 2
                    num_points += 1
                    self.e[p] = new_e
            if not PRETTY_LOG:
                sys.stderr.write('M\n')
            rmsd = math.sqrt(sum_square_displacement / (num_points if TRAIN_FOR_METRIC else 1.0))
            if PRETTY_LOG:
                sys.stderr.write('%d..' % (iteration_count + 1))
            else:
                print >>sys.stderr, 'Iteration: %d, RMSD: %.10f' % (iteration_count + 1, rmsd)
        if PRETTY_LOG:
            sys.stderr.write('\n')
        for q, intentWeights in self.queryIntentsWeights.iteritems():
            self.queryIntentsWeights[q] = sum(intentWeights) / len(intentWeights)

    def _getSessionProb(self, s):
        clickProbs = self._getClickProbs(s, [False, True])
        return clickProbs[False][9] / clickProbs[True][9]

    @staticmethod
    def getGamma(gammas, k, prevClick, layout, intent):
        index = (2 if layout[k] else 0) + (1 if intent else 0)
        return gammas[index][k][k - prevClick - 1]

    def _getClickProbs(self, s, possibleIntents):
        """
            Returns clickProbs list
            clickProbs[i][k] = P(C_1, ..., C_k | I=i)
        """
        clickProbs = dict((i, []) for i in possibleIntents)
        firstVertical = -1 if not any(s.layout[:10]) else [k for k in xrange(10) if s.layout[k]][0]
        prevClick = -1
        layout = [False] * 11 if self.ignoreLayout else s.layout
        for rank, c in enumerate(s.clicks[:10]):
            url = s.urls[rank]
            prob = {False: 0.0, True: 0.0}
            for i in possibleIntents:
                a = self.alpha[i][s.query][url]
                g = self.getGamma(self.gamma, rank, prevClick, layout, i)
                if self.explorationBias and any(s.layout[k] and s.clicks[k] for k in xrange(rank)) and not s.layout[rank]:
                    g *= 1 - self.e[firstVertical]
                prevProb = 1 if rank == 0 else clickProbs[i][-1]
                if c == 0:
                    clickProbs[i].append(prevProb * (1 - a * g))
                else:
                    clickProbs[i].append(prevProb * a * g)
            if c != 0:
                prevClick = rank
        return clickProbs


class EbUbmModel(UbmModel):
    def __init__(self, ignoreIntents=True, ignoreLayout=True):
        UbmModel.__init__(self, ignoreIntents, ignoreLayout, explorationBias=True)


class DcmModel(ClickModel):

    gammaTypesNum = 4

    def train(self, sessions):
        possibleIntents = [False] if self.ignoreIntents else [False, True]
        urlRelFractions = dict((i, [defaultdict(lambda: [1.0, 1.0]) for q in xrange(MAX_QUERY_ID)]) for i in possibleIntents)
        gammaFractions = [[[1.0, 1.0] for g in xrange(self.gammaTypesNum)] for r in xrange(10)]
        for s in sessions:
            query = s.query
            layout = [False] * 11 if self.ignoreLayout else s.layout
            lastClickedPos = 9
            for k, c in enumerate(s.clicks[:10]):
                if c != 0:
                    lastClickedPos = k
            intentWeights = {False: 1.0} if self.ignoreIntents else {False: 1 - s.intentWeight, True: s.intentWeight}
            for k, (u, c) in enumerate(zip(s.urls, s.clicks[:(lastClickedPos + 1)])):
                for i in possibleIntents:
                    if c != 0:
                        urlRelFractions[i][query][u][1] += intentWeights[i]
                        if k == lastClickedPos:
                            self.getGamma(gammaFractions[k], k, layout, i)[1] += intentWeights[i]
                        else:
                            self.getGamma(gammaFractions[k], k, layout, i)[0] += intentWeights[i]
                    else:
                        urlRelFractions[i][query][u][0] += intentWeights[i]
        self.urlRelevances = dict((i, [defaultdict(lambda: DEFAULT_REL) for q in xrange(MAX_QUERY_ID)]) for i in possibleIntents)
        self.gammas = [[0.5 for g in xrange(self.gammaTypesNum)] for r in xrange(10)]
        for i in possibleIntents:
            for query, d in enumerate(urlRelFractions[i]):
                if not d:
                    continue
                for url, relFractions in d.iteritems():
                    self.urlRelevances[i][query][url] = relFractions[1] / (relFractions[1] + relFractions[0])
        for k in xrange(10):
            for g in xrange(self.gammaTypesNum):
                self.gammas[k][g] = gammaFractions[k][g][0] / (gammaFractions[k][g][0] + gammaFractions[k][g][1])

    def _getClickProbs(self, s, possibleIntents):
        clickProbs = {False: [], True: []}          # P(C_1, ..., C_k)
        query = s.query
        layout = [False] * 11 if self.ignoreLayout else s.layout
        for i in possibleIntents:
            examinationProb = 1.0       # P(C_1, ..., C_{k - 1}, E_k = 1)
            for k, c in enumerate(s.clicks[:10]):
                r = self.urlRelevances[i][query][s.urls[k]]
                prevProb = 1 if k == 0 else clickProbs[i][-1]
                if c == 0:
                    clickProbs[i].append(prevProb - examinationProb * r)    # P(C_1, ..., C_k = 0) = P(C_1, ..., C_{k-1}) - P(C_1, ..., C_k = 1)
                    examinationProb *= 1 - r                                # P(C_1, ..., C_k, E_{k+1} = 1) = P(E_{k+1} = 1 | C_k, E_k = 1) * P(C_k | E_k = 1) *  P(C_1, ..., C_{k - 1}, E_k = 1)
                else:
                    clickProbs[i].append(examinationProb * r)
                    examinationProb *= self.getGamma(self.gammas[k], k, layout, i) * r  # P(C_1, ..., C_k, E_{k+1} = 1) = P(E_{k+1} = 1 | C_k, E_k = 1) * P(C_k | E_k = 1) *  P(C_1, ..., C_{k - 1}, E_k = 1)

        return clickProbs

    @staticmethod
    def getGamma(gammas, k, layout, intent):
        return DbnModel.getGamma(gammas, k, layout, intent)


class InputReader:
    def __init__(self, discardNoClicks=True):
        self.url_to_id = {}
        self.query_to_id = {}
        self.current_url_id = 1
        self.current_query_id = 0
        self.discardNoClicks = discardNoClicks

    def __call__(self, f):
        sessions = []
        for line in f:
            hash_digest, query, region, intentWeight, urls, layout, clicks = line.rstrip().split('\t')
            urls, layout, clicks = map(json.loads, [urls, layout, clicks])
            if len(urls) != 10:
                continue
            if self.discardNoClicks and all(c == 0 for c in clicks[:10]):
                continue
            if float(intentWeight) > 1 or float(intentWeight) < 0:
                continue
            if (query, region) in self.query_to_id:
                query_id = self.query_to_id[(query, region)]
            else:
                query_id = self.current_query_id
                self.query_to_id[(query, region)] = self.current_query_id
                self.current_query_id += 1
            intentWeight = float(intentWeight)
            # add fake G_11 to simplify gamma calculation:
            layout.append(not layout[-1])
            url_ids = []
            for u in urls:
                if u in ['_404', 'STUPID', 'VIRUS', 'SPAM']:
                    assert TRAIN_FOR_METRIC
                    u = 'IRRELEVANT'
                if u.startswith('RELEVANT_'):
                    assert TRAIN_FOR_METRIC
                    u = 'RELEVANT'
                if u in self.url_to_id:
                    if TRAIN_FOR_METRIC:
                        url_ids.append(u)
                    else:
                        url_ids.append(self.url_to_id[u])
                else:
                    urlid = self.current_url_id
                    if TRAIN_FOR_METRIC:
                        url_ids.append(u)
                    else:
                        url_ids.append(urlid)
                    self.url_to_id[u] = urlid
                    self.current_url_id += 1
            sessions.append(SessionItem(intentWeight, query_id, url_ids, layout, clicks))
        # FIXME: bad style
        global MAX_QUERY_ID
        MAX_QUERY_ID = self.current_query_id
        return sessions

if __name__ == '__main__':
    if DEBUG:
        DbnModel.testBackwardForward()
    allCombinations = []
    interestingValues = [0.9, 1.0]
    for g1 in interestingValues:
        for g2 in interestingValues:
            for g3 in interestingValues:
                for g4 in interestingValues:
                    allCombinations.append((g1, g2, g3, g4))

    readInput = InputReader()
    sessions = readInput(sys.stdin)

    if TRAIN_FOR_METRIC:
        # --------------------------------------------------------------------------------
        #                           For EBU
        # --------------------------------------------------------------------------------
        # Relevance -> P(Click | Relevance)
        p_C_R_frac = defaultdict(lambda: [0, 0.0001])
        # Relevance -> P(Leave | Click, Relevance)
        p_L_C_R_frac = defaultdict(lambda: [0, 0.0001])
        for s in sessions:
            lastClickPos = max((i for i, c in enumerate(s.clicks[:10]) if c != 0))
            for i in xrange(lastClickPos + 1):
                u = s.urls[i]
                if s.clicks[i] != 0:
                    p_C_R_frac[u][0] += 1
                    if i == lastClickPos:
                        p_L_C_R_frac[u][0] += 1
                    p_L_C_R_frac[u][1] += 1
                p_C_R_frac[u][1] += 1

        for u in ['IRRELEVANT', 'RELEVANT', 'USEFUL', 'VITAL']:
            print 'P(C|%s)\t%f\tP(L|C,%s)\t%f' % (u, float(p_C_R_frac[u][0]) / p_C_R_frac[u][1], u, float(p_L_C_R_frac[u][0]) / p_L_C_R_frac[u][1])
        # --------------------------------------------------------------------------------

    if 'DBN' in USED_MODELS:
        print 'Will going to run no more than: %.1f hours (approx)' % ((len(allCombinations) + len(interestingValues)) * MAX_ITERATIONS / 60 * len(sessions) / 1E6)

    #clickProbs = [0] * 10
    #for s in sessions:
        #for i in xrange(10):
            #clickProbs[i] += s.clicks[i]
    #print '\t'.join(str(float(x) / len(sessions)) for x in clickProbs)
    #sys.exit(0)

    if len(sys.argv) > 1:
        with open(sys.argv[1]) as test_clicks_file:
            testSessions = readInput(test_clicks_file)
    else:
        testSessions = sessions
    del readInput       # needed to minimize memory consumption (see gc.collect() below)

    print len(sessions), len(testSessions)

    if 'Baseline' in USED_MODELS:
        baselineModel = ClickModel()
        baselineModel.train(sessions)
        print 'Baseline:', baselineModel.test(testSessions)

    if 'SDBN' in USED_MODELS:
        sdbnModel = SimplifiedDbnModel()
        sdbnModel.train(sessions)
        print 'SDBN:', sdbnModel.test(testSessions)
        del sdbnModel        # needed to minimize memory consumption (see gc.collect() below)

    if 'UBM' in USED_MODELS:
        ubmModel = UbmModel()
        ubmModel.train(sessions)
        if TRAIN_FOR_METRIC:
            print '\n'.join(['%s\t%f' % r for r in \
                    [(x, ubmModel.alpha[False][0][x]) for x in \
                            ['IRRELEVANT', 'RELEVANT', 'USEFUL', 'VITAL']]])
            for d in xrange(10):
                for r in xrange(10):
                    print ('%.4f ' % (ubmModel.gamma[0][r][9-d] if r + d >= 9 else 0)),
                print
        print 'UBM', ubmModel.test(testSessions)
        del ubmModel       # needed to minimize memory consumption (see gc.collect() below)

    if 'UBM-IA' in USED_MODELS:
        ubmModel = UbmModel(ignoreIntents=False, ignoreLayout=False)
        ubmModel.train(sessions)
        print 'UBM-IA', ubmModel.test(testSessions)
        del ubmModel       # needed to minimize memory consumption (see gc.collect() below)

    if 'EB_UBM' in USED_MODELS:
        ebUbmModel = EbUbmModel()
        ebUbmModel.train(sessions)
        # print 'Exploration bias:', ebUbmModel.e
        print 'EB_UBM', ebUbmModel.test(testSessions)
        del ebUbmModel       # needed to minimize memory consumption (see gc.collect() below)

    if 'EB_UBM-IA' in USED_MODELS:
        ebUbmModel = EbUbmModel(ignoreIntents=False, ignoreLayout=False)
        ebUbmModel.train(sessions)
        # print 'Exploration bias:', ebUbmModel.e
        print 'EB_UBM-IA', ebUbmModel.test(testSessions)
        del ebUbmModel       # needed to minimize memory consumption (see gc.collect() below)

    if 'DCM' in USED_MODELS:
        dcmModel = DcmModel()
        dcmModel.train(sessions)
        if TRAIN_FOR_METRIC:
            print '\n'.join(['%s\t%f' % r for r in \
                [(x, dcmModel.urlRelevances[False][0][x]) for x in \
                        ['IRRELEVANT', 'RELEVANT', 'USEFUL', 'VITAL']]])
            print 'DCM gammas:', dcmModel.gammas
        print 'DCM', dcmModel.test(testSessions)
        del dcmModel       # needed to minimize memory consumption (see gc.collect() below)

    if 'DCM-IA' in USED_MODELS:
        dcmModel = DcmModel(ignoreIntents=False, ignoreLayout=False)
        dcmModel.train(sessions)
        # print 'DCM gammas:', dcmModel.gammas
        print 'DCM-IA', dcmModel.test(testSessions)
        del dcmModel       # needed to minimize memory consumption (see gc.collect() below)

    if 'DBN' in USED_MODELS:
        dbnModel = DbnModel((0.9, 0.9, 0.9, 0.9))
        dbnModel.train(sessions)
        print 'DBN:', dbnModel.test(testSessions)
        del dbnModel       # needed to minimize memory consumption (see gc.collect() below)

    if 'DBN-IA' in USED_MODELS:
        for gammas in allCombinations:
            gc.collect()
            dbnModel = DbnModel(gammas, ignoreIntents=False, ignoreLayout=False)
            dbnModel.train(sessions)
            print 'DBN-IA: %.2f %.2f %.2f %.2f' % gammas, dbnModel.test(testSessions)

