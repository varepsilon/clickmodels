from collections import defaultdict
from datetime import datetime
import gc
import json
import math
import random
import sys

from .config_sample import MAX_ITERATIONS, DEBUG, PRETTY_LOG, MAX_DOCS_PER_QUERY, SERP_SIZE, TRANSFORM_LOG, QUERY_INDEPENDENT_PAGER, DEFAULT_REL


class NotImplementedError(Exception):
    pass


class ClickModel:
    """
        An abstract click model interface.
    """

    def __init__(self, ignoreIntents=True, ignoreLayout=True, config=None):
        self.config = config if config is not None else {}
        self.ignoreIntents = ignoreIntents
        self.ignoreLayout = ignoreLayout

    def train(self, sessions):
        """
            Trains the model.
        """
        pass

    def test(self, sessions, reportPositionPerplexity=True):
        """
            Evaluates the prediciton power of the click model for a given sessions.
            Returns the log-likelihood, perplexity, position perplexity
            (perplexity for each rank a.k.a. position in a SERP)
            and separate perplexity values for clicks and non-clicks (skips).
        """
        logLikelihood = 0.0
        positionPerplexity = [0.0] * self.config.get('MAX_DOCS_PER_QUERY', MAX_DOCS_PER_QUERY)
        positionPerplexityClickSkip = [[0.0, 0.0] \
                for i in xrange(self.config.get('MAX_DOCS_PER_QUERY', MAX_DOCS_PER_QUERY))]
        counts = [0] * self.config.get('MAX_DOCS_PER_QUERY', MAX_DOCS_PER_QUERY)
        countsClickSkip = [[0, 0] \
                for i in xrange(self.config.get('MAX_DOCS_PER_QUERY', MAX_DOCS_PER_QUERY))]
        possibleIntents = [False] if self.ignoreIntents else [False, True]
        for s in sessions:
            iw = s.intentWeight
            intentWeight = {False: 1.0} if self.ignoreIntents else {False: 1 - iw, True: iw}
            clickProbs = self._get_click_probs(s, possibleIntents)
            N = min(len(s.clicks), self.config.get('MAX_DOCS_PER_QUERY', MAX_DOCS_PER_QUERY))
            if self.config.get('DEBUG', DEBUG):
                assert N > 1
                x = sum(clickProbs[i][N // 2] * intentWeight[i] for i in possibleIntents) / sum(clickProbs[i][N // 2 - 1] * intentWeight[i] for i in possibleIntents)
                s.clicks[N // 2] = 1 if s.clicks[N // 2] == 0 else 0
                clickProbs2 = self._get_click_probs(s, possibleIntents)
                y = sum(clickProbs2[i][N // 2] * intentWeight[i] for i in possibleIntents) / sum(clickProbs2[i][N // 2 - 1] * intentWeight[i] for i in possibleIntents)
                assert abs(x + y - 1) < 0.01, (x, y)
            # Marginalize over possible intents: P(C_1, ..., C_N) = \sum_{i} P(C_1, ..., C_N | I=i) P(I=i)
            logLikelihood += math.log(sum(clickProbs[i][N - 1] * intentWeight[i] for i in possibleIntents)) / N
            correctedRank = 0    # we are going to skip clicks on fake pager urls
            for k, click in enumerate(s.clicks):
                click = 1 if click else 0
                if s.extraclicks.get('TRANSFORMED', False) and \
                        (k + 1) % (self.config.get('SERP_SIZE', SERP_SIZE) + 1) == 0:
                    if self.config.get('DEBUG', DEBUG):
                        assert s.results[k] == 'PAGER'
                    continue
                # P(C_k | C_1, ..., C_{k-1}) = \sum_I P(C_1, ..., C_k | I) P(I) / \sum_I P(C_1, ..., C_{k-1} | I) P(I)
                curClick = dict((i, clickProbs[i][k]) for i in possibleIntents)
                prevClick = dict((i, clickProbs[i][k - 1]) for i in possibleIntents) if k > 0 else dict((i, 1.0) for i in possibleIntents)
                logProb = math.log(sum(curClick[i] * intentWeight[i] for i in possibleIntents), 2) - math.log(sum(prevClick[i] * intentWeight[i] for i in possibleIntents), 2)
                positionPerplexity[correctedRank] += logProb
                positionPerplexityClickSkip[correctedRank][click] += logProb
                counts[correctedRank] += 1
                countsClickSkip[correctedRank][click] += 1
                correctedRank += 1
        positionPerplexity = [2 ** (-x / count if count else x) for (x, count) in zip(positionPerplexity, counts)]
        positionPerplexityClickSkip = [[2 ** (-x[click] / (count[click] if count[click] else 1) if count else x) \
                for (x, count) in zip(positionPerplexityClickSkip, countsClickSkip)] for click in xrange(2)]
        perplexity = sum(positionPerplexity) / len(positionPerplexity)
        if reportPositionPerplexity:
            return logLikelihood / len(sessions), perplexity, positionPerplexity, positionPerplexityClickSkip
        else:
            return logLikelihood / len(sessions), perplexity

    def _get_click_probs(self, s, possible_intents):
        """
            Returns click probabilities list for a given list of s.clicks.
            For each intent $i$ and each rank $k$ we have:
            click_probs[i][k-1] = P(C_1, ..., C_k | I=i)
        """
        click_probs = dict((i, [0.5 ** (k + 1) for k in xrange(len(s.clicks))]) for i in possible_intents)
        return click_probs

    def get_loglikelihood(self, sessions):
        """
            Returns the average log-likelihood of the current model for given sessions.
            This is a lightweight version of the self.test() method.
        """
        return sum(self.get_log_click_probs(s) for s in sessions) / len(sessions)

    def get_log_click_probs(self, session):
        """
            Returns an average log-likelihood for a given session,
            i.e. log-likelihood of all the click events, divided
            by the number of documents in the session.
        """
        possibleIntents = [False] if self.ignoreIntents else [False, True]
        intentWeight = {False: 1.0} if self.ignoreIntents else \
                {False: 1 - session.intentWeight, True: session.intentWeight}
        clickProbs = self._get_click_probs(s, possibleIntents)
        N = min(len(session.clicks), self.config.get('MAX_DOCS_PER_QUERY', MAX_DOCS_PER_QUERY))
        # Marginalize over possible intents: P(C_1, ..., C_N) = \sum_{i} P(C_1, ..., C_N | I=i) P(I=i)
        return math.log(sum(clickProbs[i][N - 1] * intentWeight[i] for i in possibleIntents)) / N

    def get_model_relevances(self, session, intent=False):
        """
            Returns estimated relevance of each document in a given session
            based on a trained click model.
        """
        raise NotImplementedError

    def predict_click_probs(self, session, intent=False):
        """
            Predicts click probabilities for a given session. Does not use session.clicks.
            This is a vector of P(C_k = 1 | E_k = 1) for different ranks $k$.
        """
        raise NotImplementedError

    def predict_stop_probs(self, session, intent=False):
        """
            Predicts stop probabilities (after click) for each document in a session.
            This is often referred to as satisfaction probability.
            This is a vector of P(S_k = 1 | C_k = 1) for different ranks $k$.
        """
        raise NotImplementedError

    def get_abandonment_prob(self, rank, intent=False, layout=None):
        """
            Predicts probability of stopping without click after examining document at rank `rank`.
        """
        return 0.0

    def generate_clicks(self, session):
        """
            Generates clicks for a given session, assuming cascade examination order.
        """
        clicks = [0] * len(session.results)
        # First, randomly select user intent.
        intent = False  # non-vertical intent by default
        if not self.ignoreIntents:
            random_intent_prob = random.uniforma(0, 1)
            if random_intent_prob < session.intentWeight:
                intent = True
        predicted_click_probs = self.predict_click_probs(session, intent)
        predicted_stop_probs = self.predict_stop_probs(session, intent)
        for rank, result in enumerate(session.results):
            random_click_prob = random.uniform(0, 1)
            clicks[rank] = 1 if random_click_prob < predicted_click_probs[rank] else 0
            if clicks[rank] == 1:
                random_stop_prob = random.uniform(0, 1)
                if random_stop_prob < predicted_stop_probs[rank]:
                    break
            else:
                random_stop_prob = random.uniform(0, 1)
                if random_stop_prob < self.get_abandonment_prob(rank, intent, session.layout):
                    break
        return clicks


class DbnModel(ClickModel):
    def __init__(self, gammas, ignoreIntents=True, ignoreLayout=True, config=None):
        self.gammas = gammas
        ClickModel.__init__(self, ignoreIntents, ignoreLayout, config)

    def train(self, sessions):
        possibleIntents = [False] if self.ignoreIntents else [False, True]
        max_query_id = self.config.get('MAX_QUERY_ID')
        if max_query_id is None:
            print >>sys.stderr, 'WARNING: no MAX_QUERY_ID specified for', self
            max_query_id = 100000
        # intent -> query -> url -> (a_u, s_u)
        self.urlRelevances = dict((i,
                [defaultdict(lambda: {'a': self.config.get('DEFAULT_REL', DEFAULT_REL),
                                      's': self.config.get('DEFAULT_REL', DEFAULT_REL)}) \
                    for q in xrange(max_query_id)]) for i in possibleIntents
        )
        # here we store distribution of posterior intent weights given train data
        self.queryIntentsWeights = defaultdict(lambda: [])
        # EM algorithm
        if not self.config.get('PRETTY_LOG', PRETTY_LOG):
            print >>sys.stderr, '-' * 80
            print >>sys.stderr, 'Start. Current time is', datetime.now()
        for iteration_count in xrange(self.config.get('MAX_ITERATIONS', MAX_ITERATIONS)):
            # urlRelFractions[intent][query][url][r][1] --- coefficient before \log r
            # urlRelFractions[intent][query][url][r][0] --- coefficient before \log (1 - r)
            urlRelFractions = dict((i, [defaultdict(lambda: {'a': [1.0, 1.0], 's': [1.0, 1.0]}) for q in xrange(max_query_id)]) for i in [False, True])
            self.queryIntentsWeights = defaultdict(lambda: [])
            # E step
            for s in sessions:
                positionRelevances = {}
                query = s.query
                for intent in possibleIntents:
                    positionRelevances[intent] = {}
                    for r in ['a', 's']:
                        positionRelevances[intent][r] = [self.urlRelevances[intent][query][url][r] for url in s.results]
                layout = [False] * len(s.layout) if self.ignoreLayout else s.layout
                sessionEstimate = dict((intent, self._getSessionEstimate(positionRelevances[intent], layout, s.clicks, intent)) for intent in possibleIntents)
                # P(I | C, G)
                if self.ignoreIntents:
                    p_I__C_G = {False: 1, True: 0}
                else:
                    a = sessionEstimate[False]['C'] * (1 - s.intentWeight)
                    b = sessionEstimate[True]['C'] * s.intentWeight
                    p_I__C_G = {False: a / (a + b), True: b / (a + b)}
                self.queryIntentsWeights[query].append(p_I__C_G[True])
                for k, url in enumerate(s.results):
                    for intent in possibleIntents:
                        # update a
                        urlRelFractions[intent][query][url]['a'][1] += sessionEstimate[intent]['a'][k] * p_I__C_G[intent]
                        urlRelFractions[intent][query][url]['a'][0] += (1 - sessionEstimate[intent]['a'][k]) * p_I__C_G[intent]
                        if s.clicks[k] != 0:
                            # Update s
                            urlRelFractions[intent][query][url]['s'][1] += sessionEstimate[intent]['s'][k] * p_I__C_G[intent]
                            urlRelFractions[intent][query][url]['s'][0] += (1 - sessionEstimate[intent]['s'][k]) * p_I__C_G[intent]
            if not self.config.get('PRETTY_LOG', PRETTY_LOG):
                sys.stderr.write('E')
            # M step
            # update parameters and record mean square error
            sum_square_displacement = 0.0
            Q_functional = 0.0
            for i in possibleIntents:
                for query, d in enumerate(urlRelFractions[i]):
                    if not d:
                        continue
                    for url, relFractions in d.iteritems():
                        a_u_new = relFractions['a'][1] / (relFractions['a'][1] + relFractions['a'][0])
                        sum_square_displacement += (a_u_new - self.urlRelevances[i][query][url]['a']) ** 2
                        self.urlRelevances[i][query][url]['a'] = a_u_new
                        Q_functional += relFractions['a'][1] * math.log(a_u_new) + relFractions['a'][0] * math.log(1 - a_u_new)
                        s_u_new = relFractions['s'][1] / (relFractions['s'][1] + relFractions['s'][0])
                        sum_square_displacement += (s_u_new - self.urlRelevances[i][query][url]['s']) ** 2
                        self.urlRelevances[i][query][url]['s'] = s_u_new
                        Q_functional += relFractions['s'][1] * math.log(s_u_new) + relFractions['s'][0] * math.log(1 - s_u_new)
            if not self.config.get('PRETTY_LOG', PRETTY_LOG):
                sys.stderr.write('M\n')
            rmsd = math.sqrt(sum_square_displacement)
            if self.config.get('PRETTY_LOG', PRETTY_LOG):
                sys.stderr.write('%d..' % (iteration_count + 1))
            else:
                print >>sys.stderr, 'Iteration: %d, ERROR: %f' % (iteration_count + 1, rmsd)
                print >>sys.stderr, 'Q functional: %f' % Q_functional
        if self.config.get('PRETTY_LOG', PRETTY_LOG):
            sys.stderr.write('\n')
        for q, intentWeights in self.queryIntentsWeights.iteritems():
            self.queryIntentsWeights[q] = sum(intentWeights) / len(intentWeights)

    @staticmethod
    def testBackwardForward():
        positionRelevances = {'a': [0.5] * MAX_DOCS_PER_QUERY, 's': [0.5] * MAX_DOCS_PER_QUERY}
        gammas = [0.9] * 4
        layout = [False] * (MAX_DOCS_PER_QUERY + 1)
        clicks = [0] * MAX_DOCS_PER_QUERY
        alpha, beta = DbnModel.getForwardBackwardEstimates(positionRelevances, gammas, layout, clicks, False)
        x = alpha[0][0] * beta[0][0] + alpha[0][1] * beta[0][1]
        assert all(abs((a[0] * b[0] + a[1] * b[1]) / x  - 1) < 0.00001 for a, b in zip(alpha, beta))

    @staticmethod
    def getGamma(gammas, k, layout, intent):
        index = 2 * (1 if layout[k + 1] else 0) + (1 if intent else 0)
        return gammas[index]

    @staticmethod
    def getForwardBackwardEstimates(positionRelevances, gammas, layout, clicks, intent,
                                    debug=False):
        N = len(clicks)
        if debug:
            assert N + 1 == len(layout)
        alpha = [[0.0, 0.0] for i in xrange(N + 1)]
        beta = [[0.0, 0.0] for i in xrange(N + 1)]
        alpha[0] = [0.0, 1.0]
        beta[N] = [1.0, 1.0]
        # P(E_{k+1} = e, C_k | E_k = e', G, I)
        updateMatrix = [[[0.0 for e1 in [0, 1]] for e in [0, 1]] for i in xrange(N)]
        for k, C_k in enumerate(clicks):
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

        for k in xrange(N):
            for e in [0, 1]:
                alpha[k + 1][e] = sum(alpha[k][e1] * updateMatrix[k][e][e1] for e1 in [0, 1])
                beta[N - 1 - k][e] = sum(beta[N - k][e1] * updateMatrix[N - 1 - k][e1][e] for e1 in [0, 1])
        return alpha, beta

    def _getSessionEstimate(self, positionRelevances, layout, clicks, intent):
        # Returns {'a': P(A_k | I, C, G), 's': P(S_k | I, C, G), 'C': P(C | I, G), 'clicks': P(C_k | C_1, ..., C_{k-1}, I, G)} as a dict
        # sessionEstimate[True]['a'][k] = P(A_k = 1 | I = 'Fresh', C, G), probability of A_k = 0 can be calculated as 1 - p
        N = len(clicks)
        if self.config.get('DEBUG', DEBUG):
            assert N + 1 == len(layout)
        sessionEstimate = {'a': [0.0] * N, 's': [0.0] * N, 'e': [[0.0, 0.0] for k in xrange(N)], 'C': 0.0, 'clicks': [0.0] * N}
        alpha, beta = self.getForwardBackwardEstimates(positionRelevances,
                                                       self.gammas, layout, clicks, intent,
                                                       debug=self.config.get('DEBUG', DEBUG)
        )
        try:
            varphi = [((a[0] * b[0]) / (a[0] * b[0] + a[1] * b[1]), (a[1] * b[1]) / (a[0] * b[0] + a[1] * b[1])) for a, b in zip(alpha, beta)]
        except ZeroDivisionError:
            print >>sys.stderr, alpha, beta, [(a[0] * b[0] + a[1] * b[1]) for a, b in zip(alpha, beta)], positionRelevances
            sys.exit(1)
        if self.config.get('DEBUG', DEBUG):
            assert all(ph[0] < 0.01 for ph, c in zip(varphi[:N], clicks) if c != 0), (alpha, beta, varphi, clicks)
        # calculate P(C | I, G) for k = 0
        sessionEstimate['C'] = alpha[0][0] * beta[0][0] + alpha[0][1] * beta[0][1]      # == 0 + 1 * beta[0][1]
        for k, C_k in enumerate(clicks):
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

    def _get_click_probs(self, s, possibleIntents):
        """
            Returns clickProbs list:
            clickProbs[i][k] = P(C_1, ..., C_k | I=i)
            """
        # TODO: ensure that s.clicks[l] not used to calculate clickProbs[i][k] for l >= k
        positionRelevances = {}
        for intent in possibleIntents:
            positionRelevances[intent] = {}
            for r in ['a', 's']:
                positionRelevances[intent][r] = [self.urlRelevances[intent][s.query][url][r] for url in s.results]
                if self.config.get('QUERY_INDEPENDENT_PAGER', QUERY_INDEPENDENT_PAGER):
                    for k, u in enumerate(s.results):
                        if u == 'PAGER':
                            # use dummy 0 query for all fake pager URLs
                            positionRelevances[intent][r][k] = self.urlRelevances[intent][0][url][r]
        layout = [False] * len(s.layout) if self.ignoreLayout else s.layout
        return dict((i, self._getSessionEstimate(positionRelevances[i], layout, s.clicks, i)['clicks']) for i in possibleIntents)

    def get_model_relevances(self, session, intent=False):
        """
            Returns estimated relevance of each document in a given session
            based on a trained click model.

            You can make use of the fact that model trains different relevances
            for different intents by specifying `intent` argument. If it is set
            to False, simple web relevance is returned, if it is to True, then
            vertical relevance is returned, i.e., how relevant each document
            is to a vertical intent.
        """
        relevances = []
        for rank, result in enumerate(session.results):
            a = self.urlRelevances[intent][session.query][result]['a']
            s = self.urlRelevances[intent][session.query][result]['s']
            relevances.append(a * s)
        return relevances

    def predict_click_probs(self, session, intent=False):
        """
            Predicts click probabilities for a given session. Does not use clicks.
        """
        click_probs = []
        for rank, result in enumerate(session.results):
            a = self.urlRelevances[intent][session.query][result]['a']
            click_probs.append(a)
        return click_probs

    def predict_stop_probs(self, session, intent=False):
        """
            Predicts stop probabilities for each document in a session.
        """
        stop_probs = []
        for rank, result in enumerate(session.results):
            s = self.urlRelevances[intent][session.query][result]['s']
            stop_probs.append(s)
        return stop_probs

    def get_abandonment_prob(self, rank, intent=False, layout=None):
        """
            Predicts probability of stopping without click after examining document at rank `rank`.
        """
        return 1.0 - self.getGamma(self.gammas, rank, layout, intent)



class SimplifiedDbnModel(DbnModel):
    def __init__(self, ignoreIntents=True, ignoreLayout=True, config=None):
        assert ignoreIntents
        assert ignoreLayout
        DbnModel.__init__(self, (1.0, 1.0, 1.0, 1.0), ignoreIntents, ignoreLayout, config)

    def train(self, sessions):
        max_query_id = self.config.get('MAX_QUERY_ID')
        if max_query_id is None:
            print >>sys.stderr, 'WARNING: no MAX_QUERY_ID specified for', self
            max_query_id = 100000
        urlRelFractions = [defaultdict(lambda: {'a': [1.0, 1.0], 's': [1.0, 1.0]}) for q in xrange(max_query_id)]
        for s in sessions:
            query = s.query
            lastClickedPos = len(s.clicks) - 1
            for k, c in enumerate(s.clicks):
                if c != 0:
                    lastClickedPos = k
            for k, (u, c) in enumerate(zip(s.results, s.clicks[:(lastClickedPos + 1)])):
                tmpQuery = query
                if self.config.get('QUERY_INDEPENDENT_PAGER', QUERY_INDEPENDENT_PAGER) \
                        and u == 'PAGER':
                    assert self.config.get('TRANSFORM_LOG', TRANSFORM_LOG)
                    # the same dummy query for all pagers
                    query = 0
                if c != 0:
                    urlRelFractions[query][u]['a'][1] += 1
                    if k == lastClickedPos:
                        urlRelFractions[query][u]['s'][1] += 1
                    else:
                        urlRelFractions[query][u]['s'][0] += 1
                else:
                    urlRelFractions[query][u]['a'][0] += 1
                if self.config.get('QUERY_INDEPENDENT_PAGER', QUERY_INDEPENDENT_PAGER):
                    query = tmpQuery
        self.urlRelevances = dict((i,
                [defaultdict(lambda: {'a': self.config.get('DEFAULT_REL', DEFAULT_REL),
                                      's': self.config.get('DEFAULT_REL', DEFAULT_REL)}) \
                        for q in xrange(max_query_id)]) for i in [False])
        for query, d in enumerate(urlRelFractions):
            if not d:
                continue
            for url, relFractions in d.iteritems():
                self.urlRelevances[False][query][url]['a'] = relFractions['a'][1] / (relFractions['a'][1] + relFractions['a'][0])
                self.urlRelevances[False][query][url]['s'] = relFractions['s'][1] / (relFractions['s'][1] + relFractions['s'][0])


class UbmModel(ClickModel):

    gammaTypesNum = 4

    def __init__(self, ignoreIntents=True, ignoreLayout=True, explorationBias=False,
                 config=None):
        self.explorationBias = explorationBias
        ClickModel.__init__(self, ignoreIntents, ignoreLayout, config)

    def train(self, sessions):
        max_query_id = self.config.get('MAX_QUERY_ID')
        if max_query_id is None:
            print >>sys.stderr, 'WARNING: no MAX_QUERY_ID specified for', self
            max_query_id = 100000
        possibleIntents = [False] if self.ignoreIntents else [False, True]
        # alpha: intent -> query -> url -> "attractiveness probability"
        self.alpha = dict((i,
                [defaultdict(lambda: self.config.get('DEFAULT_REL', DEFAULT_REL)) \
                        for q in xrange(max_query_id)]) for i in possibleIntents)
        # gamma: freshness of the current result: gammaType -> rank -> "distance from the last click" - 1 -> examination probability
        self.gamma = [[[0.5 \
            for d in xrange(self.config.get('MAX_DOCS_PER_QUERY', MAX_DOCS_PER_QUERY))] \
                for r in xrange(self.config.get('MAX_DOCS_PER_QUERY', MAX_DOCS_PER_QUERY))] \
                    for g in xrange(self.gammaTypesNum)]
        if self.explorationBias:
            self.e = [0.5 \
                    for p in xrange(self.config.get('MAX_DOCS_PER_QUERY', MAX_DOCS_PER_QUERY))]
        if not self.config.get('PRETTY_LOG', PRETTY_LOG):
            print >>sys.stderr, '-' * 80
            print >>sys.stderr, 'Start. Current time is', datetime.now()
        for iteration_count in xrange(self.config.get('MAX_ITERATIONS', MAX_ITERATIONS)):
            self.queryIntentsWeights = defaultdict(lambda: [])
            # not like in DBN! xxxFractions[0] is a numerator while xxxFraction[1] is a denominator
            alphaFractions = dict((i, [defaultdict(lambda: [1.0, 2.0]) for q in xrange(max_query_id)]) for i in possibleIntents)
            gammaFractions = [[[[1.0, 2.0] \
                for d in xrange(self.config.get('MAX_DOCS_PER_QUERY', MAX_DOCS_PER_QUERY))] \
                    for r in xrange(self.config.get('MAX_DOCS_PER_QUERY', MAX_DOCS_PER_QUERY))] \
                        for g in xrange(self.gammaTypesNum)]
            if self.explorationBias:
                eFractions = [[1.0, 2.0] \
                        for p in xrange(self.config.get('MAX_DOCS_PER_QUERY', MAX_DOCS_PER_QUERY))]
            # E-step
            for s in sessions:
                query = s.query
                layout = [False] * len(s.layout) if self.ignoreLayout else s.layout
                if self.explorationBias:
                    explorationBiasPossible = any((l and c for (l, c) in zip(s.layout, s.clicks)))
                    firstVerticalPos = -1 if not any(s.layout[:-1]) else [k for (k, l) in enumerate(s.layout) if l][0]
                if self.ignoreIntents:
                    p_I__C_G = {False: 1.0, True: 0}
                else:
                    a = self._getSessionProb(s) * (1 - s.intentWeight)
                    b = 1 * s.intentWeight
                    p_I__C_G = {False: a / (a + b), True: b / (a + b)}
                self.queryIntentsWeights[query].append(p_I__C_G[True])
                prevClick = -1
                for rank, c in enumerate(s.clicks):
                    url = s.results[rank]
                    for intent in possibleIntents:
                        a = self.alpha[intent][query][url]
                        if self.explorationBias and explorationBiasPossible:
                            e = self.e[firstVerticalPos]
                        if c == 0:
                            g = self.getGamma(self.gamma, rank, prevClick, layout, intent)
                            gCorrection = 1
                            if self.explorationBias and explorationBiasPossible and not s.layout[k]:
                                gCorrection = 1 - e
                                g *= gCorrection
                            alphaFractions[intent][query][url][0] += a * (1 - g) / (1 - a * g) * p_I__C_G[intent]
                            self.getGamma(gammaFractions, rank, prevClick, layout, intent)[0] += g / gCorrection * (1 - a) / (1 - a * g) * p_I__C_G[intent]
                            if self.explorationBias and explorationBiasPossible:
                                eFractions[firstVerticalPos][0] += (e if s.layout[k] else e / (1 - a * g)) * p_I__C_G[intent]
                        else:
                            alphaFractions[intent][query][url][0] += 1 * p_I__C_G[intent]
                            self.getGamma(gammaFractions, rank, prevClick, layout, intent)[0] += 1 * p_I__C_G[intent]
                            if self.explorationBias and explorationBiasPossible:
                                eFractions[firstVerticalPos][0] += (e if s.layout[k] else 0) * p_I__C_G[intent]
                        alphaFractions[intent][query][url][1] += 1 * p_I__C_G[intent]
                        self.getGamma(gammaFractions, rank, prevClick, layout, intent)[1] += 1 * p_I__C_G[intent]
                        if self.explorationBias and explorationBiasPossible:
                            eFractions[firstVerticalPos][1] += 1 * p_I__C_G[intent]
                    if c != 0:
                        prevClick = rank
            if not self.config.get('PRETTY_LOG', PRETTY_LOG):
                sys.stderr.write('E')
            # M-step
            sum_square_displacement = 0.0
            for i in possibleIntents:
                for q in xrange(max_query_id):
                    for url, aF in alphaFractions[i][q].iteritems():
                        new_alpha = aF[0] / aF[1]
                        sum_square_displacement += (self.alpha[i][q][url] - new_alpha) ** 2
                        self.alpha[i][q][url] = new_alpha
            for g in xrange(self.gammaTypesNum):
                for r in xrange(self.config.get('MAX_DOCS_PER_QUERY', MAX_DOCS_PER_QUERY)):
                    for d in xrange(self.config.get('MAX_DOCS_PER_QUERY', MAX_DOCS_PER_QUERY)):
                        gF = gammaFractions[g][r][d]
                        new_gamma = gF[0] / gF[1]
                        sum_square_displacement += (self.gamma[g][r][d] - new_gamma) ** 2
                        self.gamma[g][r][d] = new_gamma
            if self.explorationBias:
                for p in xrange(self.config.get('MAX_DOCS_PER_QUERY', MAX_DOCS_PER_QUERY)):
                    new_e = eFractions[p][0] / eFractions[p][1]
                    sum_square_displacement += (self.e[p] - new_e) ** 2
                    self.e[p] = new_e
            if not self.config.get('PRETTY_LOG', PRETTY_LOG):
                sys.stderr.write('M\n')
            rmsd = math.sqrt(sum_square_displacement)
            if self.config.get('PRETTY_LOG', PRETTY_LOG):
                sys.stderr.write('%d..' % (iteration_count + 1))
            else:
                print >>sys.stderr, 'Iteration: %d, ERROR: %f' % (iteration_count + 1, rmsd)
        if self.config.get('PRETTY_LOG', PRETTY_LOG):
            sys.stderr.write('\n')
        for q, intentWeights in self.queryIntentsWeights.iteritems():
            self.queryIntentsWeights[q] = sum(intentWeights) / len(intentWeights)

    def _getSessionProb(self, s):
        clickProbs = self._get_click_probs(s, [False, True])
        N = len(s.clicks)
        return clickProbs[False][N - 1] / clickProbs[True][N - 1]

    @staticmethod
    def getGamma(gammas, k, prevClick, layout, intent):
        index = (2 if layout[k] else 0) + (1 if intent else 0)
        return gammas[index][k][k - prevClick - 1]

    def _get_click_probs(self, s, possibleIntents):
        """
            Returns clickProbs list
            clickProbs[i][k] = P(C_1, ..., C_k | I=i)
            """
        clickProbs = dict((i, []) for i in possibleIntents)
        firstVerticalPos = -1 if not any(s.layout[:-1]) else [k for (k, l) in enumerate(s.layout) if l][0]
        prevClick = -1
        layout = [False] * len(s.layout) if self.ignoreLayout else s.layout
        for rank, c in enumerate(s.clicks):
            url = s.results[rank]
            prob = {False: 0.0, True: 0.0}
            for i in possibleIntents:
                a = self.alpha[i][s.query][url]
                g = self.getGamma(self.gamma, rank, prevClick, layout, i)
                if self.explorationBias and any(s.layout[k] and s.clicks[k] for k in xrange(rank)) and not s.layout[rank]:
                    g *= 1 - self.e[firstVerticalPos]
                prevProb = 1 if rank == 0 else clickProbs[i][-1]
                if c == 0:
                    clickProbs[i].append(prevProb * (1 - a * g))
                else:
                    clickProbs[i].append(prevProb * a * g)
            if c != 0:
                prevClick = rank
        return clickProbs


class EbUbmModel(UbmModel):
    def __init__(self, ignoreIntents=True, ignoreLayout=True, config=None):
        UbmModel.__init__(self, ignoreIntents, ignoreLayout, explorationBias=True,
                         config=config)


class DcmModel(ClickModel):

    gammaTypesNum = 4

    def train(self, sessions):
        max_query_id = self.config.get('MAX_QUERY_ID')
        if max_query_id is None:
            print >>sys.stderr, 'WARNING: no MAX_QUERY_ID specified for', self
            max_query_id = 100000
        possibleIntents = [False] if self.ignoreIntents else [False, True]
        urlRelFractions = dict((i, [defaultdict(lambda: [1.0, 1.0]) for q in xrange(max_query_id)]) for i in possibleIntents)
        gammaFractions = [[[1.0, 1.0] for g in xrange(self.gammaTypesNum)] \
                for r in xrange(self.config.get('MAX_DOCS_PER_QUERY', MAX_DOCS_PER_QUERY))]
        for s in sessions:
            query = s.query
            layout = [False] * len(s.layout) if self.ignoreLayout else s.layout
            lastClickedPos = self.config.get('MAX_DOCS_PER_QUERY', MAX_DOCS_PER_QUERY) - 1
            for k, c in enumerate(s.clicks):
                if c != 0:
                    lastClickedPos = k
            intentWeights = {False: 1.0} if self.ignoreIntents else {False: 1 - s.intentWeight, True: s.intentWeight}
            for k, (u, c) in enumerate(zip(s.results, s.clicks[:(lastClickedPos + 1)])):
                for i in possibleIntents:
                    if c != 0:
                        urlRelFractions[i][query][u][1] += intentWeights[i]
                        if k == lastClickedPos:
                            self.getGamma(gammaFractions[k], k, layout, i)[1] += intentWeights[i]
                        else:
                            self.getGamma(gammaFractions[k], k, layout, i)[0] += intentWeights[i]
                    else:
                        urlRelFractions[i][query][u][0] += intentWeights[i]
        self.urlRelevances = dict((i,
                [defaultdict(lambda: self.config.get('DEFAULT_REL', DEFAULT_REL)) \
                        for q in xrange(max_query_id)]) for i in possibleIntents)
        self.gammas = [[0.5 for g in xrange(self.gammaTypesNum)] \
                for r in xrange(self.config.get('MAX_DOCS_PER_QUERY', MAX_DOCS_PER_QUERY))]
        for i in possibleIntents:
            for query, d in enumerate(urlRelFractions[i]):
                if not d:
                    continue
                for url, relFractions in d.iteritems():
                    self.urlRelevances[i][query][url] = relFractions[1] / (relFractions[1] + relFractions[0])
        for k in xrange(self.config.get('MAX_DOCS_PER_QUERY', MAX_DOCS_PER_QUERY)):
            for g in xrange(self.gammaTypesNum):
                self.gammas[k][g] = gammaFractions[k][g][0] / (gammaFractions[k][g][0] + gammaFractions[k][g][1])

    def _get_click_probs(self, s, possibleIntents):
        clickProbs = {False: [], True: []}          # P(C_1, ..., C_k)
        query = s.query
        layout = [False] * len(s.layout) if self.ignoreLayout else s.layout
        for i in possibleIntents:
            examinationProb = 1.0       # P(C_1, ..., C_{k - 1}, E_k = 1)
            for k, c in enumerate(s.clicks):
                r = self.urlRelevances[i][query][s.results[k]]
                prevProb = 1 if k == 0 else clickProbs[i][-1]
                if c == 0:
                    # P(C_1, ..., C_k = 0) = P(C_1, ..., C_{k-1}) - P(C_1, ..., C_k = 1)
                    clickProbs[i].append(prevProb - examinationProb * r)
                    # P(C_1, ..., C_k, E_{k+1} = 1) = P(E_{k+1} = 1 | C_k, E_k = 1) * P(C_k | E_k = 1) *  P(C_1, ..., C_{k - 1}, E_k = 1)
                    examinationProb *= 1 - r
                else:
                    clickProbs[i].append(examinationProb * r)
                    # P(C_1, ..., C_k, E_{k+1} = 1) = P(E_{k+1} = 1 | C_k, E_k = 1) * P(C_k | E_k = 1) *  P(C_1, ..., C_{k - 1}, E_k = 1)
                    examinationProb *= self.getGamma(self.gammas[k], k, layout, i) * r
        return clickProbs

    @staticmethod
    def getGamma(gammas, k, layout, intent):
        return DbnModel.getGamma(gammas, k, layout, intent)

