#!/usr/bin/env python
#coding: utf-8

# Read from sys.stdin
# Input format (see README.md for more details):
# hash \t query \t region \t intent_probability \t url_list (json) \t layout (json) \t clicks (json)


from clickmodels.inference import *
from clickmodels.input_reader import InputReader

try:
    from config import *
except:
    from clickmodels.config_sample import *


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

    readInput = InputReader(MIN_DOCS_PER_QUERY, MAX_DOCS_PER_QUERY,
                            EXTENDED_LOG_FORMAT, SERP_SIZE,
                            TRAIN_FOR_METRIC,
                            discard_no_clicks=True)
    sessions = readInput(sys.stdin)

    if TRAIN_FOR_METRIC and PRINT_EBU_STATS:
        # ---------------------------------------------------------------
        #                           For EBU
        # ---------------------------------------------------------------
        # Relevance -> P(Click | Relevance)
        p_C_R_frac = defaultdict(lambda: [0, 0.0001])
        # Relevance -> P(Leave | Click, Relevance)
        p_L_C_R_frac = defaultdict(lambda: [0, 0.0001])
        for s in sessions:
            lastClickPos = max((i for i, c in enumerate(s.clicks) if c != 0))
            for i in xrange(lastClickPos + 1):
                u = s.results[i]
                if s.clicks[i] != 0:
                    p_C_R_frac[u][0] += 1
                    if i == lastClickPos:
                        p_L_C_R_frac[u][0] += 1
                    p_L_C_R_frac[u][1] += 1
                p_C_R_frac[u][1] += 1

        for u in ['IRRELEVANT', 'RELEVANT', 'USEFUL', 'VITAL']:
            print 'P(C|%s)\t%f\tP(L|C,%s)\t%f' % (u, float(p_C_R_frac[u][0]) / p_C_R_frac[u][1], u, float(p_L_C_R_frac[u][0]) / p_L_C_R_frac[u][1])
    # ---------------------------------------------------------------

    if len(sys.argv) > 1:
        with open(sys.argv[1]) as test_clicks_file:
            testSessions = readInput(test_clicks_file)
    else:
        testSessions = sessions

    if TRANSFORM_LOG:
        assert EXTENDED_LOG_FORMAT
        sessions, testSessions = (
            [x for x in (InputReader.mergeExtraToSessionItem(s, SERP_SIZE) for s in ss) if x] \
                for ss in [sessions, testSessions]
        )
    else:
        sessions, testSessions = (
            [s for s in ss if InputReader.mergeExtraToSessionItem(s, SERP_SIZE)] \
                for ss in [sessions, testSessions]
        )

    print 'Train sessions: %d, test sessions: %d' % (len(sessions), len(testSessions))
    print 'Number of train sessions with 10+ urls shown:', len([s for s in sessions if len(s.results) > SERP_SIZE + 1])
    #clickProbs = [0.0] * MAX_DOCS_PER_QUERY
    #counts = [0] * MAX_DOCS_PER_QUERY
    #for s in sessions:
        #for i, c in enumerate(s.clicks):
            #clickProbs[i] += 1 if c else 0
            #counts[i] += 1
    #print '\t'.join((str(x / cnt if cnt else x) for (x, cnt) in zip(clickProbs, counts)))
    #sys.exit(0)

    config = {
        'MAX_QUERY_ID': readInput.current_query_id + 1,
        'MAX_ITERATIONS': MAX_ITERATIONS,
        'DEBUG': DEBUG,
        'PRETTY_LOG': PRETTY_LOG,
        'MAX_DOCS_PER_QUERY': MAX_DOCS_PER_QUERY,
        'SERP_SIZE': SERP_SIZE,
        'TRANSFORM_LOG': TRANSFORM_LOG,
        'QUERY_INDEPENDENT_PAGER': QUERY_INDEPENDENT_PAGER,
        'DEFAULT_REL': DEFAULT_REL
    }
    del readInput       # needed to minimize memory consumption (see gc.collect() below)

    if 'Baseline' in USED_MODELS:
        baselineModel = ClickModel(config=config)
        baselineModel.train(sessions)
        print 'Baseline:', baselineModel.test(testSessions)

    if 'SDBN' in USED_MODELS:
        sdbnModel = SimplifiedDbnModel(config=config)
        sdbnModel.train(sessions)
        if TRANSFORM_LOG:
            print '(a_p, s_p) = ', sdbnModel.urlRelevances[False][0]['PAGER']
        print 'SDBN:', sdbnModel.test(testSessions)
        del sdbnModel        # needed to minimize memory consumption (see gc.collect() below)

    if 'UBM' in USED_MODELS:
        ubmModel = UbmModel(config=config)
        ubmModel.train(sessions)
        if TRAIN_FOR_METRIC:
            print '\n'.join(['%s\t%f' % r for r in \
                [(x, ubmModel.alpha[False][0][x]) for x in \
                    ['IRRELEVANT', 'RELEVANT', 'USEFUL', 'VITAL']]])
            for d in xrange(MAX_DOCS_PER_QUERY):
                for r in xrange(MAX_DOCS_PER_QUERY):
                    print ('%.4f ' % (ubmModel.gamma[0][r][MAX_DOCS_PER_QUERY - 1 - d] if r + d >= MAX_DOCS_PER_QUERY - 1 else 0)),
                print
        print 'UBM', ubmModel.test(testSessions)
        del ubmModel       # needed to minimize memory consumption (see gc.collect() below)

    if 'UBM-IA' in USED_MODELS:
        ubmModel = UbmModel(ignoreIntents=False, ignoreLayout=False, config=config)
        ubmModel.train(sessions)
        print 'UBM-IA', ubmModel.test(testSessions)
        del ubmModel       # needed to minimize memory consumption (see gc.collect() below)

    if 'EB_UBM' in USED_MODELS:
        ebUbmModel = EbUbmModel(config=config)
        ebUbmModel.train(sessions)
        # print 'Exploration bias:', ebUbmModel.e
        print 'EB_UBM', ebUbmModel.test(testSessions)
        del ebUbmModel       # needed to minimize memory consumption (see gc.collect() below)

    if 'EB_UBM-IA' in USED_MODELS:
        ebUbmModel = EbUbmModel(ignoreIntents=False, ignoreLayout=False, config=config)
        ebUbmModel.train(sessions)
        # print 'Exploration bias:', ebUbmModel.e
        print 'EB_UBM-IA', ebUbmModel.test(testSessions)
        del ebUbmModel       # needed to minimize memory consumption (see gc.collect() below)

    if 'DCM' in USED_MODELS:
        dcmModel = DcmModel(config=config)
        dcmModel.train(sessions)
        if TRAIN_FOR_METRIC:
            print '\n'.join(['%s\t%f' % r for r in \
                [(x, dcmModel.urlRelevances[False][0][x]) for x in \
                    ['IRRELEVANT', 'RELEVANT', 'USEFUL', 'VITAL']]])
            print 'DCM gammas:', dcmModel.gammas
        print 'DCM', dcmModel.test(testSessions)
        del dcmModel       # needed to minimize memory consumption (see gc.collect() below)

    if 'DCM-IA' in USED_MODELS:
        dcmModel = DcmModel(ignoreIntents=False, ignoreLayout=False, config=config)
        dcmModel.train(sessions)
        # print 'DCM gammas:', dcmModel.gammas
        print 'DCM-IA', dcmModel.test(testSessions)
        del dcmModel       # needed to minimize memory consumption (see gc.collect() below)

    if 'DBN' in USED_MODELS:
        dbnModel = DbnModel((0.9, 0.9, 0.9, 0.9), config=config)
        dbnModel.train(sessions)
        print 'DBN:', dbnModel.test(testSessions)
        for session in testSessions:
            print "relevances", dbnModel.get_model_relevances(session)
            print "clicks", dbnModel.generate_clicks(session)
        del dbnModel       # needed to minimize memory consumption (see gc.collect() below)

    if 'DBN-IA' in USED_MODELS:
        for gammas in allCombinations:
            gc.collect()
            dbnModel = DbnModel(gammas, ignoreIntents=False, ignoreLayout=False, config=config)
            dbnModel.train(sessions)
            print 'DBN-IA: %.2f %.2f %.2f %.2f' % gammas, dbnModel.test(testSessions)

