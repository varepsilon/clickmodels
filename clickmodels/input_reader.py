from collections import namedtuple
import json

DEBUG = False

SessionItem = namedtuple('SessionItem', ['intentWeight', 'query', 'results', 'layout', 'clicks', 'extraclicks'])

class InputReader:
    def __init__(self, min_docs_per_query, max_docs_per_query,
                 extended_log_format, serp_size,
                 train_for_metric,
                 discard_no_clicks=True):
        self.url_to_id = {}
        self.query_to_id = {}
        self.current_url_id = 1
        self.current_query_id = 0

        self.min_docs_per_query = min_docs_per_query
        self.max_docs_per_query = max_docs_per_query
        self.extended_log_format = extended_log_format
        self.serp_size = serp_size
        self.train_for_metric = train_for_metric
        self.discard_no_clicks = discard_no_clicks

    def __call__(self, f):
        sessions = []
        for line in f:
            hash_digest, query, region, intentWeight, urls, layout, clicks = line.rstrip().split('\t')
            urls, layout, clicks = map(json.loads, [urls, layout, clicks])
            extra = {}
            urlsObserved = 0
            if self.extended_log_format:
                maxLen = self.max_docs_per_query
                if TRANSFORM_LOG:
                    maxLen -= self.max_docs_per_query // self.serp_size
                urls, _ = self.convertToList(urls, '', maxLen)
                for u in urls:
                    if u == '':
                        break
                    urlsObserved += 1
                urls = urls[:urlsObserved]
                layout, _ = self.convertToList(layout, False, urlsObserved)
                clicks, extra = self.convertToList(clicks, 0, urlsObserved)
            else:
                urls = urls[:self.max_docs_per_query]
                urlsObserved = len(urls)
                layout = layout[:urlsObserved]
                clicks = clicks[:urlsObserved]
            if urlsObserved < self.min_docs_per_query:
                continue
            if self.discard_no_clicks and not any(clicks):
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
            # add fake G_{self.max_docs_per_query+1} to simplify gamma calculation:
            layout.append(False)
            url_ids = []
            for u in urls:
                if u in ['_404', 'STUPID', 'VIRUS', 'SPAM']:
                    # convert Yandex-specific fields to standard ones
                    assert self.train_for_metric
                    u = 'IRRELEVANT'
                if u.startswith('RELEVANT_'):
                    # convert Yandex-specific fields to standard ones
                    assert self.train_for_metric
                    u = 'RELEVANT'
                if u in self.url_to_id:
                    if self.train_for_metric:
                        url_ids.append(u)
                    else:
                        url_ids.append(self.url_to_id[u])
                else:
                    urlid = self.current_url_id
                    if self.train_for_metric:
                        url_ids.append(u)
                    else:
                        url_ids.append(urlid)
                    self.url_to_id[u] = urlid
                    self.current_url_id += 1
            sessions.append(SessionItem(intentWeight, query_id, url_ids, layout, clicks, extra))
        return sessions

    @staticmethod
    def convertToList(sparseDict, defaultElem, maxLen):
        """ Convert dict of the format {"0": doc0, "13": doc13} to the list of the length maxLen"""
        convertedList = [defaultElem] * maxLen
        extra = {}
        for k, v in sparseDict.iteritems():
            try:
                convertedList[int(k)] = v
            except (ValueError, IndexError):
                extra[k] = v
        return convertedList, extra

    @staticmethod
    def mergeExtraToSessionItem(s, serp_size):
        """ Put pager click into the session item (presented as a fake URL) """
        if s.extraclicks.get('TRANSFORMED', False):
            return s
        else:
            newUrls = []
            newLayout = []
            newClicks = []
            a = 0
            while a + serp_size <= len(s.results):
                b = a + serp_size
                newUrls += s.results[a:b]
                newLayout += s.layout[a:b]
                newClicks += s.clicks[a:b]
                # TODO: try different fake urls for different result pages (page_1, page_2, etc.)
                newUrls.append('PAGER')
                newLayout.append(False)
                newClicks.append(1)
                a = b
            newClicks[-1] = 0 if a == len(s.results) else 1
            newLayout.append(False)
            if DEBUG:
                assert len(newUrls) == len(newClicks)
                assert len(newUrls) + 1 == len(newLayout), (len(newUrls), len(newLayout))
                assert len(newUrls) < len(s.results) + self.max_docs_per_query / serp_size, (len(s.results), len(newUrls))
            return SessionItem(s.intentWeight, s.query, newUrls, newLayout, newClicks, {'TRANSFORMED': True})

