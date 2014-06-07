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
            extra = {}
            urlsObserved = 0
            if EXTENDED_LOG_FORMAT:
                maxLen = MAX_DOCS_PER_QUERY
                if TRANSFORM_LOG:
                    maxLen -= MAX_DOCS_PER_QUERY // SERP_SIZE
                urls, _ = self.convertToList(urls, '', maxLen)
                for u in urls:
                    if u == '':
                        break
                    urlsObserved += 1
                urls = urls[:urlsObserved]
                layout, _ = self.convertToList(layout, False, urlsObserved)
                clicks, extra = self.convertToList(clicks, 0, urlsObserved)
            else:
                urls = urls[:MAX_DOCS_PER_QUERY]
                urlsObserved = len(urls)
                layout = layout[:urlsObserved]
                clicks = clicks[:urlsObserved]
            if urlsObserved < MIN_DOCS_PER_QUERY:
                continue
            if self.discardNoClicks and not any(clicks):
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
            # add fake G_{MAX_DOCS_PER_QUERY+1} to simplify gamma calculation:
            layout.append(False)
            url_ids = []
            for u in urls:
                if u in ['_404', 'STUPID', 'VIRUS', 'SPAM']:
                    # convert Yandex-specific fields to standard ones
                    assert TRAIN_FOR_METRIC
                    u = 'IRRELEVANT'
                if u.startswith('RELEVANT_'):
                    # convert Yandex-specific fields to standard ones
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
            sessions.append(SessionItem(intentWeight, query_id, url_ids, layout, clicks, extra))
        return sessions

    @staticmethod
    def convertToList(sparseDict, defaultElem=0, maxLen=MAX_DOCS_PER_QUERY):
        """ Convert dict of the format {"0": doc0, "13": doc13} to the list of the length MAX_DOCS_PER_QUERY """
        convertedList = [defaultElem] * maxLen
        extra = {}
        for k, v in sparseDict.iteritems():
            try:
                convertedList[int(k)] = v
            except (ValueError, IndexError):
                extra[k] = v
        return convertedList, extra

    @staticmethod
    def mergeExtraToSessionItem(s):
        """ Put pager click into the session item (presented as a fake URL) """
        if s.extraclicks.get('TRANSFORMED', False):
            return s
        else:
            newUrls = []
            newLayout = []
            newClicks = []
            a = 0
            while a + SERP_SIZE <= len(s.results):
                b = a + SERP_SIZE
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
                assert len(newUrls) < len(s.results) + MAX_DOCS_PER_QUERY / SERP_SIZE, (len(s.results), len(newUrls))
            return SessionItem(s.intentWeight, s.query, newUrls, newLayout, newClicks, {'TRANSFORMED': True})

