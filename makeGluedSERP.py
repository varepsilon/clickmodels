#!/usr/bin/env python3

import sys
import os
import json
import urllib
import random
import glob

from urllib.request import quote
from collections import defaultdict

END_PAGE = 30
MIN_PAGE_LEN = 0.66 * END_PAGE
SEED = 42
LANG = 'en'

RELEVANCE_SCALE = {
    'ru': ['НЕ НАШЁЛ', 'ПРОДОЛЖУ ИСКАТЬ', 'ИНФОРМАЦИЯ НАЙДЕНА'],
    'en': ['NOT FOUND', 'CONTINUE SEARCHING', 'INFORMATION FOUND'],
}

def genFreshYaURL(query):
    return 'http://yandex.{}/yandsearch?text={}&xjst=1&time_from=-3dE'.format(
            LANG,
            quote(query)
    )


def genSERPItem(index, title, snippet, url, fresh=False):
    scale = RELEVANCE_SCALE[LANG]
    x = ''.join('''<input type="radio" name="mark-{0}" id="result{0}-{1}" value="{1}" />
        <label for="result{0}-{1}">{2}</label>
    '''.format(index, k[0], k[1]) for k in enumerate(scale))
    return '''
<li class="serp_item {3}">
    <a class="title" href="{2}">{0}</a>
    <div class="snippet">{1}</div>
    <a class="link" href="{2}">{2}</a>
    <span class="assessment">
{4}
    </span>
</li>
    '''.format(title, snippet, url, 'fresh_item' if fresh else 'web_item', x)


def genBeginGlue(query):
    body_list = []
    body_list.append('<hr />')
    body_list.append('<a class="fresh_link" href="{}">{} "<b>{}</b>"</a>'.format(
                    genFreshYaURL(query),
                    'Свежие результаты по запросу' if LANG == 'ru' \
                            else 'Fresh results for the query',
                    query
            )
    )
    return body_list


def genEndGlue(query):
    body_list = []
    body_list.append('<a class="more_fresh" href="{}">'\
            '{} "<b>{}</b>"</a>'.format(
                    genFreshYaURL(query),
                    'Ещё свежие результаты по запросу' if LANG == 'ru' \
                            else 'More fresh results for the query',
                    query
            )
    )
    body_list.append('<hr />')
    return body_list


def genSERP(query, id, web_results, fresh_results, layout):
    '''
        - layout should be of form [True, False, True, True] where "True" means glue
        category is set to "fresh".
    '''
    head = '''
<html>
<head>
    <link rel="stylesheet" type="text/css" href="serp.css" />
    <META http-equiv="Content-Type" content="text/html; charset=UTF-8">
</head>
    <h1>{} "{}"</h1>
<body>
<ol class="result_list">
<form name="assessments" action="http://daiquiri.yandex.ru:8000/{}/save" method="POST">
    '''.format(
            'Найдено по запросу ' if LANG == 'ru' else 'Results for the query',
            query,
            id
    )

    tail = '''
<input type="submit" value="{}" />
</form>
</ol>
</body>
</html>
    '''.format('Перейти к следующему заданию' if LANG == 'ru' else 'Next task')

    body_list = []
    web_count = 0
    fresh_count = 0
    N = len(layout)
    for i in range(N):
        if layout[i]:
            if (i == 0 or not layout[i-1]) and i != N - 1 and layout[i+1]:
                body_list += genBeginGlue(query)
            r = fresh_results[fresh_count]
            fresh_count += 1
            body_list.append(genSERPItem(i, r['title'], r['snippet'], r['url'], True))
            if (i == N - 1 or not layout[i+1]) and i != 0 and layout[i-1]:
                body_list += genEndGlue(query)
        else:
            r = web_results[web_count]
            web_count += 1
            body_list.append(genSERPItem(i, r['title'], r['snippet'], r['url'], False))

    return '\n'.join([head, '\n'.join(body_list), tail])


if __name__ == '__main__':
    old_files = glob.glob('html/serp*.html')
    for f in old_files:
        os.unlink(f)

    queries = []
    for index, line in enumerate(sys.stdin):
        fields = line.rstrip().split('\t')
        query = fields[0]
        results = json.loads(fields[1])
        web_results = results['web']
        fresh_results = results['fresh']
        layout = [False, True, True]       # 'web', 'fresh', 'fresh'
        serp = genSERP(query, index, web_results, fresh_results, layout)
        with open('html/serp{}.html'.format(index), 'w') as f:
            f.write(serp)
        queries.append(query)

