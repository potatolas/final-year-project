file_string = ['''
def __init__(self, *args, **kwargs):
    super(LinkSpider, self).__init__(*args, **kwargs)

def parse(self, response):
    self._logger.debug("crawled url {}".format(response.request.url))
    cur_depth = 0
    if 'curdepth' in response.meta:
        cur_depth = response.meta['curdepth']

    # capture raw response
    item = RawResponseItem()
    # populated from response.meta
    item['appid'] = response.meta['appid']
    item['crawlid'] = response.meta['crawlid']
    item['attrs'] = response.meta['attrs']

    # populated from raw HTTP response
    item["url"] = response.request.url
    item["response_url"] = response.url
    item["status_code"] = response.status
    item["status_msg"] = "OK"
    item["response_headers"] = self.reconstruct_headers(response)
    item["request_headers"] = response.request.headers
    item["body"] = response.body
    item["encoding"] = response.encoding
    item["links"] = []

    # determine whether to continue spidering
    if cur_depth >= response.meta['maxdepth']:
        self._logger.debug("Not spidering links in '{}' because" \
            " cur_depth={} >= maxdepth={}".format(
                                                    response.url,
                                                    cur_depth,
                                                    response.meta['maxdepth']))
    else:
        # we are spidering -- yield Request for each discovered link
        link_extractor = LinkExtractor(
                        allow_domains=response.meta['allowed_domains'],
                        allow=response.meta['allow_regex'],
                        deny=response.meta['deny_regex'],
                        deny_extensions=response.meta['deny_extensions'])

        for link in link_extractor.extract_links(response):
            # link that was discovered
            the_url = link.url
            item["links"].append({"url": the_url, "text": link.text, })
            req = scrapy.http.Request(the_url, callback=self.parse)

            req.meta['priority'] = response.meta['priority'] - 10
            req.meta['curdepth'] = response.meta['curdepth'] + 1

            if 'useragent' in response.meta and \
                    response.meta['useragent'] is not None:
                req.headers['User-Agent'] = response.meta['useragent']

            self._logger.debug("Trying to follow link '{}'".format(req.url))
            yield req

    # raw response has been processed, yield to item pipeline
    yield item
''',

'''
def __init__(self, *args, **kwargs):
        super(WanderingSpider, self).__init__(*args, **kwargs)

def parse(self, response):
    # debug output for receiving the url
    self._logger.debug("crawled url {}".format(response.request.url))

    # step counter for how many pages we have hit
    step = 0
    if 'step' in response.meta:
        step = response.meta['step']

    # Create Item to send to kafka
    # capture raw response
    item = RawResponseItem()
    # populated from response.meta
    item['appid'] = response.meta['appid']
    item['crawlid'] = response.meta['crawlid']
    item['attrs'] = response.meta['attrs']
    # populated from raw HTTP response
    item["url"] = response.request.url
    item["response_url"] = response.url
    item["status_code"] = response.status
    item["status_msg"] = "OK"
    item["response_headers"] = self.reconstruct_headers(response)
    item["request_headers"] = response.request.headers
    item["body"] = response.body
    item["encoding"] = response.encoding
    item["links"] = []
    # we want to know how far our spider gets
    if item['attrs'] is None:
        item['attrs'] = {}

    item['attrs']['step'] = step

    self._logger.debug("Finished creating item")

    # determine what link we want to crawl
    link_extractor = LinkExtractor(
                        allow_domains=response.meta['allowed_domains'],
                        allow=response.meta['allow_regex'],
                        deny=response.meta['deny_regex'],
                        deny_extensions=response.meta['deny_extensions'])

    links = link_extractor.extract_links(response)

    # there are links on the page
    if len(links) > 0:
        self._logger.debug("Attempting to find links")
        link = random.choice(links)
        req = scrapy.http.Request(link.url, callback=self.parse)

        # increment our step counter for this crawl job
        req.meta['step'] = step + 1

        # pass along our user agent as well
        if 'useragent' in response.meta and \
                    response.meta['useragent'] is not None:
                req.headers['User-Agent'] = response.meta['useragent']

        # debug output
        self._logger.debug("Trying to yield link '{}'".format(req.url))

        # yield the Request to the scheduler
        yield req
    else:
        self._logger.info("Did not find any more links")

    # raw response has been processed, yield to item pipeline
    yield item
''',

'''
def setUp(self, s):
        self.mpm = MetaPassthroughMiddleware(MagicMock())
        self.mpm.logger = MagicMock()
        self.mpm.logger.debug = MagicMock()

def test_mpm_middleware(self):
    # create fake response
    a = MagicMock()
    a.meta = {
        'key1': 'value1',
        'key2': 'value2'
    }

    yield_count = 0
    # test all types of results from a spider
    # dicts, items, or requests
    test_list = [
        {},
        scrapy.Item(),
        scrapy.http.Request('http://istresearch.com')
    ]

    for item in self.mpm.process_spider_output(a, test_list, MagicMock()):
        if isinstance(item, scrapy.http.Request):
            self.assertEqual(a.meta, item.meta)
        yield_count += 1

    self.assertEqual(yield_count, 3)

    # 1 debug for the method, 1 debug for the request
    self.assertEqual(self.mpm.logger.debug.call_count, 2)

    # test meta unchanged if already exists
    r = scrapy.http.Request('http://aol.com')
    r.meta['key1'] = 'othervalue'

    for item in self.mpm.process_spider_output(a, [r], MagicMock()):
        # key1 value1 did not pass through, since it was already set
        self.assertEqual(item.meta['key1'], 'othervalue')
        # key2 was not set, therefor it passed through
        self.assertEqual(item.meta['key2'], 'value2')
''',

'''
appid = scrapy.Field()
crawlid = scrapy.Field()
url = scrapy.Field()
response_url = scrapy.Field()
status_code = scrapy.Field()
status_msg = scrapy.Field()
response_headers = scrapy.Field()
request_headers = scrapy.Field()
body = scrapy.Field()
links = scrapy.Field()
attrs = scrapy.Field()
success = scrapy.Field()
exception = scrapy.Field()
encoding = scrapy.Field()
''',

'''
def _set_crawler(self, crawler):
        super(RedisSpider, self)._set_crawler(crawler)
        scrapy.signals.connect(self.spider_idle,
                                     signal=scrapy.signals.spider_idle)

def spider_idle(self):
    raise scrapy.exceptions.DontCloseSpider

def parse(self, response):
    raise NotImplementedError("Please implement parse() for your spider")

def set_logger(self, logger):

    self._logger = logger

def reconstruct_headers(self, response):
    """
    Purpose of this method is to reconstruct the headers dictionary that
    is normally passed in with a "response" object from scrapy.

    Args:
        response: A scrapy response object

    Returns: A dictionary that mirrors the "response.headers" dictionary
    that is normally within a response object

    Raises: None
    Reason: Originally, there was a bug where the json.dumps() did not
    properly serialize the headers. This method is a way to circumvent
    the known issue
    """

    header_dict = {}
    # begin reconstructing headers from scratch...
    for key in list(response.headers.keys()):
        key_item_list = []
        key_list = response.headers.getlist(key)
        for item in key_list:
            key_item_list.append(item)
        header_dict[key] = key_item_list
    return header_dict
''',

'''
def setUp(self, s):
    self.mpm = MetaPassthroughMiddleware(MagicMock())
    self.mpm.logger = MagicMock()
    self.mpm.logger.debug = MagicMock()

def test_mpm_middleware(self):
    # create fake response
    a = MagicMock()
    a.meta = {
        'key1': 'value1',
        'key2': 'value2'
    }

    yield_count = 0
    # test all types of results from a spider
    # dicts, items, or requests
    test_list = [
        {},
        scrapy.Item(),
        scrapy.http.Request('http://istresearch.com')
    ]

    for item in self.mpm.process_spider_output(a, test_list, MagicMock()):
        if isinstance(item, Request):
            self.assertEqual(a.meta, item.meta)
        yield_count += 1

    self.assertEqual(yield_count, 3)

    # 1 debug for the method, 1 debug for the request
    self.assertEqual(self.mpm.logger.debug.call_count, 2)

    # test meta unchanged if already exists
    r = scrapy.http.Request('http://aol.com')
    r.meta['key1'] = 'othervalue'

    for item in self.mpm.process_spider_output(a, [r], MagicMock()):
        # key1 value1 did not pass through, since it was already set
        self.assertEqual(item.meta['key1'], 'othervalue')
        # key2 was not set, therefor it passed through
        self.assertEqual(item.meta['key2'], 'value2')
''']
