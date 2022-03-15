file_string = [

'''
def start_requests(self):
    # AttributeError: 'TestSpider' object has no attribute 'arg1'
    # self.logger.debug('self.arg1: %s' % self.arg1)  # self.arg1: val1
    # repr: 'Test\' "\xe6\xb5\x8b\xe8\xaf\x95'
    if sys.version_info[0] < 3 and getattr(self, 'arg1', None):
        self.logger.debug('self.arg1: ' + self.arg1.decode('utf-8'))
    else:
        self.logger.debug('self.arg1: %s' % getattr(self, 'arg1', None))  # self.arg1: None
    # self.logger.debug(self.settings.attributes.keys())
    # self.logger.debug(self.settings.attributes.values())
    # self.logger.debug(self.settings)
    # self.logger.debug(self.settings.attributes.items())
    if getattr(self, 'url', None):
        yield scrapy.Request(self.url)

    self.logger.debug('JOB: %s' % self.settings.get('JOB'))
    self.logger.debug('USER_AGENT: %s' % self.settings.get('USER_AGENT'))  # Scrapy/1.5.0 (+https://scrapy.org)
    self.logger.debug('ROBOTSTXT_OBEY: %s' % self.settings.getbool('ROBOTSTXT_OBEY'))  # True
    self.logger.debug('COOKIES_ENABLED: %s' % self.settings.getbool('COOKIES_ENABLED'))  # True
    self.logger.debug('CONCURRENT_REQUESTS: %s' % self.settings.getint('CONCURRENT_REQUESTS'))  # 16
    self.logger.debug('DOWNLOAD_DELAY: %s' % self.settings.getint('DOWNLOAD_DELAY'))  # 0
    self.logger.debug('CLOSESPIDER_TIMEOUT: %s' % self.settings.getint('CLOSESPIDER_TIMEOUT'))  # 0
    self.logger.debug('CLOSESPIDER_PAGECOUNT: %s' % self.settings.getint('CLOSESPIDER_PAGECOUNT'))  # 0

    self.log(u'Chinese characters: 汉字字符')
    self.logger.debug('2018-08-20 09:13:06 [apps_redis] DEBUG: Resuming crawl (675840 requests scheduled)')

    yield scrapy.Request('http://httpbin.org/redirect/1')
    yield scrapy.Request('http://httpbin.org/status/404')
    yield scrapy.Request('http://httpbin.org/headers')
    yield scrapy.Request('http://httpbin.org/headers')
    yield scrapy.Request('https://google.com/')

def parse(self, response):
    self.log(response.text)
    if self.num == 1:
        yield scrapy.Request('https://www.baidu.com/')

    yield {u'Chinese 汉字 %s' % self.num: ''.join('0' + str(i) if i < 10 else str(i) for i in range(1, 100))}

    self.num += 1
''',

'''
def start_requests(self):
    # AttributeError: 'TestSpider' object has no attribute 'arg1'
    # self.logger.debug('self.arg1: %s' % self.arg1)  # self.arg1: val1
    # repr: 'Test\' "\xe6\xb5\x8b\xe8\xaf\x95'
    if sys.version_info[0] < 3 and getattr(self, 'arg1', None):
        self.logger.debug('self.arg1: ' + self.arg1.decode('utf-8'))
    else:
        self.logger.debug('self.arg1: %s' % getattr(self, 'arg1', None))  # self.arg1: None
    # self.logger.debug(self.settings.attributes.keys())
    # self.logger.debug(self.settings.attributes.values())
    # self.logger.debug(self.settings)
    # self.logger.debug(self.settings.attributes.items())
    if getattr(self, 'url', None):
        yield scrapy.Request(self.url)

    self.logger.debug('JOB: %s' % self.settings.get('JOB'))
    self.logger.debug('USER_AGENT: %s' % self.settings.get('USER_AGENT'))  # Scrapy/1.5.0 (+https://scrapy.org)
    self.logger.debug('ROBOTSTXT_OBEY: %s' % self.settings.getbool('ROBOTSTXT_OBEY'))  # True
    self.logger.debug('COOKIES_ENABLED: %s' % self.settings.getbool('COOKIES_ENABLED'))  # True
    self.logger.debug('CONCURRENT_REQUESTS: %s' % self.settings.getint('CONCURRENT_REQUESTS'))  # 16
    self.logger.debug('DOWNLOAD_DELAY: %s' % self.settings.getint('DOWNLOAD_DELAY'))  # 0
    self.logger.debug('CLOSESPIDER_TIMEOUT: %s' % self.settings.getint('CLOSESPIDER_TIMEOUT'))  # 0
    self.logger.debug('CLOSESPIDER_PAGECOUNT: %s' % self.settings.getint('CLOSESPIDER_PAGECOUNT'))  # 0

    self.log(u'Chinese characters: 汉字字符')
    self.logger.debug('2018-08-20 09:13:06 [apps_redis] DEBUG: Resuming crawl (675840 requests scheduled)')

    yield scrapy.Request('http://httpbin.org/redirect/1')
    yield scrapy.Request('http://httpbin.org/status/404')
    yield scrapy.Request('http://httpbin.org/headers')
    yield scrapy.Request('http://httpbin.org/headers')
    yield scrapy.Request('https://google.com/')

def parse(self, response):
    self.log(response.text)
    if self.num == 1:
        yield scrapy.Request('https://www.baidu.com/')

    yield {u'Chinese 汉字 %s' % self.num: ''.join('0' + str(i) if i < 10 else str(i) for i in range(1, 100))}

    self.num += 1
''',
'''
def from_crawler(cls, crawler):
    # This method is used by Scrapy to create your spiders.
    s = cls()
    scrapy.signals.connect(s.spider_opened, signal=signals.spider_opened)
    return s

def process_spider_input(self, response, spider):
    # Called for each response that goes through the spider
    # middleware and into the spider.

    # Should return None or raise an exception.
    return None

def process_spider_output(self, response, result, spider):
    # Called with the results returned from the Spider, after
    # it has processed the response.

    # Must return an iterable of Request, dict or Item objects.
    for i in result:
        yield i

def process_spider_exception(self, response, exception, spider):
    # Called when a spider or process_spider_input() method
    # (from other spider middleware) raises an exception.

    # Should return either None or an iterable of Response, dict
    # or Item objects.
    pass

def process_start_requests(self, start_requests, spider):
    # Called with the start requests of the spider, and works
    # similarly to the process_spider_output() method, except
    # that it doesn’t have a response associated.

    # Must return only requests (not items).
    for r in start_requests:
        yield r

def spider_opened(self, spider):
    spider.logger.info('Spider opened: %s' % spider.name)
''',

'''
def from_crawler(cls, crawler):
    # This method is used by Scrapy to create your spiders.
    s = cls()
    scrapy.signals.connect(s.spider_opened, signal=signals.spider_opened)
    return s

def process_request(self, request, spider):
    # Called for each request that goes through the downloader
    # middleware.

    # Must either:
    # - return None: continue processing this request
    # - or return a Response object
    # - or return a Request object
    # - or raise IgnoreRequest: process_exception() methods of
    #   installed downloader middleware will be called
    return None

def process_response(self, request, response, spider):
    # Called with the response returned from the downloader.

    # Must either;
    # - return a Response object
    # - return a Request object
    # - or raise IgnoreRequest
    return response

def process_exception(self, request, exception, spider):
    # Called when a download handler or a process_request()
    # (from other downloader middleware) raises an exception.

    # Must either:
    # - return None: continue processing this exception
    # - return a Response object: stops process_exception() chain
    # - return a Request object: stops process_exception() chain
    pass

def spider_opened(self, spider):
    spider.logger.info('Spider opened: %s' % spider.name)
''',
'''
def from_crawler(cls, crawler):
    # This method is used by Scrapy to create your spiders.
    s = cls()
    scrapy.signals.connect(s.spider_opened, signal=signals.spider_opened)
    return s

def process_spider_input(self, response, spider):
    # Called for each response that goes through the spider
    # middleware and into the spider.

    # Should return None or raise an exception.
    return None

def process_spider_output(self, response, result, spider):
    # Called with the results returned from the Spider, after
    # it has processed the response.

    # Must return an iterable of Request, dict or Item objects.
    for i in result:
        yield i

def process_spider_exception(self, response, exception, spider):
    # Called when a spider or process_spider_input() method
    # (from other spider middleware) raises an exception.

    # Should return either None or an iterable of Response, dict
    # or Item objects.
    pass

def process_start_requests(self, start_requests, spider):
    # Called with the start requests of the spider, and works
    # similarly to the process_spider_output() method, except
    # that it doesn’t have a response associated.

    # Must return only requests (not items).
    for r in start_requests:
        yield r

def spider_opened(self, spider):
    spider.logger.info('Spider opened: %s' % spider.name)
''',

'''
def from_crawler(cls, crawler):
    # This method is used by Scrapy to create your spiders.
    s = cls()
    scrapy.signals.connect(s.spider_opened, signal=signals.spider_opened)
    return s

def process_request(self, request, spider):
    # Called for each request that goes through the downloader
    # middleware.

    # Must either:
    # - return None: continue processing this request
    # - or return a Response object
    # - or return a Request object
    # - or raise IgnoreRequest: process_exception() methods of
    #   installed downloader middleware will be called
    return None

def process_response(self, request, response, spider):
    # Called with the response returned from the downloader.

    # Must either;
    # - return a Response object
    # - return a Request object
    # - or raise IgnoreRequest
    return response

def process_exception(self, request, exception, spider):
    # Called when a download handler or a process_request()
    # (from other downloader middleware) raises an exception.

    # Must either:
    # - return None: continue processing this exception
    # - return a Response object: stops process_exception() chain
    # - return a Request object: stops process_exception() chain
    pass

def spider_opened(self, spider):
    spider.logger.info('Spider opened: %s' % spider.name)
'''
]