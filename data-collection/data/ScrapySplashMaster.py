file_string = [

'''
def __init__(self, debug=False):
        self.jars = defaultdict(CookieJar)
        self.debug = debug

@classmethod
def from_crawler(cls, crawler):
    return cls(debug=crawler.settings.getbool('SPLASH_COOKIES_DEBUG'))

def process_request(self, request, spider):
    """
    For Splash requests add 'cookies' key with current
    cookies to ``request.meta['splash']['args']`` and remove cookie
    headers sent to Splash itself.
    """
    if 'splash' not in request.meta:
        return

    if request.meta.get('_splash_processed'):
        request.headers.pop('Cookie', None)
        return

    splash_options = request.meta['splash']

    splash_args = splash_options.setdefault('args', {})
    if 'cookies' in splash_args:  # cookies already set
        return

    if 'session_id' not in splash_options:
        return

    jar = self.jars[splash_options['session_id']]

    cookies = self._get_request_cookies(request)
    har_to_jar(jar, cookies)

    splash_args['cookies'] = jar_to_har(jar)
    self._debug_cookie(request, spider)

def process_response(self, request, response, spider):
    """
    For Splash JSON responses add all cookies from
    'cookies' in a response to the cookiejar.
    """
    from scrapy_splash import SplashJsonResponse
    if not isinstance(response, SplashJsonResponse):
        return response

    if 'cookies' not in response.data:
        return response

    if 'splash' not in request.meta:
        return response

    if not request.meta.get('_splash_processed'):
        warnings.warn("SplashCookiesMiddleware requires SplashMiddleware")
        return response

    splash_options = request.meta['splash']
    session_id = splash_options.get('new_session_id',
                                    splash_options.get('session_id'))
    if session_id is None:
        return response

    jar = self.jars[session_id]
    request_cookies = splash_options['args'].get('cookies', [])
    har_to_jar(jar, response.data['cookies'], request_cookies)
    self._debug_set_cookie(response, spider)
    response.cookiejar = jar
    return response

def _get_request_cookies(self, request):
    if isinstance(request.cookies, dict):
        return [
            {'name': k, 'value': v} for k, v in request.cookies.items()
        ]
    return request.cookies or []

def _debug_cookie(self, request, spider):
    if self.debug:
        cl = request.meta['splash']['args']['cookies']
        if cl:
            logger.debug(msg, extra={'spider': spider})

def _debug_set_cookie(self, response, spider):
    if self.debug:
        cl = response.data['cookies']
        if cl:
            logger.debug(msg, extra={'spider': spider})

@staticmethod
def _har_repr(har_cookie):
    return '{}={}'.format(har_cookie['name'], har_cookie['value'])

''',

'''
def process_spider_output(self, response, result, spider):
        for el in result:
            if isinstance(el, scrapy.Request):
                yield self._process_request(el, spider)
            else:
                yield el

def process_start_requests(self, start_requests, spider):
    if not hasattr(spider, 'state'):
        spider.state = {}
    spider.state.setdefault(self.local_values_key, {})  # fingerprint => value dict

    for req in start_requests:
        yield self._process_request(req, spider)

def _process_request(self, request, spider):
    """
    Replace requested meta['splash']['args'] values with their fingerprints.
    This allows to store values only once in request queue, which helps
    with disk queue size.

    Downloader middleware should restore the values from fingerprints.
    """
    if 'splash' not in request.meta:
        return request

    if '_replaced_args' in request.meta['splash']:
        # don't process re-scheduled requests
        # XXX: does it work as expected?
        warnings.warn("Unexpected request.meta['splash']['_replaced_args']")
        return request

    request.meta['splash']['_replaced_args'] = []
    cache_args = request.meta['splash'].get('cache_args', [])
    args = request.meta['splash'].setdefault('args', {})

    for name in cache_args:
        if name not in args:
            continue
        value = args[name]
        fp = 'LOCAL+' + json_based_hash(value)
        spider.state[self.local_values_key][fp] = value
        args[name] = fp
        request.meta['splash']['_replaced_args'].append(name)

    return request
''',

'''
def __init__(self, crawler, splash_base_url, slot_policy, log_400, auth):
        self.crawler = crawler
        self.splash_base_url = splash_base_url
        self.slot_policy = slot_policy
        self.log_400 = log_400
        self.crawler.signals.connect(self.spider_opened, signals.spider_opened)
        self.auth = auth

@classmethod
def from_crawler(cls, crawler):
    s = crawler.settings
    splash_base_url = s.get('SPLASH_URL', cls.default_splash_url)
    log_400 = s.getbool('SPLASH_LOG_400', True)
    slot_policy = s.get('SPLASH_SLOT_POLICY', cls.default_policy)
    if slot_policy not in SlotPolicy._known:
        raise scrapy.exceptions.NotConfigured("Incorrect slot policy: %r" % slot_policy)

    splash_user = s.get('SPLASH_USER', '')
    splash_pass = s.get('SPLASH_PASS', '')
    auth = None
    if splash_user or splash_pass:
        auth = basic_auth_header(splash_user, splash_pass)
    return cls(crawler, splash_base_url, slot_policy, log_400, auth)

def spider_opened(self, spider):
    if _http_auth_enabled(spider):
        replace_downloader_middleware(self.crawler, RobotsTxtMiddleware,
                                        SafeRobotsTxtMiddleware)
    if not hasattr(spider, 'state'):
        spider.state = {}

    # local fingerprint => key returned by splash
    spider.state.setdefault(self.remote_keys_key, {})

@property
def _argument_values(self):
    key = SplashDeduplicateArgsMiddleware.local_values_key
    return self.crawler.spider.state[key]

@property
def _remote_keys(self):
    return self.crawler.spider.state[self.remote_keys_key]

def process_request(self, request, spider):
    if 'splash' not in request.meta:
        return
    splash_options = request.meta['splash']

    if request.method not in {'GET', 'POST'}:
        logger.error(
            "Currently only GET and POST requests are supported by "
            "SplashMiddleware; %(request)s is dropped",
            {'request': request},
            extra={'spider': spider}
        )
        self.crawler.stats.inc_value('splash/dropped/method/{}'.format(
            request.method))
        raise scrapy.exceptions.IgnoreRequest("SplashRequest doesn't support "
                            "HTTP {} method".format(request.method))

    if request.meta.get("_splash_processed"):
        # don't process the same request more than once
        return

    request.meta['_splash_processed'] = True

    slot_policy = splash_options.get('slot_policy', self.slot_policy)
    self._set_download_slot(request, request.meta, slot_policy)

    args = splash_options.setdefault('args', {})

    if '_replaced_args' in splash_options:
        # restore arguments before sending request to the downloader
        load_args = {}
        save_args = []
        local_arg_fingerprints = {}
        for name in splash_options['_replaced_args']:
            fp = args[name]
            # Use remote Splash argument cache: if Splash key
            # for a value is known then don't send the value to Splash;
            # if it is unknown then try to save the value on server using
            # ``save_args``.
            if fp in self._remote_keys:
                load_args[name] = self._remote_keys[fp]
                del args[name]
            else:
                save_args.append(name)
                args[name] = self._argument_values[fp]

            local_arg_fingerprints[name] = fp

        if load_args:
            args['load_args'] = load_args
        if save_args:
            args['save_args'] = save_args
        splash_options['_local_arg_fingerprints'] = local_arg_fingerprints

        del splash_options['_replaced_args']  # ??

    args.setdefault('url', request.url)
    if request.method == 'POST':
        args.setdefault('http_method', request.method)
        # XXX: non-UTF8 request bodies are not supported now
        args.setdefault('body', request.body.decode('utf8'))

    if not splash_options.get('dont_send_headers'):
        headers = scrapy_headers_to_unicode_dict(request.headers)
        if headers:
            # Headers set by HttpAuthMiddleware should be used for Splash,
            # not for the remote website (backwards compatibility).
            if _http_auth_enabled(spider):
                headers.pop('Authorization', None)
            args.setdefault('headers', headers)

    body = json.dumps(args, ensure_ascii=False, sort_keys=True, indent=4)
    # print(body)

    if 'timeout' in args:
        # User requested a Splash timeout explicitly.
        #
        # We can't catch a case when user requested `download_timeout`
        # explicitly because a default value for `download_timeout`
        # is set by DownloadTimeoutMiddleware.
        #
        # As user requested Splash timeout explicitly, we shouldn't change
        # it. Another reason not to change the requested Splash timeout is
        # because it may cause a validation error on the remote end.
        #
        # But we can change Scrapy `download_timeout`: increase
        # it when it's too small. Decreasing `download_timeout` is not
        # safe.

        timeout_requested = float(args['timeout'])
        timeout_expected = timeout_requested + self.splash_extra_timeout

        # no timeout means infinite timeout
        timeout_current = request.meta.get('download_timeout', 1e6)

        if timeout_expected > timeout_current:
            request.meta['download_timeout'] = timeout_expected

    endpoint = splash_options.setdefault('endpoint', self.default_endpoint)
    splash_base_url = splash_options.get('splash_url', self.splash_base_url)
    splash_url = urljoin(splash_base_url, endpoint)

    headers = scrapy.http.headers({'Content-Type': 'application/json'})
    if self.auth is not None:
        headers['Authorization'] = self.auth
    headers.update(splash_options.get('splash_headers', {}))
    new_request = request.replace(
        url=splash_url,
        method='POST',
        body=body,
        headers=headers,
        priority=request.priority + self.rescheduling_priority_adjust
    )
    new_request.meta['dont_obey_robotstxt'] = True
    self.crawler.stats.inc_value('splash/%s/request_count' % endpoint)
    return new_request

def process_response(self, request, response, spider):
    if not request.meta.get("_splash_processed"):
        return response

    splash_options = request.meta['splash']
    if not splash_options:
        return response

    # update stats
    endpoint = splash_options['endpoint']
    self.crawler.stats.inc_value(
        'splash/%s/response_count/%s' % (endpoint, response.status)
    )

    # handle save_args/load_args
    self._process_x_splash_saved_arguments(request, response)
    if get_splash_status(response) == 498:
        logger.debug("Got HTTP 498 response for {}; "
                        "sending arguments again.".format(request),
                        extra={'spider': spider})
        return self._498_retry_request(request, response)

    if splash_options.get('dont_process_response', False):
        return response

    response = self._change_response_class(request, response)

    if self.log_400 and get_splash_status(response) == 400:
        self._log_400(request, response, spider)

    return response

def _change_response_class(self, request, response):
    from scrapy_splash import SplashResponse, SplashTextResponse
    if not isinstance(response, (SplashResponse, SplashTextResponse)):
        # create a custom Response subclass based on response Content-Type
        # XXX: usually request is assigned to response only when all
        # downloader middlewares are executed. Here it is set earlier.
        # Does it have any negative consequences?
        respcls = responsetypes.from_args(headers=response.headers)
        if isinstance(response, TextResponse) and respcls is SplashResponse:
            # Even if the headers say it's binary, it has already
            # been detected as a text response by scrapy (for example
            # because it was decoded successfully), so we should not
            # convert it to SplashResponse.
            respcls = SplashTextResponse
        response = response.replace(cls=respcls, request=request)
    return response

def _log_400(self, request, response, spider):
    from scrapy_splash import SplashJsonResponse
    if isinstance(response, SplashJsonResponse):
        logger.warning(
            "Bad request to Splash: %s" % response.data,
            {'request': request},
            extra={'spider': spider}
        )

def _process_x_splash_saved_arguments(self, request, response):
    """ Keep track of arguments saved by Splash. """
    saved_args = get_splash_headers(response).get(b'X-Splash-Saved-Arguments')
    if not saved_args:
        return
    saved_args = parse_x_splash_saved_arguments_header(saved_args)
    arg_fingerprints = request.meta['splash']['_local_arg_fingerprints']
    for name, key in saved_args.items():
        fp = arg_fingerprints[name]
        self._remote_keys[fp] = key

def _498_retry_request(self, request, response):
    """
    Return a retry request for HTTP 498 responses. HTTP 498 means
    load_args are not present on server; client should retry the request
    with full argument values instead of their hashes.
    """
    meta = copy.deepcopy(request.meta)
    local_arg_fingerprints = meta['splash']['_local_arg_fingerprints']
    args = meta['splash']['args']
    args.pop('load_args', None)
    args['save_args'] = list(local_arg_fingerprints.keys())

    for name, fp in local_arg_fingerprints.items():
        args[name] = self._argument_values[fp]
        # print('remote_keys before:', self._remote_keys)
        self._remote_keys.pop(fp, None)
        # print('remote_keys after:', self._remote_keys)

    body = json.dumps(args, ensure_ascii=False, sort_keys=True, indent=4)
    # print(body)
    request = request.replace(
        meta=meta,
        body=body,
        priority=request.priority+self.retry_498_priority_adjust
    )
    return request

def _set_download_slot(self, request, meta, slot_policy):
    if slot_policy == SlotPolicy.PER_DOMAIN:
        # Use the same download slot to (sort of) respect download
        # delays and concurrency options.
        meta['download_slot'] = self._get_slot_key(request)

    elif slot_policy == SlotPolicy.SINGLE_SLOT:
        # Use a single slot for all Splash requests
        meta['download_slot'] = '__splash__'

    elif slot_policy == SlotPolicy.SCRAPY_DEFAULT:
        # Use standard Scrapy concurrency setup
        pass

def _get_slot_key(self, request_or_response):
    return self.crawler.engine.downloader._get_slot_key(
        request_or_response, None
    )
''',

'''
def process_request(self, request, spider):
    # disable robots.txt for Splash requests
    if _http_auth_enabled(spider) and 'splash' in request.meta:
        return
    return super(SafeRobotsTxtMiddleware, self).process_request(
        request, spider)
''',

'''
def _http_auth_enabled(spider):
    # FIXME: this function should always return False if HttpAuthMiddleware is
    # not in a middleware list.
    return getattr(spider, 'http_user', '') or getattr(spider, 'http_pass', '')


def replace_downloader_middleware(crawler, old_cls, new_cls):
    """ Replace downloader middleware with another one """
    try:
        new_mw = new_cls.from_crawler(crawler)
    except scrapy.exceptions.NotConfigured:
        return

    mw_manager = crawler.engine.downloader.middleware
    mw_manager.middlewares = tuple([
        mw if mw.__class__ is not old_cls else new_mw
        for mw in mw_manager.middlewares
    ])
    for method_name, callbacks in mw_manager.methods.items():
        for idx, meth in enumerate(callbacks):
            method_cls = meth.__self__.__class__
            if method_cls is old_cls:
                new_meth = getattr(new_mw, method_name)
                # logger.debug("{} is replaced with {}".format(meth, new_meth))
                callbacks[idx] = new_meth
''',

'''
def __init__(self,
                 url,
                 callback=None,
                 method='GET',
                 endpoint='render.html',
                 args=None,
                 splash_url=None,
                 slot_policy=SlotPolicy.PER_DOMAIN,
                 splash_headers=None,
                 dont_process_response=False,
                 dont_send_headers=False,
                 magic_response=True,
                 session_id='default',
                 http_status_from_error_code=True,
                 cache_args=None,
                 meta=None,
                 **kwargs):

    url = to_unicode(url)

    meta = copy.deepcopy(meta) or {}
    splash_meta = meta.setdefault('splash', {})
    splash_meta.setdefault('endpoint', endpoint)
    splash_meta.setdefault('slot_policy', slot_policy)
    if splash_url is not None:
        splash_meta['splash_url'] = splash_url
    if splash_headers is not None:
        splash_meta['splash_headers'] = splash_headers
    if dont_process_response:
        splash_meta['dont_process_response'] = True
    else:
        splash_meta.setdefault('magic_response', magic_response)
    if dont_send_headers:
        splash_meta['dont_send_headers'] = True
    if http_status_from_error_code:
        splash_meta['http_status_from_error_code'] = True
    if cache_args is not None:
        splash_meta['cache_args'] = cache_args

    if session_id is not None:
        if splash_meta['endpoint'].strip('/') == 'execute':
            splash_meta.setdefault('session_id', session_id)

    _args = {'url': url}  # put URL to args in order to preserve #fragment
    _args.update(args or {})
    _args.update(splash_meta.get('args', {}))
    splash_meta['args'] = _args

    # This is not strictly required, but it strengthens Splash
    # requests against AjaxCrawlMiddleware
    meta['ajax_crawlable'] = True

    super(SplashRequest, self).__init__(url, callback, method, meta=meta,
                                        **kwargs)

@property
def _processed(self):
    return self.meta.get('_splash_processed')

@property
def _splash_args(self):
    return self.meta.get('splash', {}).get('args', {})

@property
def _original_url(self):
    return self._splash_args.get('url')

@property
def _original_method(self):
    return self._splash_args.get('http_method', 'GET')

def __str__(self):
    if not self._processed:
        return super(SplashRequest, self).__str__()
    return "<%s %s via %s>" % (self._original_method, self._original_url, self.url)

__repr__ = __str__
''',

'''
def __init__(self, url=None, callback=None, method=None, formdata=None,
                 body=None, **kwargs):
        # First init FormRequest to get url, body and method
        if formdata:
            scrapy.http.FormRequest.__init__(
                self, url=url, method=method, formdata=formdata)
            url, method, body = self.url, self.method, self.body
        # Then pass all other kwargs to SplashRequest
        SplashRequest.__init__(
            self, url=url, callback=callback, method=method, body=body,
            **kwargs)
''',

'''

def dict_hash(obj, start=''):
    """ Return a hash for a dict, based on its contents """
    h = hashlib.sha1(to_bytes(start))
    h.update(to_bytes(obj.__class__.__name__))
    if isinstance(obj, dict):
        for key, value in sorted(obj.items()):
            h.update(to_bytes(key))
            h.update(to_bytes(dict_hash(value)))
    elif isinstance(obj, (list, tuple)):
        for el in obj:
            h.update(to_bytes(dict_hash(el)))
    else:
        # basic types
        if isinstance(obj, bool):
            value = str(int(obj))
        elif isinstance(obj, (six.integer_types, float)):
            value = str(obj)
        elif isinstance(obj, (six.text_type, bytes)):
            value = obj
        elif obj is None:
            value = b''
        else:
            raise ValueError("Unsupported value type: %s" % obj.__class__)
        h.update(to_bytes(value))
    return h.hexdigest()


def _process(value, sha=False):
    if isinstance(value, (six.text_type, bytes)):
        if sha:
            return hashlib.sha1(to_bytes(value)).hexdigest()
        return 'h', hash(value)
    if isinstance(value, dict):
        return {_process(k, sha=True): _process(v, sha) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_process(v, sha) for v in value]
    return value


def _fast_hash(value):
    """
    Return a hash for any JSON-serializable value.
    Hash is not guaranteed to be the same in different Python processes,
    but it is very fast to compute for data structures with large string
    values.
    """
    return _json_based_hash(_process(value))


_hash_cache = {}  # fast hash => hash
def json_based_hash(value):
    """
    Return a hash for any JSON-serializable value.

    >>> json_based_hash({"foo": "bar", "baz": [1, 2]})
    '0570066939bea46c610bfdc35b20f37ef09d05ed'
    """
    fp = _fast_hash(value)
    if fp not in _hash_cache:
        _hash_cache[fp] = _json_based_hash(_process(value, sha=True))
    return _hash_cache[fp]


def _json_based_hash(value):
    v = json.dumps(value, sort_keys=True, ensure_ascii=False).encode('utf8')
    return hashlib.sha1(v).hexdigest()


def headers_to_scrapy(headers):
    """
    Return scrapy.http.Headers instance from headers data.
    3 data formats are supported:

    * {name: value, ...} dict;
    * [(name, value), ...] list;
    * [{'name': name, 'value': value'}, ...] list (HAR headers format).
    """
    if isinstance(headers or {}, dict):
        return scrapy.http.Headers(headers or {})

    if isinstance(headers[0], dict):
        return scrapy.http.Headers([
            (d['name'], d.get('value', ''))
            for d in headers
        ])

    return scrapy.http.Headers(headers)


def scrapy_headers_to_unicode_dict(headers):
    """
    Convert scrapy.http.Headers instance to a dictionary
    suitable for JSON encoding.
    """
    return {
        to_unicode(key): to_unicode(b','.join(value))
        for key, value in headers.items()
    }


def parse_x_splash_saved_arguments_header(value):
    """
    Parse X-Splash-Saved-Arguments header value.

    >>> value = u"name1=9a6747fc6259aa374ab4e1bb03074b6ec672cf99;name2=ba001160ef96fe2a3f938fea9e6762e204a562b3"
    >>> dct = parse_x_splash_saved_arguments_header(value)
    >>> sorted(list(dct.keys()))
    ['name1', 'name2']
    >>> dct['name1']
    '9a6747fc6259aa374ab4e1bb03074b6ec672cf99'
    >>> dct['name2']
    'ba001160ef96fe2a3f938fea9e6762e204a562b3'

    Binary header values are also supported:
    >>> dct2 = parse_x_splash_saved_arguments_header(value.encode('utf8'))
    >>> dct2 == dct
    True
    """
    value = to_unicode(value)
    return dict(kv.split('=', 1) for kv in  value.split(";"))
''',

'''
def test_dict_hash():
    h1 = dict_hash({"foo": "bar", "bar": "baz"})
    h2 = dict_hash({"foo": "bar", "bar": "baz"})
    assert h1 == h2

    h3 = dict_hash({"egg": "spam"})
    assert h3 != h2


def test_dict_hash_nested():
    h1 = dict_hash({"foo": "bar", "bar": {"baz": "spam"}})
    h2 = dict_hash({"foo": "bar", "bar": {"baz": "spam"}})
    assert h1 == h2

    h3 = dict_hash({"foo": "bar", "bar": {"baz": "egg"}})
    h4 = dict_hash({"foo": "bar", "bar": {"bam": "spam"}})
    assert h3 != h2
    assert h4 != h2


def test_dict_hash_non_strings():
    h1 = dict_hash({"foo": "bar", "float": 1.1, "int": 2, "bool": False,
                    "seq": ["x", "y", (2, 3.7, {"x": 5, "y": [6, 7]})]})
    h2 = dict_hash({"foo": "bar", "float": 1.2, "int": 2, "bool": False})
    assert h1 != h2


def test_dict_hash_invalid():
    with pytest.raises(ValueError):
        dict_hash({"foo": scrapy})


def test_request_fingerprint_nosplash():
    r1 = scrapy.Request("http://example.com")
    r2 = scrapy.Request("http://example.com", meta={"foo": "bar"})
    assert scrapy.dupefilters.request_fingerprint(r1) == splash_request_fingerprint(r1)
    assert scrapy.dupefilters.request_fingerprint(r1) == scrapy.dupefilters.request_fingerprint(r2)
    assert scrapy.dupefilters.request_fingerprint(r1) == splash_request_fingerprint(r2)


def assert_fingerprints_match(r1, r2):
    assert splash_request_fingerprint(r1) == splash_request_fingerprint(r2)


def assert_fingerprints_dont_match(r1, r2):
    assert splash_request_fingerprint(r1) != splash_request_fingerprint(r2)


def test_request_fingerprint_splash():
    r1 = scrapy.Request("http://example.com")
    r2 = scrapy.Request("http://example.com", meta={"splash": {"args": {"html": 1}}})
    r3 = scrapy.Request("http://example.com", meta={"splash": {"args": {"png": 1}}})
    r4 = scrapy.Request("http://example.com", meta={"foo": "bar", "splash": {"args": {"html": 1}}})
    r5 = scrapy.Request("http://example.com", meta={"splash": {"args": {"html": 1, "wait": 1.0}}})

    assert request_fingerprint(r1) == scrapy.dupefilters.request_fingerprint(r2)
    assert_fingerprints_dont_match(r1, r2)
    assert_fingerprints_dont_match(r1, r3)
    assert_fingerprints_dont_match(r1, r4)
    assert_fingerprints_dont_match(r1, r5)
    assert_fingerprints_dont_match(r2, r3)

    # only "splash" contents is taken into account
    assert_fingerprints_match(r2, r4)


@pytest.fixture()
def splash_middleware():
    return _get_mw()


@pytest.fixture
def splash_mw_process(splash_middleware):
    def _process(r):
        r_copy = r.replace(meta=deepcopy(r.meta))
        return splash_middleware.process_request(r_copy, None) or r
    return _process


@pytest.fixture()
def requests():
    url1 = "http://example.com/foo?x=1&y=2"
    url2 = "http://example.com/foo?y=2&x=1"
    url3 = "http://example.com/foo?x=1&y=2&z=3"
    url4 = "http://example.com/foo?x=1&y=2#id2"
    url5 = "http://example.com/foo?x=1&y=2#!id2"
    request_kwargs = [
        dict(url=url1),                         # 0
        dict(url=url1, method='POST'),          # 1
        dict(url=url1, endpoint='render.har'),  # 2
        dict(url=url2),                         # 3
        dict(url=url1, args={'wait': 0.5}),     # 4
        dict(url=url2, args={'wait': 0.5}),     # 5
        dict(url=url3),                         # 6
        dict(url=url2, method='POST'),          # 7
        dict(url=url3, args={'wait': 0.5}),     # 8
        dict(url=url3, args={'wait': 0.5}),     # 9
        dict(url=url3, args={'wait': 0.7}),     # 10
        dict(url=url4),                         # 11
    ]
    splash_requests = [SplashRequest(**kwargs) for kwargs in request_kwargs]
    scrapy_requests = [
        scrapy.Request(url=url1),               # 12
        scrapy.Request(url=url2),               # 13
        scrapy.Request(url=url4),               # 14
        scrapy.Request(url=url5),               # 15
    ]
    return splash_requests + scrapy_requests


@pytest.mark.parametrize(["i", "dupe_indices"], [
    (0, {3}),
    (1, {7}),
    (2, set()),
    (3, {0}),
    (4, {5}),
    (5, {4}),
    (6, set()),
    (7, {1}),
    (8, {9}),
    (9, {8}),
    (10, set()),
    (11, set()),
    (12, {13, 14}),
    (13, {12, 14}),
    (14, {13, 12}),
    (15, set()),
])
def test_duplicates(i, dupe_indices, requests, splash_mw_process):
    def assert_not_filtered(r1, r2):
        assert_fingerprints_dont_match(r1, r2)
        assert_fingerprints_dont_match(
            splash_mw_process(r1),
            splash_mw_process(r2),
        )

    def assert_filtered(r1, r2):
        # request is filtered if it is filtered either
        # before rescheduling or after
        fp1 = splash_request_fingerprint(r1)
        fp2 = splash_request_fingerprint(r2)
        if fp1 != fp2:
            assert_fingerprints_match(
                splash_mw_process(r1),
                splash_mw_process(r2),
            )

    dupe_indices = set(dupe_indices)
    dupe_indices.add(i)
    non_dupe_indices = set(range(len(requests))) - dupe_indices

    for j in dupe_indices:
        assert_filtered(requests[i], requests[j])
    for j in non_dupe_indices:
        assert_not_filtered(requests[i], requests[j])

''',

'''
def start_requests(self):
    yield SplashRequest(self.url)

def parse(self, response):
    yield {'response': response}
''',

'''
def start_requests(self):
    yield SplashRequest(self.url,
                        endpoint='execute',
                        args={'lua_source': DEFAULT_SCRIPT},
                        headers=self.headers,
                        splash_headers=self.splash_headers)
''',

'''
""" Spider with incorrect (old, insecure) auth method """
http_user = 'user'
http_pass = 'userpass'
http_auth_domain = None
''',

'''
""" Spider which uses HTTP auth and doesn't use Splash """
http_user = 'user'
http_pass = 'userpass'
http_auth_domain = None

def start_requests(self):
    yield scrapy.Request(self.url)
''',

'''

def assert_single_response(items):
    assert len(items) == 1
    return items[0]['response']


@requires_splash
@inlineCallbacks
def test_basic(settings):
    items, url, crawler = yield crawl_items(ResponseSpider, HelloWorld,
                                            settings)
    resp = assert_single_response(items)
    assert resp.url == url
    assert resp.css('body::text').extract_first().strip() == "hello world!"


@requires_splash
@inlineCallbacks
def test_reload(settings):

    class ReloadSpider(ResponseSpider):
        """ Make two requests to URL, store both responses.
        This spider activates both start_requests and parse methods,
        and checks that dupefilter takes fragment into account. """

        def parse(self, response):
            yield {'response': response}
            yield SplashRequest(self.url + '#foo')

    items, url, crawler = yield crawl_items(ReloadSpider, HelloWorld, settings)
    assert len(items) == 2
    assert crawler.stats.get_value('dupefilter/filtered') == 1
    resp = items[0]['response']
    assert resp.url == url
    assert resp.css('body::text').extract_first().strip() == "hello world!"
    assert resp.status == resp.splash_response_status == 200
    assert resp.headers == resp.splash_response_headers
    assert resp.splash_response_headers['Content-Type'] == b"text/html; charset=utf-8"

    resp2 = items[1]['response']
    assert resp2.body == resp.body
    assert resp2 is not resp
    assert resp2.url == resp.url + "#foo"


@requires_splash
@inlineCallbacks
def test_basic_lua(settings):

    class LuaScriptSpider(ResponseSpider):
        """ Make a request using a Lua script similar to the one from README
        """
        def start_requests(self):
            yield SplashRequest(self.url + "#foo", endpoint='execute',
                            args={'lua_source': DEFAULT_SCRIPT, 'foo': 'bar'})


    items, url, crawler = yield crawl_items(LuaScriptSpider, HelloWorld,
                                            settings)
    resp = assert_single_response(items)
    assert resp.url == url + "/#foo"
    assert resp.status == resp.splash_response_status == 200
    assert resp.css('body::text').extract_first().strip() == "hello world!"
    assert resp.data['jsvalue'] == 3
    assert resp.headers['X-MyHeader'] == b'my value'
    assert resp.headers['Content-Type'] == b'text/html'
    assert resp.splash_response_headers['Content-Type'] == b'application/json'
    assert resp.data['args']['foo'] == 'bar'


@requires_splash
@inlineCallbacks
def test_bad_request(settings):
    class BadRequestSpider(ResponseSpider):
        def start_requests(self):
            yield SplashRequest(self.url, endpoint='execute',
                                args={'lua_source': DEFAULT_SCRIPT, 'wait': 'bar'})

    items, url, crawler = yield crawl_items(BadRequestSpider, HelloWorld,
                                            settings)
    resp = assert_single_response(items)
    assert resp.status == 400
    assert resp.splash_response_status == 400

    items, url, crawler = yield crawl_items(LuaSpider, Http400Resource,
                                            settings)
    resp = assert_single_response(items)
    assert resp.status == 400
    assert resp.splash_response_status == 200


@requires_splash
@inlineCallbacks
def test_cache_args(settings):

    class CacheArgsSpider(ResponseSpider):
        def _request(self, url):
            return SplashRequest(url, endpoint='execute',
                                 args={'lua_source': DEFAULT_SCRIPT, 'x': 'yy'},
                                 cache_args=['lua_source'])

        def start_requests(self):
            yield self._request(self.url)

        def parse(self, response):
            yield {'response': response}
            yield self._request(self.url + "#foo")


    items, url, crawler = yield crawl_items(CacheArgsSpider, HelloWorld,
                                            settings)
    assert len(items) == 2
    resp = items[0]['response']
    assert b"function main(splash)" in resp.request.body
    assert b"yy" in resp.request.body
    print(resp.body, resp.request.body)

    resp = items[1]['response']
    assert b"function main(splash)" not in resp.request.body
    assert b"yy" in resp.request.body
    print(resp.body, resp.request.body)


@requires_splash
@inlineCallbacks
def test_cookies(settings):

    # 64K for headers is over Twisted limit,
    # so if these headers are sent to Splash request would fail.
    BOMB = 'x' * 64000

    class LuaScriptSpider(ResponseSpider):
        """ Cookies must be sent to website, not to Splash """
        custom_settings = {
            'SPLASH_COOKIES_DEBUG': True,
            'COOKIES_DEBUG': True,
        }

        def start_requests(self):
            # cookies set without Splash should be still
            # sent to a remote website. FIXME: this is not the case.
            yield scrapy.Request(self.url + "/login", self.parse,
                                 cookies={'x-set-scrapy': '1'})

        def parse(self, response):
            yield SplashRequest(self.url + "#egg", self.parse_1,
                                endpoint='execute',
                                args={'lua_source': DEFAULT_SCRIPT},
                                cookies={'x-set-splash': '1'})

        def parse_1(self, response):
            yield {'response': response}
            yield SplashRequest(self.url + "#foo", self.parse_2,
                                endpoint='execute',
                                args={'lua_source': DEFAULT_SCRIPT})

        def parse_2(self, response):
            yield {'response': response}
            yield scrapy.Request(self.url, self.parse_3)

        def parse_3(self, response):
            # Splash (Twisted) drops requests with huge http headers,
            # but this one should work, as cookies are not sent
            # to Splash itself.
            yield {'response': response}
            yield SplashRequest(self.url + "#bar", self.parse_4,
                                endpoint='execute',
                                args={'lua_source': DEFAULT_SCRIPT},
                                cookies={'bomb': BOMB})

        def parse_4(self, response):
            yield {'response': response}


    def _cookie_dict(har_cookies):
        return {c['name']: c['value'] for c in har_cookies}

    items, url, crawler = yield crawl_items(LuaScriptSpider, ManyCookies,
                                            settings)
    assert len(items) == 4

    # cookie should be sent to remote website, not to Splash
    resp = items[0]['response']
    splash_request_headers = resp.request.headers
    cookies = resp.data['args']['cookies']
    print(splash_request_headers)
    print(cookies)
    assert _cookie_dict(cookies) == {
        # 'login': '1',   # FIXME
        'x-set-splash': '1'
    }
    assert splash_request_headers.get(b'Cookie') is None

    # new cookie should be also sent to remote website, not to Splash
    resp2 = items[1]['response']
    splash_request_headers = resp2.request.headers
    headers = resp2.data['args']['headers']
    cookies = resp2.data['args']['cookies']
    assert canonicalize_url(headers['Referer']) == canonicalize_url(url)
    assert _cookie_dict(cookies) == {
        # 'login': '1',
        'x-set-splash': '1',
        'sessionid': 'ABCD'
    }
    print(splash_request_headers)
    print(headers)
    print(cookies)
    assert splash_request_headers.get(b'Cookie') is None

    # TODO/FIXME: Cookies fetched when working with Splash should be picked up
    # by Scrapy
    resp3 = items[2]['response']
    splash_request_headers = resp3.request.headers
    cookie_header = splash_request_headers.get(b'Cookie')
    assert b'x-set-scrapy=1' in cookie_header
    assert b'login=1' in cookie_header
    assert b'x-set-splash=1' in cookie_header
    # assert b'sessionid=ABCD' in cookie_header  # FIXME

    # cookie bomb shouldn't cause problems
    resp4 = items[3]['response']
    splash_request_headers = resp4.request.headers
    cookies = resp4.data['args']['cookies']
    assert _cookie_dict(cookies) == {
        # 'login': '1',
        'x-set-splash': '1',
        'sessionid': 'ABCD',
        'bomb': BOMB,
    }
    assert splash_request_headers.get(b'Cookie') is None


@requires_splash
@inlineCallbacks
def test_access_http_auth(settings):
    # website is protected
    items, url, crawler = yield crawl_items(LuaSpider, HelloWorldProtected,
                                            settings)
    response = assert_single_response(items)
    assert response.status == 401
    assert response.splash_response_status == 200

    # header can be used to access it
    AUTH_HEADERS = {'Authorization': basic_auth_header('user', 'userpass')}
    kwargs = {'headers': AUTH_HEADERS}
    items, url, crawler = yield crawl_items(LuaSpider, HelloWorldProtected,
                                            settings, kwargs)
    response = assert_single_response(items)
    assert 'hello' in response.body_as_unicode()
    assert response.status == 200
    assert response.splash_response_status == 200


@requires_splash
@inlineCallbacks
def test_protected_splash_no_auth(settings_auth):
    items, url, crawler = yield crawl_items(LuaSpider, HelloWorld,
                                            settings_auth)
    response = assert_single_response(items)
    assert 'Unauthorized' in response.body_as_unicode()
    assert 'hello' not in response.body_as_unicode()
    assert response.status == 401
    assert response.splash_response_status == 401


@requires_splash
@inlineCallbacks
def test_protected_splash_manual_headers_auth(settings_auth):
    AUTH_HEADERS = {'Authorization': basic_auth_header('user', 'userpass')}
    kwargs = {'splash_headers': AUTH_HEADERS}

    # auth via splash_headers should work
    items, url, crawler = yield crawl_items(LuaSpider, HelloWorld,
                                            settings_auth, kwargs)
    response = assert_single_response(items)
    assert 'hello' in response.body_as_unicode()
    assert response.status == 200
    assert response.splash_response_status == 200

    # but only for Splash, not for a remote website
    items, url, crawler = yield crawl_items(LuaSpider, HelloWorldProtected,
                                            settings_auth, kwargs)
    response = assert_single_response(items)
    assert 'hello' not in response.body_as_unicode()
    assert response.status == 401
    assert response.splash_response_status == 200


@requires_splash
@inlineCallbacks
def test_protected_splash_settings_auth(settings_auth):
    settings_auth['SPLASH_USER'] = 'user'
    settings_auth['SPLASH_PASS'] = 'userpass'

    # settings works
    items, url, crawler = yield crawl_items(LuaSpider, HelloWorld,
                                            settings_auth)
    response = assert_single_response(items)
    assert 'Unauthorized' not in response.body_as_unicode()
    assert 'hello' in response.body_as_unicode()
    assert response.status == 200
    assert response.splash_response_status == 200

    # they can be overridden via splash_headers
    bad_auth = {'splash_headers': {'Authorization': 'foo'}}
    items, url, crawler = yield crawl_items(LuaSpider, HelloWorld,
                                            settings_auth, bad_auth)
    response = assert_single_response(items)
    assert response.status == 401
    assert response.splash_response_status == 401

    # auth error on remote website
    items, url, crawler = yield crawl_items(LuaSpider, HelloWorldProtected,
                                            settings_auth)
    response = assert_single_response(items)
    assert response.status == 401
    assert response.splash_response_status == 200

    # auth both for Splash and for the remote website
    REMOTE_AUTH = {'Authorization': basic_auth_header('user', 'userpass')}
    remote_auth_kwargs = {'headers': REMOTE_AUTH}
    items, url, crawler = yield crawl_items(LuaSpider, HelloWorldProtected,
                                            settings_auth, remote_auth_kwargs)
    response = assert_single_response(items)
    assert response.status == 200
    assert response.splash_response_status == 200
    assert 'hello' in response.body_as_unicode()

    # enable remote auth, but not splash auth - request should fail
    del settings_auth['SPLASH_USER']
    del settings_auth['SPLASH_PASS']
    items, url, crawler = yield crawl_items(LuaSpider,
                                            HelloWorldProtected,
                                            settings_auth, remote_auth_kwargs)
    response = assert_single_response(items)
    assert response.status == 401
    assert response.splash_response_status == 401


@requires_splash
@inlineCallbacks
def test_protected_splash_httpauth_middleware(settings_auth):
    # httpauth middleware should enable auth for Splash, for backwards
    # compatibility reasons
    items, url, crawler = yield crawl_items(ScrapyAuthSpider, HelloWorld,
                                            settings_auth)
    response = assert_single_response(items)
    assert 'Unauthorized' not in response.body_as_unicode()
    assert 'hello' in response.body_as_unicode()
    assert response.status == 200
    assert response.splash_response_status == 200

    # but not for a remote website
    items, url, crawler = yield crawl_items(ScrapyAuthSpider,
                                            HelloWorldProtected,
                                            settings_auth)
    response = assert_single_response(items)
    assert 'hello' not in response.body_as_unicode()
    assert response.status == 401
    assert response.splash_response_status == 200

    # headers shouldn't be sent to robots.txt file
    items, url, crawler = yield crawl_items(ScrapyAuthSpider,
                                            HelloWorldDisallowAuth,
                                            settings_auth)
    response = assert_single_response(items)
    assert 'hello' in response.body_as_unicode()
    assert response.status == 200
    assert response.splash_response_status == 200

    # httpauth shouldn't be disabled for non-Splash requests
    items, url, crawler = yield crawl_items(NonSplashSpider,
                                            HelloWorldProtected,
                                            settings_auth)
    response = assert_single_response(items)
    assert 'hello' in response.body_as_unicode()
    assert response.status == 200
    assert not hasattr(response, 'splash_response_status')


@pytest.mark.xfail(
    parse_version(scrapy.__version__) < parse_version("1.1"),
    reason="https://github.com/scrapy/scrapy/issues/1471",
    strict=True,
    run=True,
)
@requires_splash
@inlineCallbacks
def test_robotstxt_can_work(settings_auth):

    def assert_robots_disabled(items):
        response = assert_single_response(items)
        assert response.status == response.splash_response_status == 200
        assert b'hello' in response.body

    def assert_robots_enabled(items, crawler):
        assert len(items) == 0
        assert crawler.stats.get_value('downloader/exception_type_count/scrapy.exceptions.IgnoreRequest') == 1

    def _crawl_items(spider, resource):
        return crawl_items(
            spider,
            resource,
            settings_auth,
            url_path='/',  # https://github.com/scrapy/protego/issues/17
        )

    # when old auth method is used, robots.txt should be disabled
    items, url, crawler = yield _crawl_items(ScrapyAuthSpider,
                                             HelloWorldDisallowByRobots)
    assert_robots_disabled(items)

    # but robots.txt should still work for non-Splash requests
    items, url, crawler = yield _crawl_items(NonSplashSpider,
                                             HelloWorldDisallowByRobots)
    assert_robots_enabled(items, crawler)

    # robots.txt should work when a proper auth method is used
    settings_auth['SPLASH_USER'] = 'user'
    settings_auth['SPLASH_PASS'] = 'userpass'
    items, url, crawler = yield _crawl_items(LuaSpider,
                                             HelloWorldDisallowByRobots)
    assert_robots_enabled(items, crawler)

    # disable robotstxt middleware - robots middleware shouldn't work
    class DontObeyRobotsSpider(LuaSpider):
        custom_settings = {
            'HTTPERROR_ALLOW_ALL': True,
            'ROBOTSTXT_OBEY': False,
        }
    items, url, crawler = yield _crawl_items(DontObeyRobotsSpider,
                                             HelloWorldDisallowByRobots)
    assert_robots_disabled(items)

    # disable robotstxt middleware via request meta
    class MetaDontObeyRobotsSpider(ResponseSpider):
        def start_requests(self):
            yield SplashRequest(self.url,
                                endpoint='execute',
                                meta={'dont_obey_robotstxt': True},
                                args={'lua_source': DEFAULT_SCRIPT})

    items, url, crawler = yield _crawl_items(MetaDontObeyRobotsSpider,
                                             HelloWorldDisallowByRobots)
    assert_robots_disabled(items)

''',

'''

def _get_crawler(settings_dict):
    settings_dict = settings_dict.copy()
    settings_dict['DOWNLOAD_HANDLERS'] = {'s3': None}  # for faster test running
    crawler = scrapy.utils.test.get_crawler(settings_dict=settings_dict)
    if not hasattr(crawler, 'logformatter'):
        crawler.logformatter = None
    crawler.engine = scrapy.core.engine.ExecutionEngine(crawler, lambda _: None)
    # spider = crawler._create_spider("foo")
    return crawler


def _get_mw(settings_dict=None):
    crawler = _get_crawler(settings_dict or {})
    return SplashMiddleware.from_crawler(crawler)


def _get_cookie_mw():
    return SplashCookiesMiddleware(debug=True)


def test_nosplash():
    mw = _get_mw()
    cookie_mw = _get_cookie_mw()
    req = scrapy.Request("http://example.com")
    old_meta = copy.deepcopy(req.meta)

    assert cookie_mw.process_request(req, None) is None
    assert mw.process_request(req, None) is None
    assert old_meta == req.meta

    # response is not changed
    response = scrapy.http.Response("http://example.com", request=req)
    response2 = mw.process_response(req, response, None)
    response3 = cookie_mw.process_response(req, response, None)
    assert response2 is response
    assert response3 is response
    assert response3.url == "http://example.com"


def test_splash_request():
    mw = _get_mw()
    cookie_mw = _get_cookie_mw()

    req = SplashRequest("http://example.com?foo=bar&url=1&wait=100")
    assert repr(req) == "<GET http://example.com?foo=bar&url=1&wait=100>"

    # check request preprocessing
    req2 = cookie_mw.process_request(req, None) or req
    req2 = mw.process_request(req2, None) or req2

    assert req2 is not None
    assert req2 is not req
    assert req2.url == "http://127.0.0.1:8050/render.html"
    assert req2.headers == {b'Content-Type': [b'application/json']}
    assert req2.method == 'POST'
    assert isinstance(req2, SplashRequest)
    assert repr(req2) == "<GET http://example.com?foo=bar&url=1&wait=100 via http://127.0.0.1:8050/render.html>"

    expected_body = {'url': req.url}
    assert json.loads(to_unicode(req2.body)) == expected_body

    # check response post-processing
    response = scrapy.http.TextResponse("http://127.0.0.1:8050/render.html",
                            # Scrapy doesn't pass request to constructor
                            # request=req2,
                            headers={b'Content-Type': b'text/html'},
                            body=b"<html><body>Hello</body></html>")
    response2 = mw.process_response(req2, response, None)
    response2 = cookie_mw.process_response(req2, response2, None)
    assert isinstance(response2, scrapy_splash.SplashTextResponse)
    assert response2 is not response
    assert response2.real_url == req2.url
    assert response2.url == req.url
    assert response2.body == b"<html><body>Hello</body></html>"
    assert response2.css("body").extract_first() == "<body>Hello</body>"
    assert response2.headers == {b'Content-Type': [b'text/html']}

    # check .replace method
    response3 = response2.replace(status=404)
    assert response3.status == 404
    assert isinstance(response3, scrapy_splash.SplashTextResponse)
    for attr in ['url', 'real_url', 'headers', 'body']:
        assert getattr(response3, attr) == getattr(response2, attr)


def test_dont_process_response():
    mw = _get_mw()
    req = SplashRequest("http://example.com/",
        endpoint="render.html",
        dont_process_response=True,
    )
    req2 = mw.process_request(req, None)
    resp = scrapy.http.Response("http://example.com/")
    resp2 = mw.process_response(req2, resp, None)
    assert resp2.__class__ is Response
    assert resp2 is resp


def test_splash_request_parameters():
    mw = _get_mw()
    cookie_mw = _get_cookie_mw()

    def cb():
        pass

    req = SplashRequest("http://example.com/#!start", cb, 'POST',
        body="foo=bar",
        splash_url="http://mysplash.example.com",
        slot_policy=SlotPolicy.SINGLE_SLOT,
        endpoint="execute",
        splash_headers={'X-My-Header': 'value'},
        args={
            "lua_source": "function main() end",
            "myarg": 3.0,
        },
        magic_response=False,
        headers={'X-My-Header': 'value'}
    )
    req2 = cookie_mw.process_request(req, None) or req
    req2 = mw.process_request(req2, None) or req2

    assert req2.meta['ajax_crawlable'] is True
    assert req2.meta['splash'] == {
        'endpoint': 'execute',
        'splash_url': "http://mysplash.example.com",
        'slot_policy': SlotPolicy.SINGLE_SLOT,
        'splash_headers': {'X-My-Header': 'value'},
        'magic_response': False,
        'session_id': 'default',
        'http_status_from_error_code': True,
        'args': {
            'url': "http://example.com/#!start",
            'http_method': 'POST',
            'body': 'foo=bar',
            'cookies': [],
            'lua_source': 'function main() end',
            'myarg': 3.0,
            'headers': {
                'X-My-Header': 'value',
            }
        },
    }
    assert req2.callback == cb
    assert req2.headers == {
        b'Content-Type': [b'application/json'],
        b'X-My-Header': [b'value'],
    }

    # check response post-processing
    res = {
        'html': '<html><body>Hello</body></html>',
        'num_divs': 0.0,
    }
    res_body = json.dumps(res)
    response = scrapy.http.TextResponse("http://mysplash.example.com/execute",
                            # Scrapy doesn't pass request to constructor
                            # request=req2,
                            headers={b'Content-Type': b'application/json'},
                            body=res_body.encode('utf8'))
    response2 = mw.process_response(req2, response, None)
    response2 = cookie_mw.process_response(req2, response2, None)
    assert isinstance(response2, scrapy_splash.SplashJsonResponse)
    assert response2 is not response
    assert response2.real_url == req2.url
    assert response2.url == req.meta['splash']['args']['url']
    assert response2.data == res
    assert response2.body == res_body.encode('utf8')
    assert response2.text == response2.body_as_unicode() == res_body
    assert response2.encoding == 'utf8'
    assert response2.headers == {b'Content-Type': [b'application/json']}
    assert response2.splash_response_headers == response2.headers
    assert response2.status == response2.splash_response_status == 200


def test_magic_response():
    mw = _get_mw()
    cookie_mw = _get_cookie_mw()

    req = SplashRequest('http://example.com/',
                        endpoint='execute',
                        args={'lua_source': 'function main() end'},
                        magic_response=True,
                        cookies=[{'name': 'foo', 'value': 'bar'}])
    req = cookie_mw.process_request(req, None) or req
    req = mw.process_request(req, None) or req

    resp_data = {
        'url': "http://exmaple.com/#id42",
        'html': '<html><body>Hello 404</body></html>',
        'http_status': 404,
        'headers': [
            {'name': 'Content-Type', 'value': "text/html"},
            {'name': 'X-My-Header', 'value': "foo"},
            {'name': 'Set-Cookie', 'value': "bar=baz"},
        ],
        'cookies': [
            {'name': 'foo', 'value': 'bar'},
            {'name': 'bar', 'value': 'baz', 'domain': '.example.com'},
            {'name': 'session', 'value': '12345', 'path': '/',
             'expires': '2055-07-24T19:20:30Z'},
        ],
    }
    resp = scrapy.http.TextResponse("http://mysplash.example.com/execute",
                        headers={b'Content-Type': b'application/json'},
                        body=json.dumps(resp_data).encode('utf8'))
    resp2 = mw.process_response(req, resp, None)
    resp2 = cookie_mw.process_response(req, resp2, None)
    assert isinstance(resp2, scrapy_splash.SplashJsonResponse)
    assert resp2.data == resp_data
    assert resp2.body == b'<html><body>Hello 404</body></html>'
    assert resp2.text == '<html><body>Hello 404</body></html>'
    assert resp2.headers == {
        b'Content-Type': [b'text/html'],
        b'X-My-Header': [b'foo'],
        b'Set-Cookie': [b'bar=baz'],
    }
    assert resp2.splash_response_headers == {b'Content-Type': [b'application/json']}
    assert resp2.status == 404
    assert resp2.splash_response_status == 200
    assert resp2.url == "http://exmaple.com/#id42"
    assert len(resp2.cookiejar) == 3
    cookies = [c for c in resp2.cookiejar]
    assert {(c.name, c.value) for c in cookies} == {
        ('bar', 'baz'),
        ('foo', 'bar'),
        ('session', '12345')
    }

    # send second request using the same session and check the resulting cookies
    req = SplashRequest('http://example.com/foo',
                        endpoint='execute',
                        args={'lua_source': 'function main() end'},
                        magic_response=True,
                        cookies={'spam': 'ham'})
    req = cookie_mw.process_request(req, None) or req
    req = mw.process_request(req, None) or req

    resp_data = {
        'html': '<html><body>Hello</body></html>',
        'headers': [
            {'name': 'Content-Type', 'value': "text/html"},
            {'name': 'X-My-Header', 'value': "foo"},
            {'name': 'Set-Cookie', 'value': "bar=baz"},
        ],
        'cookies': [
            {'name': 'spam', 'value': 'ham'},
            {'name': 'egg', 'value': 'spam'},
            {'name': 'bar', 'value': 'baz', 'domain': '.example.com'},
           #{'name': 'foo', 'value': ''},  -- this won't be in response
            {'name': 'session', 'value': '12345', 'path': '/',
             'expires': '2056-07-24T19:20:30Z'},
        ],
    }
    resp = scrapy.http.TextResponse("http://mysplash.example.com/execute",
                        headers={b'Content-Type': b'application/json'},
                        body=json.dumps(resp_data).encode('utf8'))
    resp2 = mw.process_response(req, resp, None)
    resp2 = cookie_mw.process_response(req, resp2, None)
    assert isinstance(resp2, scrapy_splash.SplashJsonResponse)
    assert resp2.data == resp_data
    cookies = [c for c in resp2.cookiejar]
    assert {c.name for c in cookies} == {'session', 'egg', 'bar', 'spam'}
    for c in cookies:
        if c.name == 'session':
            assert c.expires == 2731692030
        if c.name == 'spam':
            assert c.value == 'ham'


def test_cookies():
    mw = _get_mw()
    cookie_mw = _get_cookie_mw()

    def request_with_cookies(cookies):
        req = SplashRequest(
            'http://example.com/foo',
            endpoint='execute',
            args={'lua_source': 'function main() end'},
            magic_response=True,
            cookies=cookies)
        req = cookie_mw.process_request(req, None) or req
        req = mw.process_request(req, None) or req
        return req

    def response_with_cookies(req, cookies):
        resp_data = {
            'html': '<html><body>Hello</body></html>',
            'headers': [],
            'cookies': cookies,
        }
        resp = scrapy.http.TextResponse(
            'http://mysplash.example.com/execute',
            headers={b'Content-Type': b'application/json'},
            body=json.dumps(resp_data).encode('utf8'))
        resp = mw.process_response(req, resp, None)
        resp = cookie_mw.process_response(req, resp, None)
        return resp

    # Concurent requests
    req1 = request_with_cookies({'spam': 'ham'})
    req2 = request_with_cookies({'bom': 'bam'})
    resp1 = response_with_cookies(req1, [
        {'name': 'spam', 'value': 'ham'},
        {'name': 'spam_x', 'value': 'ham_x'},
    ])
    resp2 = response_with_cookies(req2, [
        {'name': 'spam', 'value': 'ham'},  # because req2 was made after req1
        {'name': 'bom_x', 'value': 'bam_x'},
    ])
    assert resp1.cookiejar is resp2.cookiejar
    cookies = {c.name: c.value for c in resp1.cookiejar}
    assert cookies == {'spam': 'ham', 'spam_x': 'ham_x', 'bom_x': 'bam_x'}

    # Removing already removed
    req1 = request_with_cookies({'spam': 'ham'})
    req2 = request_with_cookies({'spam': 'ham', 'pom': 'pam'})
    resp2 = response_with_cookies(req2, [
        {'name': 'pom', 'value': 'pam'},
    ])
    resp1 = response_with_cookies(req1, [])
    assert resp1.cookiejar is resp2.cookiejar
    cookies = {c.name: c.value for c in resp1.cookiejar}
    assert cookies == {'pom': 'pam'}


def test_magic_response2():
    # check 'body' handling and another 'headers' format
    mw = _get_mw()
    req = SplashRequest('http://example.com/', magic_response=True,
                        headers={'foo': 'bar'}, dont_send_headers=True)
    req = mw.process_request(req, None) or req
    assert 'headers' not in req.meta['splash']['args']

    resp_data = {
        'body': base64.b64encode(b"binary data").decode('ascii'),
        'headers': {'Content-Type': 'text/plain'},
    }
    resp = scrapy.http.TextResponse("http://mysplash.example.com/execute",
                        headers={b'Content-Type': b'application/json'},
                        body=json.dumps(resp_data).encode('utf8'))
    resp2 = mw.process_response(req, resp, None)
    assert resp2.data == resp_data
    assert resp2.body == b'binary data'
    assert resp2.headers == {b'Content-Type': [b'text/plain']}
    assert resp2.splash_response_headers == {b'Content-Type': [b'application/json']}
    assert resp2.status == resp2.splash_response_status == 200
    assert resp2.url == "http://example.com/"


def test_unicode_url():
    mw = _get_mw()
    req = SplashRequest(
        # note unicode URL
        u"http://example.com/", endpoint='execute')
    req2 = mw.process_request(req, None) or req
    res = {'html': '<html><body>Hello</body></html>'}
    res_body = json.dumps(res)
    response = scrapy.http.TextResponse("http://mysplash.example.com/execute",
                            # Scrapy doesn't pass request to constructor
                            # request=req2,
                            headers={b'Content-Type': b'application/json'},
                            body=res_body.encode('utf8'))
    response2 = mw.process_response(req2, response, None)
    assert response2.url == "http://example.com/"


def test_magic_response_http_error():
    mw = _get_mw()
    req = SplashRequest('http://example.com/foo')
    req = mw.process_request(req, None) or req

    resp_data = {
        "info": {
            "error": "http404",
            "message": "Lua error: [string \\\"function main(splash)\\r...\\\"]:3: http404",
            "line_number": 3,
            "type": "LUA_ERROR",
            "source": "[string \\\"function main(splash)\\r...\\\"]"
        },
        "description": "Error happened while executing Lua script",
        "error": 400,
        "type": "ScriptError"
    }
    resp = scrapy.http.TextResponse("http://mysplash.example.com/execute", status=400,
                        headers={b'Content-Type': b'application/json'},
                        body=json.dumps(resp_data).encode('utf8'))
    resp = mw.process_response(req, resp, None)
    assert resp.data == resp_data
    assert resp.status == 404
    assert resp.splash_response_status == 400
    assert resp.url == "http://example.com/foo"


def test_change_response_class_to_text():
    mw = _get_mw()
    req = SplashRequest('http://example.com/', magic_response=True)
    req = mw.process_request(req, None) or req
    # Such response can come when downloading a file,
    # or returning splash:html(): the headers say it's binary,
    # but it can be decoded so it becomes a TextResponse.
    resp = scrapy.http.TextResponse('http://mysplash.example.com/execute',
                        headers={b'Content-Type': b'application/pdf'},
                        body=b'ascii binary data',
                        encoding='utf-8')
    resp2 = mw.process_response(req, resp, None)
    assert isinstance(resp2, TextResponse)
    assert resp2.url == 'http://example.com/'
    assert resp2.headers == {b'Content-Type': [b'application/pdf']}
    assert resp2.body == b'ascii binary data'


def test_change_response_class_to_json_binary():
    mw = _get_mw()
    # We set magic_response to False, because it's not a kind of data we would
    # expect from splash: we just return binary data.
    # If we set magic_response to True, the middleware will fail,
    # but this is ok because magic_response presumes we are expecting
    # a valid splash json response.
    req = SplashRequest('http://example.com/', magic_response=False)
    req = mw.process_request(req, None) or req
    resp = scrapy.http.Response('http://mysplash.example.com/execute',
                    headers={b'Content-Type': b'application/json'},
                    body=b'non-decodable data: \x98\x11\xe7\x17\x8f',
                    )
    resp2 = mw.process_response(req, resp, None)
    assert isinstance(resp2, Response)
    assert resp2.url == 'http://example.com/'
    assert resp2.headers == {b'Content-Type': [b'application/json']}
    assert resp2.body == b'non-decodable data: \x98\x11\xe7\x17\x8f'


def test_magic_response_caching(tmpdir):
    # prepare middlewares
    spider = scrapy.Spider(name='foo')
    crawler = _get_crawler({
        'HTTPCACHE_DIR': str(tmpdir.join('cache')),
        'HTTPCACHE_STORAGE': 'scrapy_splash.SplashAwareFSCacheStorage',
        'HTTPCACHE_ENABLED': True
    })
    cache_mw = scrapy.downloadermiddlewares.httpcache.HttpCacheMiddleware.from_crawler(crawler)
    mw = _get_mw()
    cookie_mw = _get_cookie_mw()

    def _get_req():
        return SplashRequest(
            url="http://example.com",
            endpoint='execute',
            magic_response=True,
            args={'lua_source': 'function main(splash) end'},
        )

    # Emulate Scrapy middleware chain.

    # first call
    req = _get_req()
    req = cookie_mw.process_request(req, spider) or req
    req = mw.process_request(req, spider) or req
    req = cache_mw.process_request(req, spider) or req
    assert isinstance(req, scrapy.Request)  # first call; the cache is empty

    resp_data = {
        'html': "<html><body>Hello</body></html>",
        'render_time': 0.5,
    }
    resp_body = json.dumps(resp_data).encode('utf8')
    resp = scrapy.http.TextResponse("http://example.com",
                        headers={b'Content-Type': b'application/json'},
                        body=resp_body)

    resp2 = cache_mw.process_response(req, resp, spider)
    resp3 = mw.process_response(req, resp2, spider)
    resp3 = cookie_mw.process_response(req, resp3, spider)

    assert resp3.text == "<html><body>Hello</body></html>"
    assert resp3.css("body").extract_first() == "<body>Hello</body>"
    assert resp3.data['render_time'] == 0.5

    # second call
    req = _get_req()
    req = cookie_mw.process_request(req, spider) or req
    req = mw.process_request(req, spider) or req
    cached_resp = cache_mw.process_request(req, spider) or req

    # response should be from cache:
    assert cached_resp.__class__ is TextResponse
    assert cached_resp.body == resp_body
    resp2_1 = cache_mw.process_response(req, cached_resp, spider)
    resp3_1 = mw.process_response(req, resp2_1, spider)
    resp3_1 = cookie_mw.process_response(req, resp3_1, spider)

    assert isinstance(resp3_1, scrapy_splash.SplashJsonResponse)
    assert resp3_1.body == b"<html><body>Hello</body></html>"
    assert resp3_1.text == "<html><body>Hello</body></html>"
    assert resp3_1.css("body").extract_first() == "<body>Hello</body>"
    assert resp3_1.data['render_time'] == 0.5
    assert resp3_1.headers[b'Content-Type'] == b'text/html; charset=utf-8'


def test_cache_args():
    spider = scrapy.Spider(name='foo')
    mw = _get_mw()
    mw.crawler.spider = spider
    mw.spider_opened(spider)
    dedupe_mw = SplashDeduplicateArgsMiddleware()

    # ========= Send first request - it should use save_args:
    lua_source = 'function main(splash) end'
    req = SplashRequest('http://example.com/foo',
                        endpoint='execute',
                        args={'lua_source': lua_source},
                        cache_args=['lua_source'])

    assert req.meta['splash']['args']['lua_source'] == lua_source
    # <---- spider
    req, = list(dedupe_mw.process_start_requests([req], spider))
    # ----> scheduler
    assert req.meta['splash']['args']['lua_source'] != lua_source
    assert list(mw._argument_values.values()) == [lua_source]
    assert list(mw._argument_values.keys()) == [req.meta['splash']['args']['lua_source']]
    # <---- scheduler
    # process request before sending it to the downloader
    req = mw.process_request(req, spider) or req
    # -----> downloader
    assert req.meta['splash']['args']['lua_source'] == lua_source
    assert req.meta['splash']['args']['save_args'] == ['lua_source']
    assert 'load_args' not in req.meta['splash']['args']
    assert req.meta['splash']['_local_arg_fingerprints'] == {
        'lua_source': list(mw._argument_values.keys())[0]
    }
    # <---- downloader
    resp_body = b'{}'
    resp = scrapy.http.TextResponse("http://example.com",
                        headers={
                            b'Content-Type': b'application/json',
                            b'X-Splash-Saved-Arguments': b'lua_source=ba001160ef96fe2a3f938fea9e6762e204a562b3'
                        },
                        body=resp_body)
    resp = mw.process_response(req, resp, None)

    # ============ Send second request - it should use load_args
    req2 = SplashRequest('http://example.com/bar',
                        endpoint='execute',
                        args={'lua_source': lua_source},
                        cache_args=['lua_source'])
    req2, item = list(dedupe_mw.process_spider_output(resp, [req2, {'key': 'value'}], spider))
    assert item == {'key': 'value'}
    # ----> scheduler
    assert req2.meta['splash']['args']['lua_source'] != lua_source
    # <---- scheduler
    # process request before sending it to the downloader
    req2 = mw.process_request(req2, spider) or req2
    # -----> downloader
    assert req2.meta['splash']['args']['load_args'] == {"lua_source": "ba001160ef96fe2a3f938fea9e6762e204a562b3"}
    assert "lua_source" not in req2.meta['splash']['args']
    assert "save_args" not in req2.meta['splash']['args']
    assert json.loads(req2.body.decode('utf8')) == {
        'load_args': {'lua_source': 'ba001160ef96fe2a3f938fea9e6762e204a562b3'},
        'url': 'http://example.com/bar'
    }
    # <---- downloader
    resp = scrapy.http.TextResponse("http://example.com/bar",
                        headers={b'Content-Type': b'application/json'},
                        body=b'{}')
    resp = mw.process_response(req, resp, spider)

    # =========== Third request is dispatched to another server where
    # =========== arguments are expired:
    req3 = SplashRequest('http://example.com/baz',
                         endpoint='execute',
                         args={'lua_source': lua_source},
                         cache_args=['lua_source'])
    req3, = list(dedupe_mw.process_spider_output(resp, [req3], spider))
    # ----> scheduler
    assert req3.meta['splash']['args']['lua_source'] != lua_source
    # <---- scheduler
    req3 = mw.process_request(req3, spider) or req3
    # -----> downloader
    assert json.loads(req3.body.decode('utf8')) == {
        'load_args': {'lua_source': 'ba001160ef96fe2a3f938fea9e6762e204a562b3'},
        'url': 'http://example.com/baz'
    }
    # <---- downloader

    resp_body = json.dumps({
        "type": "ExpiredArguments",
        "description": "Arguments stored with ``save_args`` are expired",
        "info": {"expired": ["html"]},
        "error": 498
    })
    resp = scrapy.http.TextResponse("127.0.0.1:8050",
                        headers={b'Content-Type': b'application/json'},
                        status=498,
                        body=resp_body.encode('utf8'))
    req4 = mw.process_response(req3, resp, spider)
    assert isinstance(req4, SplashRequest)

    # process this request again
    req4, = list(dedupe_mw.process_spider_output(resp, [req4], spider))
    req4 = mw.process_request(req4, spider) or req4

    # it should become save_args request after all middlewares
    assert json.loads(req4.body.decode('utf8')) == {
        'lua_source': 'function main(splash) end',
        'save_args': ['lua_source'],
        'url': 'http://example.com/baz'
    }
    assert mw._remote_keys == {}


def test_post_request():
    mw = _get_mw()
    for body in [b'', b'foo=bar']:
        req1 = scrapy.Request("http://example.com",
                              method="POST",
                              body=body,
                              meta={'splash': {'endpoint': 'render.html'}})
        req = mw.process_request(req1, None)
        assert json.loads(to_unicode(req.body)) == {
            'url': 'http://example.com',
            'http_method': 'POST',
            'body': to_unicode(body),
        }


def test_override_splash_url():
    mw = _get_mw()
    req1 = scrapy.Request("http://example.com", meta={
        'splash': {
            'endpoint': 'render.png',
            'splash_url': 'http://splash.example.com'
        }
    })
    req = mw.process_request(req1, None)
    req = mw.process_request(req, None) or req
    assert req.url == 'http://splash.example.com/render.png'
    assert json.loads(to_unicode(req.body)) == {'url': req1.url}


def test_url_with_fragment():
    mw = _get_mw()
    url = "http://example.com#id1"
    req = scrapy.Request("http://example.com", meta={
        'splash': {'args': {'url': url}}
    })
    req = mw.process_request(req, None) or req
    assert json.loads(to_unicode(req.body)) == {'url': url}


def test_splash_request_url_with_fragment():
    mw = _get_mw()
    url = "http://example.com#id1"
    req = SplashRequest(url)
    req = mw.process_request(req, None) or req
    assert json.loads(to_unicode(req.body)) == {'url': url}


def test_float_wait_arg():
    mw = _get_mw()
    req1 = scrapy.Request("http://example.com", meta={
        'splash': {
            'endpoint': 'render.html',
            'args': {'wait': 0.5}
        }
    })
    req = mw.process_request(req1, None)
    assert json.loads(to_unicode(req.body)) == {'url': req1.url, 'wait': 0.5}


def test_slot_policy_single_slot():
    mw = _get_mw()
    meta = {'splash': {
        'slot_policy': scrapy_splash.SlotPolicy.SINGLE_SLOT
    }}

    req1 = scrapy.Request("http://example.com/path?key=value", meta=meta)
    req1 = mw.process_request(req1, None)

    req2 = scrapy.Request("http://fooexample.com/path?key=value", meta=meta)
    req2 = mw.process_request(req2, None)

    assert req1.meta.get('download_slot')
    assert req1.meta['download_slot'] == req2.meta['download_slot']


def test_slot_policy_per_domain():
    mw = _get_mw()
    meta = {'splash': {
        'slot_policy': scrapy_splash.SlotPolicy.PER_DOMAIN
    }}

    req1 = scrapy.Request("http://example.com/path?key=value", meta=meta)
    req1 = mw.process_request(req1, None)

    req2 = scrapy.Request("http://example.com/path2", meta=meta)
    req2 = mw.process_request(req2, None)

    req3 = scrapy.Request("http://fooexample.com/path?key=value", meta=meta)
    req3 = mw.process_request(req3, None)

    assert req1.meta.get('download_slot')
    assert req3.meta.get('download_slot')

    assert req1.meta['download_slot'] == req2.meta['download_slot']
    assert req1.meta['download_slot'] != req3.meta['download_slot']


def test_slot_policy_scrapy_default():
    mw = _get_mw()
    req = scrapy.Request("http://example.com", meta={'splash': {
        'slot_policy': scrapy_splash.SlotPolicy.SCRAPY_DEFAULT
    }})
    req = mw.process_request(req, None)
    assert 'download_slot' not in req.meta


def test_adjust_timeout():
    mw = _get_mw()
    req1 = scrapy.Request("http://example.com", meta={
        'splash': {'args': {'timeout': 60, 'html': 1}},

        # download_timeout is always present,
        # it is set by DownloadTimeoutMiddleware
        'download_timeout': 30,
    })
    req1 = mw.process_request(req1, None)
    assert req1.meta['download_timeout'] > 60

    req2 = scrapy.Request("http://example.com", meta={
        'splash': {'args': {'html': 1}},
        'download_timeout': 30,
    })
    req2 = mw.process_request(req2, None)
    assert req2.meta['download_timeout'] == 30


def test_auth():
    def assert_auth_header(user, pwd, header):
        mw = _get_mw({'SPLASH_USER': user, 'SPLASH_PASS': pwd})
        req = mw.process_request(SplashRequest("http://example.com"), None)
        assert 'Authorization' in req.headers
        assert req.headers['Authorization'] == header

    def assert_no_auth_header(user, pwd):
        if user is not None or pwd is not None:
            mw = _get_mw({'SPLASH_USER': user, 'SPLASH_PASS': pwd})
        else:
            mw = _get_mw()
        req = mw.process_request(SplashRequest("http://example.com"), None)
        assert 'Authorization' not in req.headers

    assert_auth_header('root', '', b'Basic cm9vdDo=')
    assert_auth_header('root', 'pwd', b'Basic cm9vdDpwd2Q=')
    assert_auth_header('', 'pwd', b'Basic OnB3ZA==')

    assert_no_auth_header('', '')
    assert_no_auth_header(None, None)
'''

]