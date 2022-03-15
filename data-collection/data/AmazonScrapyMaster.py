file_string = [

'''

class CateItem(scrapy.Item):
    title = scrapy.Field()
    link = scrapy.Field()
    level = scrapy.Field()
    pid = scrapy.Field()
    pass

class AsinBestItem(scrapy.Item):
    asin = scrapy.Field()
    cid = scrapy.Field()
    rank = scrapy.Field()
    pass

class DetailItem(scrapy.Item):
    asin = scrapy.Field()
    image = scrapy.Field()
    title = scrapy.Field()
    star = scrapy.Field()
    reviews = scrapy.Field()
    seller_price = scrapy.Field()
    amazon_price = scrapy.Field()
    pass

class ReviewProfileItem(scrapy.Item):
    asin = scrapy.Field()
    product = scrapy.Field()
    brand = scrapy.Field()
    seller = scrapy.Field()
    image = scrapy.Field()
    review_total = scrapy.Field()
    review_rate = scrapy.Field()
    pct_five = scrapy.Field()
    pct_four = scrapy.Field()
    pct_three = scrapy.Field()
    pct_two = scrapy.Field()
    pct_one = scrapy.Field()
    pass


class ReviewDetailItem(scrapy.Item):
    asin = scrapy.Field()
    review_id = scrapy.Field()
    reviewer = scrapy.Field()
    review_url = scrapy.Field()
    star = scrapy.Field()
    date = scrapy.Field()
    title = scrapy.Field()
    content = scrapy.Field()
    pass


class KeywordRankingItem(scrapy.Item):
    skwd_id = scrapy.Field()
    rank = scrapy.Field()
    date = scrapy.Field()


class SalesRankingItem(scrapy.Item):
    rank = scrapy.Field()
    classify = scrapy.Field()
    asin = scrapy.Field()

''',

'''
def __init__(self):
        scrapy.Spider.__init__(self)
        pydispatch.dispatcher.connect(self.handle_spider_closed, signals.spider_closed)
        # all asin scrapied will store in the array
        self.asin_pool = []

def start_requests(self):
    cates = Sql.findall_cate_level1()
    for row in cates:
        row['link'] += '?ajax=1'
        yield scrapy.Request(url=row['link']+'&pg=1', callback=self.parse, meta={'cid': row['id'], 'page': 1, 'link': row['link']})

def parse(self, response):
    list = response.css('.zg_itemImmersion')

    # scrapy next page  go go go !
    response.meta['page'] = response.meta['page'] +1
    if response.meta['page'] < 6:
        yield scrapy.Request(url=response.meta['link']+'&pg='+str(response.meta['page']), callback=self.parse, meta=response.meta)

    # yield the asin
    for row in list:
        try:
            info = row.css('.zg_itemWrapper')[0].css('div::attr(data-p13n-asin-metadata)')[0].extract()
            rank = int(float(row.css('.zg_rankNumber::text')[0].extract()))

        except:
            continue
            pass
        info = json.loads(info)
        item = AsinBestItem()
        item['asin'] = info['asin']
        item['cid'] = response.meta['cid']
        item['rank'] = rank
        yield item

def handle_spider_closed(self, spider):
    Sql.store_best_asin()
    work_time = datetime.now() - spider.started_on
    print('total spent:', work_time)
    print('done')
''',

'''
def start_requests(self):

        urls = [
            'https://www.amazon.com/Best-Sellers/zgbs/',
        ]
        Sql.clear_cate(self.level)
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse, meta={'level': self.level})

def parse(self, response):

    if response.meta['level'] == 1:
        list = response.css('#zg_browseRoot ul')[0].css('li a')
    elif response.meta['level'] == 2:
        list = response.css('#zg_browseRoot ul')[0].css('ul')[0].css('li a')
    else:
        return 0
    item = CateItem()
    leve_cur = response.meta['level']
    response.meta['level'] = response.meta['level'] + 1

    for one in list:
        item['title'] = one.css('::text')[0].extract()
        link = one.css('::attr(href)')[0].extract()
        item['link'] = link.split('ref=')[0]
        item['level'] = leve_cur
        item['pid'] = 1
        yield item
        if int(float(self.level)) > 1:
            yield scrapy.Request(url=item['link'], callback=self.parse, meta=response.meta)
''',
'''
def __init__(self):
        scrapy.Spider.__init__(self)
        pydispatch.dispatcher.connect(self.handle_spider_closed, signals.spider_closed)
        # all asin scrapied will store in the array
        self.product_pool = {}
        self.log = []
        self.products = []

def start_requests(self):
    self.products = Sql.findall_asin_level1()
    print(len(self.products))
    for row in self.products:
        yield scrapy.Request(
                url='https://www.amazon.com/gp/offer-listing/' + row['asin'] + '/?f_new=true',
                callback=self.listing_parse,
                meta={
                    'asin': row['asin'],
                    'cid': row['cid'],
                }
        )

def review_parse(self, response):
    item = self.fetch_detail_from_review_page(response)
    self.product_pool[item['asin']] = item
    yield item

def listing_parse(self, response):
    print(response.status)

    if not response.css('#olpProductImage'):
        yield scrapy.Request(
                url='https://www.amazon.com/product-reviews/' + response.meta['asin'],
                callback=self.review_parse,
                meta={'asin': response.meta['asin'], 'cid': response.meta['cid']}
        )
        return
    try:
        item = self.fetch_detail_from_listing_page(response)
        self.product_pool[item['asin']] = item
    except Exception as err:
        print(err)
        print(response.meta['asin'])
    yield item

def handle_spider_closed(self, spider):
    work_time = datetime.now() - spider.started_on
    print('total spent:', work_time)
    print(len(self.product_pool), 'item fetched')
    print(self.product_pool)
    print('done')
    print(self.log)




def fetch_detail_from_listing_page(self, response):
    item = DetailItem()
    item['asin'] = response.meta['asin']
    item['image'] = response.css('#olpProductImage img::attr(src)')[0].extract().strip().replace('_SS160', '_SS320')
    item['title'] = response.css('title::text')[0].extract().split(':')[2].strip()

    try:
        item['star'] = response.css('.a-icon-star span::text')[0].extract().split(' ')[0].strip()
    except:
        item['star'] = 0
    try:
        item['reviews'] = response.css('.a-size-small > .a-link-normal::text')[0].extract().strip().split(' ')[0]
    except:
        item['reviews'] = 0

    price_info_list = response.css(\".olpOffer[role=\\\"row\\\"] \")
    item['amazon_price'] = 0
    item['seller_price'] = 0
    for row in price_info_list:
        if (item['amazon_price'] == 0) and row.css(".olpSellerName > img"):
            try:
                item['amazon_price'] = row.css('.olpOfferPrice::text')[0].extract().strip().lstrip('$')
            except:
                item['amazon_price'] = 0
            continue
        if (item['seller_price'] == 0) and (not row.css(".olpSellerName > img")):
            try:
                item['seller_price'] = row.css('.olpOfferPrice::text')[0].extract().strip().lstrip('$')
            except:
                item['seller_price'] = 0
    return item

def fetch_detail_from_review_page(self, response):


    info = response.css('#cm_cr-product_info')[0].extract()
    item = DetailItem()
    item['asin'] = response.meta['asin']
    item['image'] = response.css('.product-image img::attr(src)')[0].extract().strip().replace('S60', 'S320')
    item['title'] = response.css('.product-title >h1>a::text')[0].extract().strip()
    item['star'] = re.findall("([0-9].[0-9]) out of", info)[0]

    # 获取评价总数
    item['reviews'] = response.css('.AverageCustomerReviews .totalReviewCount::text')[0].extract().strip()
    item['reviews'] = Helper.get_num_split_comma(item['reviews'])
    item['seller_price'] = 0
    item['amazon_price'] = 0
    price = response.css('.arp-price::text')[0].extract().strip().lstrip('$')
    item['amazon_price'] = price
    return item
''',

'''

def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.items = {}
    self.found = {}
    self.keyword_pool = {}
    self.store_poll = {}
    self.store_date = {}
    dispatcher.connect(self.init_scrapy, signals.engine_started)
    dispatcher.connect(self.close_scrapy, signals.engine_stopped)

def start_requests(self):
    for keyword, poll in self.keyword_pool.items():
        yield scrapy.Request(('https://www.amazon.com/s/?field-keywords=%s&t=' + Helper.random_str(10)) % keyword,
                                self.load_first_page, meta={'items': poll})

def parse(self, response):
    result_li = response.xpath('//li[@data-asin]')
    for item in response.meta['items']:
        if len(result_li) == 0:
            self.found[item['id']] = 'none'
        else:
            for result in result_li:
                data_asin = result.xpath('./@data-asin').extract()[0]
                if data_asin == item['asin']:
                    # print(item)
                    self.found[item['id']] = True
                    # keywordItem = KeywordRankingItem()
                    data_id = result.xpath('./@id').extract()[0]
                    item_id = data_id.split('_')[1]
                    rank = int(item_id) +1
                    if item['id'] in self.store_poll.keys():
                        self.store_poll[item['id']].append(rank)
                    else:
                        self.store_poll[item['id']] = [rank]
                    self.store_date[item['id']] = Helper.get_now_date()
                    break

def load_first_page(self, response):
    page = response.css('#bottomBar span.pagnDisabled::text').extract()
    page = 1 if len(page) == 0 else int(page[0])
    page_num = 1
    while page_num <= page:
        # yield scrapy.Request(response.url + '&page=%s' % page_num, self.parse, meta={'asin': response.meta['item']['asin'],
        #                                                                              'skwd_id': response.meta['item']['id']})
        yield scrapy.Request(response.url + '&page=%s' % page_num, self.parse, meta={'items': response.meta['items']})
        page_num += 1

def init_scrapy(self):
    self.items = RankingSql.fetch_keywords_ranking()
    for item in self.items:
        if item['keyword'] in self.keyword_pool.keys():
            self.keyword_pool[item['keyword']].append({'id': item['id'], 'asin': item['asin']})
        else:
            self.keyword_pool[item['keyword']] = [{'id': item['id'], 'asin': item['asin']}]

    self.found = {item['id']: False for item in self.items}

def close_scrapy(self):
    for skwd_id, is_found in self.found.items():
        if is_found is not True:
            if is_found == 'none':
                # RankingSql.update_keywords_none_rank(skwd_id)
                logging.info('[keyword none] skwd_id:[%s]' % skwd_id)
            else:
                RankingSql.update_keywords_expire_rank(skwd_id)
        else:
            keywordrank = KeywordRankingItem()
            keywordrank['skwd_id'] = skwd_id
            keywordrank['rank'] = min(self.store_poll[skwd_id])
            keywordrank['date'] = self.store_date[skwd_id]
            RankingSql.insert_keyword_ranking(keywordrank)

''',

'''

def __init__(self, asin, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.asin = asin

def start_requests(self):
    yield scrapy.Request('https://www.amazon.com/product-reviews/%s' % self.asin, callback=self.parse)

def parse(self, response):
    item = ReviewProfileItem()

    item['asin'] = response.meta['asin'] if 'asin' in response.meta else self.asin
    # 获取平均评价数值
    average = response.css('.averageStarRatingNumerical a span::text').extract()  # 获取平均评价值
    item['review_rate'] = Helper.get_star_split_str(average[0])  # 获取平均值
    # 获取评价总数
    total = response.css('.AverageCustomerReviews .totalReviewCount::text').extract()  # 获取评价总数
    item['review_total'] = Helper.get_num_split_comma(total[0])
    # 获取产品名称
    product = response.css('.product-title h1 a::text').extract()
    item['product'] = product[0]
    # 获取产品 brand
    item['brand'] = response.css('.product-by-line a::text').extract()[0]
    item['image'] = response.css('.product-image img::attr(src)').extract()[0]

    # 获取产品商家
    item['seller'] = item['brand']
    # 获取各星评价百分比数
    review_summary = response.css('.reviewNumericalSummary .histogram '
                                    '#histogramTable tr td:last-child').re(r'\d{1,3}\%')

    pct = list(map(lambda x: x[0:-1], review_summary))

    item['pct_five'] = pct[0]
    item['pct_four'] = pct[1]
    item['pct_three'] = pct[2]
    item['pct_two'] = pct[3]
    item['pct_one'] = pct[4]

    yield item
''',

'''
def __init__(self, asin, daily=0, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.asin = asin
    self.last_review = 0
    self.profile_update_self = False    # profile自动计数更新
    self.updated = False   # profile是否更新过
    self.daily = True if int(daily) == 1 else False  # 判断是否是每日更新
    self.start_urls = [
        'https://www.amazon.com/product-reviews/%s?sortBy=recent&filterByStar=three_star' % self.asin,
        'https://www.amazon.com/product-reviews/%s?sortBy=recent&filterByStar=two_star' % self.asin,
        'https://www.amazon.com/product-reviews/%s?sortBy=recent&filterByStar=one_star' % self.asin
    ]
    dispatcher.connect(self.update_profile_self, signals.engine_stopped)
    dispatcher.connect(self.init_profile, signals.engine_started)

def start_requests(self):
    self.load_profile()
    for url in self.start_urls:
        yield scrapy.Request(url, callback=self.get_detail)

def parse(self, response):
    reviews = response.css('.review-views .review')
    for row in reviews:
        item = ReviewDetailItem()
        item['asin'] = self.asin
        item['review_id'] = row.css('div::attr(id)')[0].extract()
        item['reviewer'] = row.css('.author::text')[0].extract()
        item['title'] = row.css('.review-title::text')[0].extract()
        item['review_url'] = row.css('.review-title::attr(href)')[0].extract()
        item['date'] = Helper.get_date_split_str(row.css('.review-date::text')[0].extract())
        item['star'] = Helper.get_star_split_str(row.css('.review-rating span::text')[0].extract())
        content = row.css('.review-data .review-text::text').extract()
        item['content'] = '<br />'.join(content) if len(content) > 0 else ''
        yield item

def get_detail(self, response):
    # 获取页面数
    page = response.css('ul.a-pagination li a::text')

    i = 1
    # 获取评价总数
    total = response.css('.AverageCustomerReviews .totalReviewCount::text').extract()  # 获取评价总数
    now_total = Helper.get_num_split_comma(total[0])
    last_review = self.last_review
    sub_total = int(now_total) - int(last_review)
    if sub_total != 0:
        # if sub_total != 0:  # 若计算出的页数 不为0 则说明有新的评论，更新profile
        self.updated = True
        yield scrapy.Request('https://www.amazon.com/product-reviews/%s' % self.asin,
                                callback=self.profile_parse)
        if len(page) < 3:  # 若找到的a标签总数小于3 说明没有page组件 只有1页数据
            yield scrapy.Request(url=response.url + '&pageNumber=1', callback=self.parse)
        else:
            if self.daily:
                page_num = math.ceil(sub_total / 10)
                print('update item page_num is %s' % page_num)
            else:
                self.profile_update_self = True
                page_num = Helper.get_num_split_comma(page[len(page) - 3].extract())  # 获得总页数
            while i <= int(page_num):
                yield scrapy.Request(url=response.url + '&pageNumber=%s' % i,
                                        callback=self.parse)
                i = i + 1
    else:
        print('there is no item to update')

def profile_parse(self, response):
    item = ReviewProfileItem()

    item['asin'] = self.asin
    # 获取平均评价数值
    average = response.css('.averageStarRatingNumerical a span::text').extract()  # 获取平均评价值
    item['review_rate'] = Helper.get_star_split_str(average[0])  # 获取平均值
    # 获取评价总数
    total = response.css('.AverageCustomerReviews .totalReviewCount::text').extract()  # 获取评价总数
    item['review_total'] = Helper.get_num_split_comma(total[0])
    # 获取产品名称
    product = response.css('.product-title h1 a::text').extract()
    item['product'] = product[0]
    # 获取产品 brand
    item['brand'] = response.css('.product-by-line a::text').extract()[0]
    item['image'] = response.css('.product-image img::attr(src)').extract()[0]

    # 获取产品商家
    item['seller'] = item['brand']
    # 获取各星评价百分比数
    review_summary = response.css('.reviewNumericalSummary .histogram '
                                    '#histogramTable tr td:last-child').re(r'\d{1,3}\%')

    pct = list(map(lambda x: x[0:-1], review_summary))

    item['pct_five'] = pct[0]
    item['pct_four'] = pct[1]
    item['pct_three'] = pct[2]
    item['pct_two'] = pct[3]
    item['pct_one'] = pct[4]

    yield item

def load_profile(self):
    # 若没有profile记录 则抓取新的profile 录入数据库
    if self.last_review is False:
        self.profile_update_self = True
        print('this asin profile is not exist, now to get the profile of asin:', self.asin)
        yield scrapy.Request('https://www.amazon.com/product-reviews/%s' % self.asin,
                                callback=self.profile_parse)
        self.last_review = ReviewSql.get_last_review_total(self.asin)

# scrapy 完成后 加载，如果是没有记录的profile 初次insert lastest_review为0 将所有多余的数据跑完后 防止第二次重复跑取，将latest_total更新
def update_profile_self(self):
    if self.profile_update_self is True and self.updated is False:
        # 若需要自主更新 并且 未更新状态
        ReviewSql.update_profile_self(self.asin)

# scrapy 开始前加载，获取目前asin的latest_review
def init_profile(self):
    self.last_review = ReviewSql.get_last_review_total(self.asin)

''',

'''
def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.items = []
        dispatcher.connect(self.load_asin, signals.engine_started)

def start_requests(self):
    for item in self.items:
        yield scrapy.Request('https://www.amazon.com/dp/%s' % item['asin'], self.parse, meta={'item': item})

def parse(self, response):
    product_detail = response.xpath('//div/table').re(r'#[0-9,]+(?:.*)in.*\(.*[Ss]ee [Tt]op.*\)')
    if len(product_detail) == 0:
        product_detail = response.css('div #SalesRank').re(r'#[0-9,]+(?:.*)in.*\(.*[Ss]ee [Tt]op.*\)')
    if len(product_detail) != 0:
        item = SalesRankingItem()
        key_rank_str = product_detail[0]
        key_rank_tuple = Helper.get_rank_classify(key_rank_str)
        item['rank'] = Helper.get_num_split_comma(key_rank_tuple[0])
        item['classify'] = key_rank_tuple[1]
        item['asin'] = response.meta['item']['asin']
        yield item
    else:
        raise Exception('catch asin[%s] sales ranking error' % response.meta['item']['asin'])

def load_asin(self):
    self.items = RankingSql.fetch_sales_ranking()

''',

'''
def start_requests(self):
    url = "http://fineproxy.org/eng/fresh-proxies/"
    yield scrapy.Request(url=url, callback=self.parse, meta={})

def parse(self, response):
    pattern = "<strong>Fast proxies: </strong>(.*)<strong>Other fresh and working proxies:</strong>"
    tmp = re.findall(pattern, response.text)[0]
    proxy = re.findall("([0-9]{1,4}.[0-9]{1,4}.[0-9]{1,4}.[0-9]{1,4}:[0-9]{1,4})", tmp)
    with open('proxy.json', 'w') as f:
        json.dump(proxy, f)
''',

'''
def start_requests(self):

    self.headers = {
        'Host': 'www.kuaidaili.com',
        'Upgrade-Insecure-Requests': '1',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.11; rv:52.0) Gecko/20100101 Firefox/52.0',
    }

    url = "http://www.kuaidaili.com/free/inha/"
    yield scrapy.Request(url=url, callback=self.parse, meta={})

def parse(self, response):
    print(response.status)
    print('3333')
    print(response.css('.center tr').re('td'))

''',

'''
def start_requests(self):
    url = "http://www.qq.com"
    db = pymysql.connect(settings.MYSQL_HOST, settings.MYSQL_USER, settings.MYSQL_PASSWORD, settings.MYSQL_DB, charset=settings.MYSQL_CHARSET, cursorclass=pymysql.cursors.DictCursor)
    cursor = db.cursor()

    sql = "SELECT CONCAT_WS(':', ip, port) AS proxy FROM proxy where work = 1"
    cursor.execute(sql)

    proxy_array = []
    proxy_list = cursor.fetchall()
    for item in proxy_list:
        proxy_array.append(item['proxy'])

    with open('proxy.json', 'w') as f:
        json.dump(proxy_array, f)
    yield scrapy.Request(url=url, callback=self.parse, meta={})

def parse(self, response):
    print('proxy update done')
''',

'''
def __init__(self):
    pydispatch.dispatcher.connect(self.handle_spider_closed, signals.spider_closed)
    self.result_pool = {}
    self.log = []

def start_requests(self):
    return

def parse(self, response):
    return

def print_progress(self, spider):
    work_time = datetime.now() - spider.started_on
    print('Spent:', work_time, ':', len(self.result_pool), 'item fetched')

def handle_spider_closed(self):
    return
''',

'''
def __init__(self, asin='B07K97BQDF'):
        AmazonBaseSpider.__init__(self)
        self.asin = asin

def start_requests(self):
    yield scrapy.Request(
        url='https://www.amazon.com/dp/' + self.asin,
        callback=self.parse,
        meta={
            'asin': self.asin,
            'cid': -10
        }
    )

def parse(self, response):
    print(response.meta['asin'])
    self.result_pool[response.meta['asin']] = {}
    self.result_pool[response.meta['asin']]['title'] = 'title for ' + response.meta['asin']

# Bingo! Here we get the result and You can restore or output it
def handle_spider_closed(self, spider):
    print(self.result_pool.get(self.asin))
    AmazonBaseSpider.print_progress(self, spider)

'''

]