from scrapy.item import Item, Field
from scrapy.spider import BaseSpider
from scrapy.selector import HtmlXPathSelector
from apple.items import AppItem
import pdb

class AppItem(Item):

    app_name = Field()
    category = Field()
    appstore_link = Field()
    img_src = Field()

class AppleSpider(BaseSpider):
    name = "apple"
    allowed_domains = ["apple.com"]
    start_urls = ["https://itunes.apple.com/us/app/"]

    def parse(self, response):
        hxs = HtmlXPathSelector(response)
        apps = hxs.select('//*[@id="content"]/section/ul/li')
        pdb.set_trace()
        count = 0
        items = []

        for app in apps:

            item = AppItem()
            item['app_name'] = app.select('//h3/a/text()')[count].extract()
            item['appstore_link'] = app.select('//h3/a/@href')[count].extract()
            item['category'] = app.select('//h4/a/text()')[count].extract()
            item['img_src'] = app.select('//a/img/@src')[count].extract()

            items.append(item)
            count += 1

        return items
AppleSpider()