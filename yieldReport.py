"""
Access some database and create yield reports 
"""
from pyathenajdbc import connect
import pandas as pd
import pdb
import sys
import pickle 
import os, glob
from scipy.stats import beta
import datetime as dt

Athconn = connect(access_key='xxx',
               secret_key='xxx',
               s3_staging_dir='xxx',
               region_name='xxx')  

def maker(campaign, connection):
    try:
        post = []
        query = """select d, rpm_micros, app_bundle_id, ad_event.ad_event_time, ad_event.ad_event_parameters from won_sessions 
        CROSS JOIN UNNEST(ad_events) AS t(ad_event) 
        where ad_event.ad_event_type = 'post_install' 
        and campaign_id in (""" + campaign + """) 
        and d >= 20180100 
        and dst = 'B0094' 
        and impression_time is not null 
        """

        query = query.replace('\n', '')

        for _ in pd.read_sql(query, Athconn, chunksize  = 1000):
            post.append(_)
        post = pd.concat(post)
        # pdb.set_trace()
        post['ad_event_parameters'] = post.ad_event_parameters.apply(lambda x: x.lower())
        post_ = post[post.ad_event_parameters.str.contains('purchase')]
        _post = post[post.ad_event_parameters.str.contains('pf7')]
        post = pd.concat([post_, _post])
        def pricer(x):
            return x.split('value=')[1].replace('}', '')
        post['Rev'] = post.ad_event_parameters.apply(lambda x: pricer(x))
        post = post.drop('ad_event_parameters', axis = 1)
        # pdb.set_trace()
        import datetime as dt
        post['pDate'] = post.ad_event_time.apply(lambda x: x.date())
        post['iDate'] = post.d.apply(lambda x: dt.datetime.strptime(str(x), "%Y%m%d").date())
        post = post.drop('d', axis = 1)
        post = post.drop('ad_event_time', axis = 1)

        post['tDiff'] = post['pDate'] - post['iDate']
        post['tDiff'] = post['tDiff'].apply(lambda x: x.days)
        post['Rev'] = post['Rev'].astype(float)

        fin = []
        for d in post.iDate.unique():
            t = post[post.iDate == d]
            for app in t.app_bundle_id.unique():
                tt = t[t.app_bundle_id == app]
                d3 = tt[tt.tDiff <= 3].Rev.sum()
                d7 = tt[tt.tDiff <= 7].Rev.sum()
                d14 = tt[tt.tDiff <= 14].Rev.sum()
                cpi = tt.rpm_micros.unique()[0]
                fin.append([d, app, d3, d7, d14, cpi])

        fins = pd.DataFrame(fin)
        fins.columns = ['date', 'app', 'd3', 'd7', 'd14', 'rev']
        fins['rev'] = fins['rev'] / 1000 / 1000

        import matplotlib.pyplot as plt
        plt.clf()
        plt.close()
        def dirMaker():
            d = '/mnt/499f5b4c-132a-4775-a414-ce6eba504896/data/reporting/testing/'+str(dt.datetime.now().date())
            if not os.path.exists(d):
                os.makedirs(d)
            print(d)
            os.chdir(d)

        dirMaker()
        t = fins
        t = t.groupby(['date']).sum().reset_index().sort_values('date').reset_index(drop = True)
        t = t[t.date >= dt.date(2017, 11, 30)]
        t = t[t.date <= dt.date(2018, 8, 1)]
        t['yield3'] = t['d3'] / t['rev']
        t['yield7'] = t['d7'] / t['rev']

        t['yield3Rolling'] = pd.rolling_mean(t['yield3'], window = 7)
        t['yield7Rolling'] = pd.rolling_mean(t['yield7'], window = 7)

        plt.plot(t.date, t.yield3Rolling, label = 'd3Yield')
        plt.plot(t.date, t.yield7Rolling, label = 'd7Yield')

        plt.grid()
        plt.legend()
        locs, labels = plt.xticks()
        # plt.title('testang')
        plt.setp(labels, rotation=70)
        # plt.show()
        plt.savefig(campaign + '.pdf')
        fins.to_csv('test.csv')
    except:
        print(campaign)

campaigns = ["'38'"]

for campaign in campaigns:
    maker(campaign, Athconn)