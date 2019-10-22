
"""
Scrape the google play store for certain keywords in the app descriptions
"""

# coding: utf-8

from pyathenajdbc import connect
import pandas as pd
import pdb
import sys
import pickle 
import os, glob
from scipy.stats import beta
import datetime as dt
import numpy as np

Athconn = connect(access_key='xxx',
               secret_key='xxx',
               s3_staging_dir='xxx',
               region_name='xxx')  

finish = str(dt.datetime.now().date() - dt.timedelta(days=1)).replace('-', '')
start = str(dt.datetime.now().date() - dt.timedelta(days=7)).replace('-', '')


data = []
query = """select distinct app_bundle_id from won_sessions 
where device_type = 'Phone' 
and operating_system = 'Android' 
and dst = 'B0094' 
and d >= """ + start + """ 
and d <= """ + finish + """ 
and impression_time is not null 
limit 300000 """ 
query = query.replace('\n', '')
for data_ in pd.read_sql(query, Athconn, chunksize  = 1000):
    data.append(data_)
data = pd.concat(data)

import google_play, time


bundles = data.app_bundle_id.unique()


def dirMaker(x):
    d = '/home/jr/data/reporting/virusSearch/' + str(x)
    if not os.path.exists(d):
        os.makedirs(d)
    os.chdir(d)
dirMaker(finish)

descHolder = []
z = 0
for p, i in enumerate(bundles):
    try:
        desc = google_play.app(i)['description']
        if any(word in desc.lower() for word in ['vpn', 'virus']):
            if 'zombie' in desc.lower():
                continue
            else:    
                descHolder.append([i, desc])
    except:
        continue
    z += 1
    if z == 20:
        time.sleep(1)
        z = 0
    if p % 1000 == 0: 
        pd.DataFrame(descHolder).to_csv(str(p) +'.csv')
        descHolder = []
        print(p/len(bundles))

descHolder = pd.DataFrame(descHolder)

descHolder.to_csv('androidViri.csv')

