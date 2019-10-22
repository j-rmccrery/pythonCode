"""
Scrape the iOS store for certain keywords in the app descriptions
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
and operating_system = 'iOS' 
and dst = 'B0094' 
and d >= """ + start + """ 
and d <= """ + finish + """ 
and impression_time is not null 
limit 300000 """ 
query = query.replace('\n', '')
for data_ in pd.read_sql(query, Athconn, chunksize  = 1000):
    data.append(data_)
data = pd.concat(data)

import itunes

bundles = data.app_bundle_id.unique()
descHolder = []
for z, i in enumerate(bundles):
    if z % 1000 == 0:
        print(z/len(bundles))
        continue
    try:
        desc = itunes.lookup(i).description
        if any(word in desc.lower() for word in ['vpn', 'virus']):
            if 'zombie' in desc.lower():
                continue
            else:    
                descHolder.append([i, desc])
    except:
        continue

def dirMaker(x):
    d = '/home/jr/data/reporting/virusSearch/' + str(x)
    if not os.path.exists(d):
        os.makedirs(d)
    os.chdir(d)
dirMaker(finish)
descHolder = pd.DataFrame(descHolder)


descHolder.to_csv('viri.csv')

