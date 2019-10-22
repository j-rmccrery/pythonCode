"""
Scrapes database daily for users who pass some threshold on a variety of events of importance.  Uploads to DSP for automated bidding and user targeting 
"""

# coding: utf-8

from pyathenajdbc import connect
import pandas as pd
import pdb
import sys
import pickle 
import os, glob

Athconn = connect(access_key='xxxx',
               secret_key='xxxx',
               s3_staging_dir='xxx',
               region_name='xxxx')  

def mainFunction(Athconn, event, days, tag):
    import datetime as dt
    day = dt.datetime.today().date()

    start = str((day - dt.timedelta(days=days)))

    query = """select 
       distinct user_id
       from won_sessions
       CROSS JOIN UNNEST(ad_events) AS t(ad_event) 
       where ad_event.ad_event_type = '""" + event +"""' 
        and dst = 'B0094' 
        and ad_event.ad_event_time >= timestamp '"""+ start +""" 00:08:00.000'""" 

    query = query.replace('\n', '')
    # pdb.set_trace()
    engagers = []
    for data in pd.read_sql(query, Athconn, chunksize  = 1000):
    #    pdb.set_trace()
        engagers.append(data)
    engagers = pd.concat(engagers)

    import beeswax_api
    from beeswax_api import models
    from beeswax_api import SessionFactory
    from beeswax_api.models import Session

    sesh = SessionFactory.SessionFactory(email = 'jrmccrery@tresensa.com', password = 'Kestrel1', 
                                  url = 'https://tresensa.api.beeswax.com')

    def dirMaker(x):
        d = '/home/jr/data/beeswax/' + x +'/'+str(dt.datetime.now().date())
        if not os.path.exists(d):
            os.makedirs(d)
        print(d)
        os.chdir(d)

    dirMaker(event + '_'+ str(days) +'D')
    #flist_ = pd.DataFrame(engagers['user_id'].unique())
    engagers['user_id'] = engagers['user_id'] + '|tresensa-' + str(tag)
    engagers.to_csv('deleteMe.csv', index = None,  header = False)

    z = 0
    for i in pd.read_csv('deleteMe.csv', chunksize = 1000*180):
        z += 1
        i.to_csv(str(z) + 'upload.csv', index = False)

    os.remove('deleteMe.csv')

    #pdb.set_trace()

    uploader = sesh.login()
    files = glob.glob('*')
    for i in files:
        uploader.segment_upload_local_file(i)

def main():
    masterKey = {}
    masterKey[0] = {}
    masterKey[0]['event']  = 'engagement'
    masterKey[0]['days']  = 1
    masterKey[0]['tag']  = 267
    masterKey[1] = {}
    masterKey[1]['event']  = 'engagement'
    masterKey[1]['days']  = 30
    masterKey[1]['tag']  = 63
    masterKey[2] = {}
    masterKey[2]['event']  = 'click_through'
    masterKey[2]['days']  = 30
    masterKey[2]['tag']  = 88
    masterKey[3] = {}
    masterKey[3]['event']  = 'app_install'
    masterKey[3]['days']  = 30
    masterKey[3]['tag']  = 89
    for i in masterKey.keys():
        mainFuntion(Athconn, masterKey[i]['event'], masterKey[i]['days'], masterKey[i]['tag'])

if __name__ == '__main__':
    main()