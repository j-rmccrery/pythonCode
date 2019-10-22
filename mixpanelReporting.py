
"""
Get reporting from mixpanel for in product ad campaign and email results 
"""

# coding: utf-8

from mixpanel_query.client import MixpanelQueryClient
MIXPANEL_API_KEY, MIXPANEL_API_SECRET = 'xxx', 'xxx'
MP = MixpanelQueryClient(MIXPANEL_API_KEY, MIXPANEL_API_SECRET)


import pandas as pd
import datetime as dt
import pdb

campaign = 'BudLight'
keys = {}
keys['iOS, Mobile'] = [2880,2881,2882,2884,2883,2885,2879,2878,2877,2846,2847,2848,2871,2870,2872,2845,2844,2843]
keys['iOS, Tablet'] = [2891,2890,2889,2888,2887,2886,2854,2853,2852,2851,2850,2849]
keys['Android'] = [2900,2899,2898,2897,2896,2895,2894,2893,2892,2860,2859,2858,2875,2874,2873,2857,2856,2855]

finish = dt.datetime.now().date() - dt.timedelta(days=1)
start = dt.datetime.now().date() - dt.timedelta(days=10)

variables = ['game visible',
'ageGate',
'engagement',
'engagement secondary',
'budLight',
'charge',
'opponent','mead',
'score',
'scoreLost',
'gameOver',
'screen',
'replay',
'clickthru',
'video',
'heartbeat']
splitters = {}
splitters['ageGate'] = 'value'
splitters['budLight'] = 'value'
splitters['screen'] = 'reason'
splitters['gameOver'] = 'value'
splitters['clickthru'] = 'source'
splitters['video'] = 'value'

container = []
for seg_ in keys.keys():
    for seg in keys[seg_]:
        for var in variables:
            print (seg)
            if var in splitters.keys():
                temp = MP.get_segmentation(event_name = var, start_date = str(start), end_date = str(finish), 
                            where = 'properties["line_item_id"] == "'+ str(seg) +'"', 
                                       on = 'properties["'+ splitters[var] +'"]', data_type = 'general')
                for sub in temp['data']['values'].keys():
                    temp_ = pd.DataFrame.from_dict([temp['data']['values'][sub]]).transpose()
                    temp_.columns = [sub]
                    temp_['device'] = seg_
                    temp_['line'] = seg
                    container.append(temp_)
            else:
                temp = MP.get_segmentation(event_name = var, start_date = str(start), end_date = str(finish), 
                                where = 'properties["line_item_id"] == "'+ str(seg) +'"', data_type = 'general')
                temp = pd.DataFrame(temp['data']['values'])
                temp['device'] = seg_
                temp['line'] = seg
                container.append(temp)

total = pd.concat(container).fillna(0).reset_index()
total = total.rename(columns={"index": "date"})


import os
def dirMaker(x):
    d = '/home/jr/data/reporting/'+ campaign +'/' + str(x)
    if not os.path.exists(d):
        os.makedirs(d)
    os.chdir(d)
dirMaker(finish)

devices = total.groupby(['date', 'device']).sum().reset_index()
devices = devices[['date', 'device', 'game visible', 'engagement',
       'engagement secondary',  'kartA', 'kartB','default', 'marble', 'domino', 'dice', 'laneswitch', 'boost', 'win', 'lose_2nd_place', 'lose_damage', 'replay', 'button', 'endscreen', 'heartbeat']]
devices['heartbeat'] = devices['heartbeat'] * 10

devices.to_csv(campaign+'_' +str(finish)+ '.csv')

import glob
files = glob.glob('*')

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.mime.application import MIMEApplication
import smtplib
from mimetypes import guess_type
from email import encoders
msg = MIMEMultipart()
gmail_user = "jrmccrery@gmail.com"
gmail_password = 'rfysgotfnitsmdar'
for filename in files:
    f = filename
    part = MIMEBase('application', "octet-stream")
    part.set_payload( open(f,"rb").read() )
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', 'attachment; filename="%s"' % os.path.basename(f))
    msg.attach(part)

msg.add_header('From', gmail_user)
msg.add_header('Subject', campaign +'_'+ str(finish))
server = smtplib.SMTP('smtp.gmail.com', 587)
server.set_debuglevel(True) # set to True for verbose output
server.starttls()
server.login(gmail_user,gmail_password)
server.sendmail(msg['From'], ["jrmccrery@gmail.com"], msg.as_string())
print ('Email sent')
server.quit() # bye bye

