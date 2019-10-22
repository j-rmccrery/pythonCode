
"""
Simple email report
"""

from pyathenajdbc import connect
import pandas as pd
import pdb
import sys
import pickle 
import os, glob
from scipy.stats import beta
import datetime as dt

Athconn = connect(access_key='AKIAII2DPKKZAUGC6KQA',
               secret_key='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',
               s3_staging_dir='s3://analytics.tresensa.com/athena/dev/staging/python_test/',
               region_name='us-east-1')  

date = dt.datetime.now().date()

def dirMaker(date):
    d = '/home/jr/data/reporting/Andrea/' + str(date)
    if not os.path.exists(d):
        os.makedirs(d)
    os.chdir(d)

dirMaker(date)

finish = str(dt.datetime.now().date() - dt.timedelta(days=1)).replace('-', '')
start = str(dt.datetime.now().date() - dt.timedelta(days=1)).replace('-', '')

impressions = []
query = """SELECT b.advertiser_name AS Advertiser,b.campaign_id AS CampaignID,b.campaign_name AS CampaignName, count(a.session_id) as Impressions, sum(win_price_micros) as Cost,
sum(rpm_micros) as Gross
FROM won_sessions a,campaign_info b 
    where a.d >= """ + start + """ 
    AND a.d <= """ + finish + """ 
    AND a.impression_time is not null 
    AND a.dst = b.dst 
    AND a.line_item_id = b.line_item_id 
GROUP BY  b.campaign_id, b.campaign_name, b.advertiser_name 
ORDER BY b.advertiser_name, count(a.session_id) DESC"""
query = query.replace('\n', '')
for data in pd.read_sql(query, Athconn, chunksize  = 1000):
    impressions.append(data)
impressions = pd.concat(impressions)
impressions['Net Revenue'] = (impressions['Gross'] - impressions['Cost']) / 1000 / 1000
impressions.to_csv(str(date) + '.csv', index = False)

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
msg.add_header('Subject', 'Campaign Activity' +' '+ str(finish))
server = smtplib.SMTP('smtp.gmail.com', 587)
server.set_debuglevel(False) # set to True for verbose output
server.starttls()
server.login(gmail_user,gmail_password)
server.sendmail(msg['From'], ['adibben@gmail.com', 'jrmccrery@gmail.com'], msg.as_string())
print ('Email sent')
server.quit() # bye bye
