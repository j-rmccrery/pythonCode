
# coding: utf-8
"""
Automated script for add campaign performance.  Connects to wombat database, sends the user an email if alarming day over day behavior is present
"""
import sys
import imaplib
import getpass
import email
import email.header
import datetime
import pdb
import os
import glob
import pandas as pd
import datetime as dt

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.mime.application import MIMEApplication
import smtplib
from mimetypes import guess_type
from email import encoders
    
def dirMaker(d):
    d = '/mnt/499f5b4c-132a-4775-a414-ce6eba504896/data/reporting/wombat/' + str(d)
    if not os.path.exists(d):
        os.makedirs(d)
    os.chdir(d)
    
import psycopg2

def connQuery( conn, query ) :
    cur = conn.cursor()
    cur.execute( query )
    x = cur.fetchall()
    return x

hostname='wombat2-prod-us-east-1a.cyjtxxcykiey.us-east-1.rds.amazonaws.com' 
username='wombadmin'
password='9kV48JHOMMwLWnJEhT6Y'
database='wombat_prod_db'

try:
    conn = psycopg2.connect(host=hostname, user=username, password=password, dbname=database)
except:
    print ("I am unable to connect to the database")
    
query = """select cost.date, cost.external_line_item_id, cost.publisher_bundle_id, cost.impressions, cost.cost, coalesce(rev.rev_num_events, 0) as installs,  
coalesce(rev.rev, 0) as revenue
from 
(SELECT (timeframe_start at time zone 'America/New_York')::date as date, 
  external_line_item_id, publisher_bundle_id,
  sum(num_events) as impressions, 
  sum(aggregate_value2) / 1000000 as cost  
FROM aggregate_event_time_event_data 
WHERE (timeframe_start at time zone 'America/New_York')::date >= (TIMESTAMP 'yesterday' at time zone 'America/New_York')::date 
and event_type_id = 2 
and external_line_item_id = '2617' 
and dst = 'B0094' 
and (timeframe_start at time zone 'America/New_York')::time < (SELECT (timeframe_start at time zone 'America/New_York')::time 
FROM aggregate_event_time_event_data 
WHERE (timeframe_start at time zone 'America/New_York')::date > (TIMESTAMP 'yesterday' at time zone 'America/New_York')::date 
order by timeframe_start DESC 
LIMIT 1) 
GROUP BY 
  (timeframe_start at time zone 'America/New_York')::date, 
  external_line_item_id, publisher_bundle_id) cost 
left join  
(SELECT (timeframe_start at time zone 'America/New_York')::date as date, 
  external_line_item_id, publisher_bundle_id,
  sum(num_events) as rev_num_events, 
  sum(aggregate_value1) / 1000000 as rev 
FROM aggregate_event_time_event_data 
WHERE (timeframe_start at time zone 'America/New_York')::date >= (TIMESTAMP 'yesterday' at time zone 'America/New_York')::date 
and event_type_id = 3 
and dst = 'B0094' 
and external_line_item_id = '2617' 
and (timeframe_start at time zone 'America/New_York')::time <  (SELECT (timeframe_start at time zone 'America/New_York')::time 
FROM aggregate_event_time_event_data 
WHERE (timeframe_start at time zone 'America/New_York')::date > (TIMESTAMP 'yesterday' at time zone 'America/New_York')::date 
order by timeframe_start DESC 
LIMIT 1) 
GROUP BY 
  (timeframe_start at time zone 'America/New_York')::date, 
  external_line_item_id, publisher_bundle_id) rev 
ON (cost.date = rev.date and  
    cost.external_line_item_id = rev.external_line_item_id and 
    cost.publisher_bundle_id = rev.publisher_bundle_id)""" 

data = connQuery(conn, query)
data = pd.DataFrame(data)
data.columns = ['date', 'line_item_id', 'app_bundle_id', 'impressions', 'cost', 'installs', 'revenue']
data['revenue'] = data['revenue'].astype(float)
data['cost'] = data['cost'].astype(float)
data['net'] = data['revenue'] - data['cost']

today = data[data.date == sorted(data.date.unique())[1]]
yest = data[data.date == sorted(data.date.unique())[0]]

sendEmail = 'no'
if today.impressions.sum() / yest.impressions.sum() < .3:
    sendEmail = 'yes'

if today.impressions.sum() < 20000:
    sendEmail = 'yes'

if sendEmail == 'yes':

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

    msg.add_header('From', gmail_user)
    msg.add_header('Subject', "2617 is broken")
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.set_debuglevel(True) # set to True for verbose output
    server.starttls()
    server.login(gmail_user,gmail_password)
    server.sendmail(msg['From'], ["jrmccrery@gmail.com"], msg.as_string())
    print ('Email sent')
    server.quit() # bye bye

