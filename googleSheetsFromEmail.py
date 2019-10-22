
"""
Scrapes email for files of interest and updates a google sheet cell by cell with the new information
"""

# coding: utf-8

import datetime 
import sys
import imaplib
import getpass
import email
import email.header
import os, glob, pdb

gmail_user = "jrmccrery@gmail.com"
gmail_password = 'rfysgotfnitsmdar'

EMAIL_ACCOUNT = gmail_user
EMAIL_FOLDER = gmail_password

M = imaplib.IMAP4_SSL('imap.gmail.com')

try:
    rv, data = M.login(EMAIL_ACCOUNT, gmail_password)
except imaplib.IMAP4.error:
    print ("LOGIN FAILED!!! ")

def dirMaker():
    d = '/home/jr/data/reporting/ChoctawEmails/'+str(datetime.datetime.now().date())
    print(d)
    if not os.path.exists(d):
        os.makedirs(d)
    else:
        files = glob.glob('*')
        for fi in files:
            os.remove(fi)
    os.chdir(d)
dirMaker()


# In[140]:


def process_mailbox(M):
    """
    Do something with emails messages in the folder.  
    For the sake of this example, print some headers.
    """

    rv, data = M.search(None, 'SUBJECT', "Choctaw")
    if rv != 'OK':
        print("No messages found!")
        return
    msgs = []
    #pdb.set_trace()
    for file in data[0].split()[-1:]:
        for num in file.split():
            rv, data_ = M.fetch(num, '(RFC822)')
            if rv != 'OK':
                print("ERROR getting message", num)
                return
            msg = email.message_from_bytes(data_[0][1])
            hdr = email.header.make_header(email.header.decode_header(msg['Subject']))
            sndr = email.header.make_header(email.header.decode_header(msg['From']))
            sender = str(sndr)
            subject = str(hdr)
            if 'Choctaw' in subject and '@gsngames.com' in sender:
                for part in msg.walk():
                    if 'application' in part.get_content_type():
                        open('attachment '+ ' '.join(msg['Date'].split(' ')[0:3]) +'.csv', 'wb').write(part.get_payload(decode=True))
M.select("INBOX")
process_mailbox(M)

rv, data = M.search(None, 'SUBJECT', "Choctaw")
msgs = []
for file in data[0].split()[-1:]:
    for num in file.split():
        rv, data_ = M.fetch(num, '(RFC822)')
        msg = email.message_from_bytes(data_[0][1])
        hdr = email.header.make_header(email.header.decode_header(msg['Subject']))
        sndr = email.header.make_header(email.header.decode_header(msg['From']))
        sender = str(sndr)
        subject = str(hdr)
        if 'Choctaw' in subject and '@gsngames.com' in sender:
            attachments = msg.get_payload()
            for part in attachments:
                if 'application' in part.get_content_type():
                        print(part.get_filename())
                        open(part.get_filename(), 'wb').write(part.get_payload(decode=True))

geos = ['Grant', 'Durant', 'Pocola']
def dater(x):
    try:
        return datetime.datetime.strptime(lastD, "%m/%d/%Y").date()
    except:
        return datetime.datetime.strptime(lastD, "%Y-%m-%d").date()

import pandas as pd
import glob
files = glob.glob('*')

import gspread
from oauth2client.service_account import ServiceAccountCredentials
scopes = ['https://spreadsheets.google.com/feeds/', 'https://www.googleapis.com/auth/drive']
# credentials = ServiceAccountCredentials.from_json_keyfile_name('/home/jr/Downloads/My Project 27844-1554ae78ca54.json', scopes)
credentials = ServiceAccountCredentials.from_json_keyfile_name('/home/jr/Downloads/My Project 27844-0a47274ee88f.json', scopes)

gc = gspread.authorize(credentials)
table = gc.openall()
pdb.set_trace()
choc = gc.open_by_key(table[0].id)

# In[273]:
import pdb
#pdb.set_trace()
for geo in geos:
    #pdb.set_trace()
    sh = 'Sizmek, ' + geo
    sheet = choc.worksheet(sh)
    choctawColumns = ['date', 'Interactions', 'Clicks', 'CTR']
    SheetValues = pd.DataFrame(sheet.get_all_values())
    SheetValues.columns = choctawColumns
    lastD = SheetValues.tail(1).date.unique()[0]
    cell_ = sheet.find(lastD)       
    #pdb.set_trace()
    Email = pd.read_excel([x for x in files if geo in x][0]).drop('Placement Name', axis = 1)    
    toAdd = Email[Email.Date > dater(lastD)]
    
    toAdd.columns = choctawColumns
    toAdd['date'] = toAdd['date'].apply(lambda x: datetime.datetime.strftime(x.date(), '%m/%d/%Y'))
    toAdd['CTR'] = toAdd['CTR'] * 100

    for i in range(0, len(toAdd.values)):
        row = i + 1 + cell_.row
        for z in range(0, len(choctawColumns)):
            col = z + 1
            sheet.update_cell(row, col, toAdd.values[i][z])