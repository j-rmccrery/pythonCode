"""
connect to Redis database and update files per some standard naming nomanclature
"""

import pandas as pd
import pickle, redis, glob, os, sys
import pdb 
import tempfile
import shutil
cwd = os.getcwd()
inputVars = {}
inputVars['redisHost'] = sys.argv[1]
inputVars['redisPort'] = sys.argv[2]
inputVars['campaign'] = sys.argv[3]
inputVars['updateDay'] = sys.argv[4]

import boto3
key =  'xxx'
secret =  'xxx'

s3 = boto3.resource(
    's3',
    aws_access_key_id=key,
    aws_secret_access_key=secret,
)

file_names = ['coordinates',
 'installRate',
 'meanAndVariance',
 'scalingFactors']

dirpath = tempfile.mkdtemp()

def directoryMaker(campaign, date):
    name = str(date) + '.csv'
    ### will need to change this to the directory you want the files downloaded to.  Make the same changed to pickleGetter
    directory = 'D:\\bidderFiles\\' + str(campaign) + '\\' + str(date) + '\\'
    if not os.path.exists(directory):
        os.makedirs(directory)
directoryMaker(inputVars['campaign'], inputVars['updateDay'])

os.chdir(dirpath)
for file_ in file_names:
    filName = file_ + str(inputVars['campaign']) + '.pickle'
    print(filName)
    s3.meta.client.download_file('beeswax-bidder-models-tresensa-com', 'new_model_test/' + filName, filName)

def pickleGetter(campaign, date):
    directory = 'D:\\bidderFiles\\' + str(campaign) + '\\' + str(date) + '\\'
    os.chdir(directory)

pickleGetter(inputVars['campaign'], inputVars['updateDay'])

files = glob.glob('*')

example = [x for x in files if file_names[0] in x][0]
to_split = example.split(file_names[0])[1]

fileHolder = {}
for file_ in files:
    name = file_.split(to_split)[0]
    if name in ['coordinates', 'scalingFactors']:
        unpickledFile = pd.read_pickle(file_)
        fileHolder[name] = unpickledFile
    else:
        with open(file_, 'rb') as handle:
            unpickledFile = pickle.load(handle)
        fileHolder[name] = unpickledFile
conn = redis.Redis(inputVars['redisHost'], port=inputVars['redisPort'])
for key in fileHolder.keys():
    print(key)
    name = 'model:'+ str(inputVars['campaign']) +':' + key
    if 'coordinates' in key:
        conn.set(name, fileHolder[key].values.tolist()) 
    elif 'installRate' in key:
        conn.set(name, fileHolder[key])
    elif 'meanAndVariance' in key:
        for key_ in fileHolder[key].keys():
            name = 'model:'+ str(inputVars['campaign']) +':meanAndVariance:' + key_
            print(key_)
            conn.hmset(name, fileHolder[key][key_]) 
    else:
        pdb.set_trace()
        holder = {}
        for scal in fileHolder[key].keys():
            holder[scal] = fileHolder[key][scal][0]
        conn.hmset(name, holder) 
os.chdir(cwd)
shutil.rmtree(dirpath)
