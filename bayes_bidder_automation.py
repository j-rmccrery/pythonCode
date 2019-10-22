
"""
In single run, collect historical data from cloud storage, prep for a bespoke baysian-esque factorization machine inspired linear model, train model, 
create files to be used in production bidding environment, and upload to cloud storage for retrieval 
"""

# coding: utf-8

import pdb
import sys
import pickle 
import os
import glob
import pdb 
import tempfile
import shutil
import math

import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd
import numpy as np

from scipy.stats import beta
from pyathenajdbc import connect
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from scipy.stats import beta
from scipy.stats import norm

def cDirMaker(x):
    d = '/mnt/499f5b4c-132a-4775-a414-ce6eba504896/data/bidderFiles/' + x +'/'+str(dt.datetime.now().date())
    if not os.path.exists(d):
        os.makedirs(d)
    print(d)
    os.chdir(d)

def dataPuller():
    data = []
    query = """select session_id, line_item_id, app_bundle_id, device, city, metro, utc_offset, bid_request_time, country,
    segment_ids from won_sessions 
    where campaign_id in (""" + campaign + """) 
    and line_item_id in (""" + lines + """) 
    and dst = 'B0094' 
    and impression_time is not null 
    and d >= """ + start + """ 
    and d <= """ + finish + """ 
    limit 250000 """ 
    query = query.replace('\n', '')
    for data_ in pd.read_sql(query, Athconn, chunksize  = 1000):
        data.append(data_)
    data = pd.concat(data)

    query = """select session_id from won_sessions 
    CROSS JOIN UNNEST(ad_events) AS t(ad_event) 
    where ad_event.ad_event_type = '""" + target + """'
    and campaign_id in (""" + campaign + """) 
    and line_item_id in (""" + lines + """) 
    and dst = 'B0094' 
    and d >= """ + start + """ 
    and d <= """ + finish  
    query = query.replace('\n', '')
    installs = []
    for data_ in pd.read_sql(query, Athconn, chunksize  = 1000):
        installs.append(data_)
    installs = pd.concat(installs)

    dat_ = data[data.session_id.isin(installs.session_id.unique())]
    dat_['install'] = 1
    _dat = data[~data.session_id.isin(installs.session_id.unique())]
    _dat['install'] = 0
    data = pd.concat([_dat, dat_])
    _dat = None
    dat_ = None
    data = data.sample(len(data)).reset_index(drop = True)

    data['weekday'] = data['bid_request_time'].apply(lambda x: x.weekday())
    data['hour'] = data['bid_request_time'].apply(lambda x: x.hour)
    data = data.drop('bid_request_time', axis = 1)
    data = data.fillna('0')
    for i in ['app_bundle_id', 'city', 'weekday', 'hour', 'metro', 'carrier', 'line_item_id']:
        try:
            data[i] = data[i].astype(str)  
            data[i] = data[i].astype(float)
            data[i] = data[i].astype(int)    
            data[i] = data[i].astype(str)    
        except:
            continue

    return data

def importancer(data):
    variables = []
    for i in ['app_bundle_id', 'device', 'city', 'line_item_id',
           'metro', 'segment_ids','weekday',
           'hour']:
        variables.append(i)
        temp = data[[i, 'install']].groupby(i).count()
        vals = temp[temp.install > 500].index.values
        temp = data[data[i].isin(vals)][[i, 'install']].groupby(i).mean().replace(0, .001)
        temp_ = temp.reset_index()
        temp_.columns = [i, i + '_instRate']
        data = pd.merge(data, temp_, on = i, how = 'left')
    rateVars = [x for x in data.columns if '_inst' in x]
    pca_weights = pd.DataFrame(variables, np.linalg.svd(data[rateVars].fillna(0), compute_uv = False)).reset_index()
    pca_weights.columns = ['principle component', 'variable']

    forest = ExtraTreesClassifier(n_estimators=100,
                                  random_state=1)

    forest.fit(data[rateVars].fillna(.001), data['install'])
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)

    indices = np.argsort(importances)[::-1]
    realNames = []
    for i in indices:
        realNames.append(data[rateVars].columns[i])
    print("Feature ranking:")
    weights = {}
    for f in range(data[rateVars].shape[1]):
        print(rateVars[indices[f]])
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
        weights[rateVars[indices[f]]] = importances[indices[f]]

    f = plt.figure()

    plt.title("Feature importances")
    plt.bar(range(data[rateVars].shape[1]), importances[indices],
           color="r", yerr=std[indices], align="center")
    plt.xticks(range(data[rateVars].shape[1]), realNames, rotation=70)
    plt.xlim([-1, data[rateVars].shape[1]])

    f.savefig("Features.pdf", bbox_inches='tight')

    for i in weights.keys():
        weights[i.split('_instRate')[0]] = weights.pop(i)
    for i in weights.keys():
        weights[i.split('_instRate')[0]] = weights.pop(i)
    for i in weights.keys():
        if i in changer.keys():
            weights[changer[i]] = weights.pop(i)

    return weights

def dirMaker(x):
    d = '/mnt/499f5b4c-132a-4775-a414-ce6eba504896/data/bidderFilesBackup_/' + x +'/'+str(dt.datetime.now().date())
    if not os.path.exists(d):
        os.makedirs(d)
    print(d)
    os.chdir(d)

def backupFiles():
    for file_ in file_names:
        try:
            filName = file_ + str(campaign_id) + '.pickle'
            print(filName)
            s3.meta.client.download_file('beeswax-bidder-models-tresensa-com', 'new_model_test/' + filName, filName)
        except:
            filName = file_ + str(87) + '.pickle'
            print(filName)
            s3.meta.client.download_file('beeswax-bidder-models-tresensa-com', 'new_model_test/' + filName, filName)

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
    return fileHolder

def redisCleaner(redisDataRecent, redisData):
    for var in redisDataRecent.keys():
        toAdd = list(redisDataRecent[var].keys())
        for z in toAdd:
            redisData[var][z] = redisDataRecent[var].pop(z)
    for i in ['dayOfWeek', 'hour']:
        for z in redisData[i].keys():
            redisData[i][str(z)] = redisData[i].pop(z)
    for i in ['dayOfWeek', 'hour']:
        for z in redisData[i].keys():
            redisData[i][str(z)] = redisData[i].pop(z)
    for i in redisData['dayOfWeek'].keys():
        redisData['dayOfWeek'][i][1] = .000001
    for i in redisData['hour'].keys():
        redisData['hour'][i][1] = .000001
    return redisData

def segmenter(x):
    try:
        return str(sorted([x for x in x.replace(']', ', ').replace('[', '').replace('-', ', ').split(', ') 
                    if x != 'tresensa' and x != ''])    ).replace("', '", ',')[2::][:-2]
    except:
        return x

def referenceRate():
    impressions_ = pd.read_sql("""SELECT count(session_id) as impressions 
            FROM won_sessions 
            WHERE line_item_id in (""" + lines + """) 
            and campaign_id in (""" + campaign + """) 
            and dst = 'B0094' 
            and d >= """ + start + """ 
            and d <= """ + finish + """ """, Athconn)['impressions'][0]

    installs_ = pd.read_sql("""SELECT count(session_id) as installs 
            FROM won_sessions 
            CROSS JOIN UNNEST(ad_events) AS t(ad_event) 
            WHERE ad_event.ad_event_type = '""" + target + """' 
            AND line_item_id in (""" + lines + """) 
            and campaign_id in (""" + campaign + """) 
            and dst = 'B0094' 
            and d >= """ + start + """ 
            and d <= """ + finish + """ """, Athconn)['installs'][0]
    overallInstRate = installs_/impressions_
    return overallInstRate

def dataRun(data, overallInstRate, weights, redisData):
    fullCont = []
    testang = data[variables].reset_index(drop = True)
    for i in testang.iterrows():
        probabilityContainer = []
        for z in i[1].keys():
            if z in ['session_id', 'install', 'line_item_id']:
                continue
            try:
                mean, variance = redisData[z][str(i[1][z])]
            except:
                mean, variance = overallInstRate, .001
            dist = norm(mean, math.sqrt( variance ))
            x = dist.cdf(overallInstRate) 
            x = 1 - x
            if x == 0:
                x = -1 * math.log1p(overallInstRate / mean)
                if x < -1:
                    x = -1
            modifier = weights[z]
            scaledProbability = modifier * x
            probabilityContainer.append(scaledProbability)
        probabilityContainer.append(i[1]['session_id'])
        probabilityContainer.append(i[1]['install'])
        fullCont.append(probabilityContainer)
    final = pd.DataFrame(fullCont)
    final.columns = [ 'publisherBundleId',
     'device',
     'segments',
     'metro',
     'city',
     'dayOfWeek',
     'hour',
     'session_id',
     'install']

    final['gamma'] = final.apply(lambda x: x.publisherBundleId + x.device + x.city + x.segments + x.metro + x.dayOfWeek + x.hour, axis = 1)

    final = final.sort_values('gamma').reset_index(drop = True)

    final = final.dropna()
    return final 

def prepper():
    campaign = sys.argv[1]
    lines = sys.argv[2]
    target = sys.argv[3]

    finish = str(dt.datetime.now().date() - dt.timedelta(days=3)).replace('-', '')
    start = str(dt.datetime.now().date() - dt.timedelta(days=13)).replace('-', '')

    Athconn = connect(access_key='xxx',
                   secret_key='xxxx',
                   s3_staging_dir='xxx',
                   region_name='xxx')

    import boto3
    key =  'xxx'
    secret =  'xxx'

    s3 = boto3.resource(
        's3',
        aws_access_key_id=key,
        aws_secret_access_key=secret)

    campaign_id = pd.read_sql('SELECT distinct campaign_id FROM campaign_info WHERE line_item_id in (' + lines + ')', Athconn)['campaign_id'][0]
    return campaign_id

def modeler():
    inputVars = {}
    inputVars['campaign'] = int(campaign.replace("'", ""))
    file_names = ['coordinates',
     'installRate',
     'meanAndVariance',
     'scalingFactors']

    variables = ['publisherBundleId',
         'device',
         'segments',
         'metro',
         'city',
         'dayOfWeek',
         'hour',
         'session_id',
         'install']

    changer = {}
    changer['app_bundle_id'] = 'publisherBundleId'
    changer['weekday'] = 'dayOfWeek'
    changer['zip_code'] = 'zipCode'
    changer['segment_ids'] = 'segments'

    campaign_id = prepper()
    data = dataPuller()
    cDirMaker(campaign)
    weights = importancer(data)
    
    for i in changer.keys():
        data = data.rename(columns={i:changer[i]})
    data['segments'] = data['segments'].apply(lambda x: segmenter(x))

    dirMaker(str(inputVars['campaign']))
    fileHolder = backupFiles()
    os.chdir('/mnt/499f5b4c-132a-4775-a414-ce6eba504896/data/modelData/'+ campaign +'/' + str(dt.datetime.now().date()))
    with open('redisData.pickle', 'rb') as handle:
        redisData = pickle.load(handle)
    with open('redisDataRecent.pickle', 'rb') as handle:
        redisDataRecent = pickle.load(handle)
    redisData = redisCleaner(redisDataRecent, redisData)
    overallInstRate = referenceRate()

    final = dataRun(data, overallInstRate, weights, redisData)
    cDirMaker(campaign)

    toPlot = []
    for i in np.array_split(final, 30):
        toPlot.append([i['gamma'].mean(), i['install'].mean()])
    toPlot = pd.DataFrame(toPlot)
    p = np.poly1d(np.polyfit(toPlot[0].values, toPlot[1].values, 3))
    f = plt.figure()
    plt.plot(toPlot[0].values, toPlot[1])
    plt.plot(toPlot[0].values, p(toPlot[0]), "o")
    plt.xlabel('model output')
    plt.ylabel('realized completion rate')
    plt.grid()
    f.savefig("Performance.pdf", bbox_inches='tight')

    with open('meanAndVariance' + str(campaign_id)+'.pickle', 'wb') as handle:
            pickle.dump(redisData, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('installRate' + str(campaign_id)+ '.pickle', 'wb') as handle:
            pickle.dump(overallInstRate, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('coordinates' + str(campaign_id)+ '.pickle', 'wb') as handle:
            pickle.dump(toPlot, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('scalingFactors' + str(campaign_id)+ '.pickle', 'wb') as handle:  
        try:
            pickle.dump(pd.DataFrame(weights), handle, protocol=pickle.HIGHEST_PROTOCOL)
        except:
            pickle.dump(pd.DataFrame(weights, index=[0]), handle, protocol=pickle.HIGHEST_PROTOCOL)

    files = glob.glob('*')

    import boto3
    key =  'xxxx'
    secret =  'xxxx'

    s3 = boto3.resource(
        's3',
        aws_access_key_id=key,
        aws_secret_access_key=secret)

    for file_ in files:
        s3.meta.client.upload_file(os.getcwd() +'/'+file_, 'custom-model.tresensa.com', file_)