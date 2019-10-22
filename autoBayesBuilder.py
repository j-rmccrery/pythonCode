
"""
In single run, collect historical data from cloud storage, prep for a bespoke baysian-esque factorization machine inspired linear model, train model, 
create files to be used in production bidding environment, and upload to cloud storage for retrieval 
"""

# coding: utf-8

from pyathenajdbc import connect
import pdb
import sys
import pickle 
import os
from scipy.stats import beta
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import shutil
import pandas as pd

def compiler(campaign_, lines_, internalID):
    def dirMaker(x):
        d = '/home/jr/data/bidderFiles/' + x +'/'+str(dt.datetime.now().date())
        if not os.path.exists(d):
            os.makedirs(d)
        print(d)
        os.chdir(d)
    dirMaker(internalID)
    campaign = campaign_
    lines = lines_
    target = 'app_install'

    Athconn = connect(access_key='XXXX',
                   secret_key='XXXX',
                   s3_staging_dir='XXXX',
                   region_name='XXXX')

    finish = str(dt.datetime.now().date() - dt.timedelta(days=2)).replace('-', '')
    start = str(dt.datetime.now().date() - dt.timedelta(days=8)).replace('-', '')

    data = []
    query = """select session_id, line_item_id, app_bundle_id, device, city, metro, utc_offset, bid_request_time, country,
    segment_ids from won_sessions 
    where campaign_id in (""" + campaign + """) 
    and line_item_id in (""" + lines + """) 
    and dst = 'B0094' 
    and d >= """ + start + """ 
    and d <= """ + finish + """ 
    and impression_time is not null 
    limit 250000""" 
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

    for i in ['app_bundle_id', 'device', 'city', 'line_item_id',
           'metro', 'segment_ids','weekday',
           'hour']:
        print(i)
        temp = data[[i, 'install']].groupby(i).count()
        vals = temp[temp.install > 600].index.values
        temp = data[data[i].isin(vals)][[i, 'install']].groupby(i).mean().replace(0, .001)
        temp_ = temp.reset_index()
        temp_.columns = [i, i + '_instRate']
        data = pd.merge(data, temp_, on = i, how = 'left')
    rateVars = [x for x in data.columns if '_inst' in x]
    # np.linalg.svd(data[rateVars].fillna(0), compute_uv = False)
    def dirMaker(x):
        d = '/home/jr/data/bidderFiles/' + x +'/'+str(dt.datetime.now().date())
        if not os.path.exists(d):
            os.makedirs(d)
        print(d)
        os.chdir(d)
    dirMaker(internalID)

    from sklearn.datasets import make_classification
    from sklearn.ensemble import ExtraTreesClassifier

    forest = ExtraTreesClassifier(n_estimators=100,
                                  random_state=1)

    forest.fit(data[rateVars].fillna(.001), data['install'])
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)

    indices = np.argsort(importances)[::-1]

    print("Feature ranking:")
    weights = {}
    for f in range(data[rateVars].shape[1]):
        print(rateVars[indices[f]])
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
        weights[rateVars[indices[f]]] = importances[indices[f]]
        plt.figure()
    plt.title("Feature importances")
    plt.bar(range(data[rateVars].shape[1]), importances[indices],
           color="r", yerr=std[indices], align="center")
    plt.xticks(range(data[rateVars].shape[1]), indices)
    plt.xlim([-1, data[rateVars].shape[1]])
    plt.savefig('variableImportance.pdf')
    plt.clf()
    plt.close()

    cwd = os.getcwd()

    import boto3
    key =  'XXX'
    secret =  'XXX'

    s3 = boto3.resource(
        's3',
        aws_access_key_id=key,
        aws_secret_access_key=secret)

    file_names = ['coordinates',
     'installRate',
     'meanAndVariance',
     'scalingFactors']

    def dirMaker(x):
        d = '/home/jr/data/bidderFilesBackup/' + x +'/'+str(dt.datetime.now().date())
        if not os.path.exists(d):
            os.makedirs(d)
        print(d)
        os.chdir(d)

    dirMaker(internalID)
    for file_ in file_names:
        filName = file_ + internalID + '.pickle'
        print(filName)
        s3.meta.client.download_file('beeswax-bidder-models-tresensa-com', 'new_model_test/' + filName, filName)
    import glob
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

    from scipy.stats import beta
    from scipy.stats import norm
    import math

    os.chdir('/home/jr/data/modelData/'+ internalID +'/' + str(dt.datetime.now().date()))
    with open('redisData.pickle', 'rb') as handle:
        redisData = pickle.load(handle)
    with open('redisDataRecent.pickle', 'rb') as handle:
        redisDataRecent = pickle.load(handle)

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

    impressions_ = pd.read_sql("""SELECT count(session_id) as impressions 
            FROM won_sessions 
            WHERE line_item_id in (""" + lines + """) 
            and campaign_id in (""" + campaign + """) 
            and device_type = 'Phone' 
            and dst = 'B0094' 
            and d >= """ + start + """ 
            and d <= """ + finish + """ 
            AND impression_time is NOT null""", Athconn)['impressions'][0]

    installs_ = pd.read_sql("""SELECT count(session_id) as installs 
    FROM won_sessions 
    CROSS JOIN UNNEST(ad_events) AS t(ad_event) 
            WHERE ad_event.ad_event_type = '""" + target + """' 
            AND line_item_id in (""" + lines + """) 
            and campaign_id in (""" + campaign + """) 
            and device_type = 'Phone' 
            and dst = 'B0094' 
            and d >= """ + start + """ 
            and d <= """ + finish + """ 
            AND impression_time is NOT null""", Athconn)['installs'][0]
    overallInstRate = installs_/impressions_

    weights_ = fileHolder['scalingFactors'].to_dict()

    changer = {}
    changer['app_bundle_id'] = 'publisherBundleId'
    changer['weekday'] = 'dayOfWeek'
    changer['zip_code'] = 'zipCode'
    changer['segment_ids'] = 'segments'
    for i in changer.keys():
        data = data.rename(columns={i:changer[i]})

    def segmenter(x):
        try:
            return str(sorted([x for x in x.replace(']', ', ').replace('[', '').replace('-', ', ').split(', ') 
                        if x != 'tresensa' and x != ''])    ).replace("', '", ',')[2::][:-2]
        except:
            return x
    data['segments'] = data['segments'].apply(lambda x: segmenter(x))

    variables = ['publisherBundleId',
     'device',
     'segments',
     'metro',
     'city',
     'dayOfWeek',
     'hour',
     'session_id',
     'install']

    for i in weights.keys():
        weights[i.split('_instRate')[0]] = weights.pop(i)
    for i in weights.keys():
        weights[i.split('_instRate')[0]] = weights.pop(i)
    for i in weights.keys():
        if i in changer.keys():
            weights[changer[i]] = weights.pop(i)

    for i in redisData['dayOfWeek'].keys():
        redisData['dayOfWeek'][i][1] = .000001

    weights_['line_item_id'] = {}
    weights_['line_item_id'][0] = 1 

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
            modifier = weights_[z][0]
            scaledProbability = modifier * x
            probabilityContainer.append(scaledProbability)
        probabilityContainer.append(i[1]['session_id'])
    #     probabilityContainer.append(i[1]['line_item_id'])
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

    final['gamma'] = final.apply(lambda x: x.publisherBundleId + x.device +x.city+ x.segments  +x.metro+
                                 x.dayOfWeek + x.hour, axis = 1)
    final = final.sort_values('gamma').reset_index(drop = True)

    print(final.install.mean())

    def dirMaker(x):
        d = '/home/jr/data/bidderFiles/' + x +'/'+str(dt.datetime.now().date())
        if not os.path.exists(d):
            os.makedirs(d)
        print(d)
        os.chdir(d)
    dirMaker(internalID)

    toPlot = []
    for i in np.array_split(final, 30):
        toPlot.append([i['gamma'].mean(), i['install'].mean()])
    toPlot = pd.DataFrame(toPlot)
    p = np.poly1d(np.polyfit(toPlot[0].values, toPlot[1].values, 3))
    # p = np.poly1d(np.polyfit(fileHolder['coordinates'][0].values, fileHolder['coordinates'][1].values, 3))

    plt.plot(toPlot[0].values, toPlot[1])
    plt.plot(toPlot[0].values, p(toPlot[0]), "o")
    # plt.plot(toPlot[0].values, toPlot[2], "r")

    # plt.title('tripeaks android device whitelist')
    plt.xlabel('model output')
    plt.ylabel('realized eng rate')
    plt.grid()
    plt.savefig('modelBacktest.pdf')
    from sklearn import linear_model
    clf = linear_model.LinearRegression()
    clf.fit(final[['gamma']].values, final['install'].values)

    final = pd.merge(data[['session_id', 'line_item_id']], final, on  = 'session_id') 
    final.to_csv('outputData.csv')

    with open('meanAndVariance' + internalID+'.pickle', 'wb') as handle:
            pickle.dump(redisData, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('installRate' + internalID+'.pickle', 'wb') as handle:
            pickle.dump(overallInstRate, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('coordinates' + internalID+'.pickle', 'wb') as handle:
            pickle.dump(toPlot, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('scalingFactors' + internalID+'.pickle', 'wb') as handle:
            pickle.dump(pd.DataFrame(weights_), handle, protocol=pickle.HIGHEST_PROTOCOL)

    print (os.getcwd())
    import glob
    files = glob.glob('*.pickle*')
    for file_ in files:
        s3.meta.client.upload_file(os.getcwd() +'/'+file_, 'XXXX', 
                                   'new_model_test/' + file_)

