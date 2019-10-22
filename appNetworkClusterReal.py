

"""
Takes a history of app visits by user, and creates community clusters by running a hierarchical clustering model 
on a conditional probability matrix of app connecitons
"""

# coding: utf-8
import subprocess
import json
import pdb
import os

import pandas as pd
import scipy.sparse as ss
import numpy as np
import datetime as dt

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.base import BaseEstimator
from sklearn.utils import check_array
from os.path import isfile
from itertools import permutations

files = [
    # "9b88ae09-9b28-40c3-ae1b-323129355c95",  ### app histories, no other columns  all 2018
    # "b5c1d650-5595-4a95-92c7-ac12eec47401",  ### app histories, no other columns  > 20180500
    "58fba123-ee58-4fa5-b8a5-8d94be579593",  ### app histories, no other columns  all time data

    # "9da29fc5-f014-4319-a7f9-21d7169e98dd"  ### app histories > 20180600, impressions, installs > 20180600
    "32fba8f8-2901-40ef-8cf1-e3ce8ffbf24b"  ### all app histories, impressions, installs > 20180600
]

def dirMaker(x):
    d = '/mnt/499f5b4c-132a-4775-a414-ce6eba504896/data/clustering/' + x +'/'+str(dt.datetime.now().date())
    if not os.path.exists(d):
        os.makedirs(d)
    print(d)
    os.chdir(d)
    return d

dirMaker('placeholder')

appDict = {}
data = pd.read_parquet("/mnt/499f5b4c-132a-4775-a414-ce6eba504896/Parquet/"+files[0]+".parquet")[['appHistory']]

def permutator(x, appDict):
    apps = x.replace('[', '').replace(']', '').replace(' ', '').split(',')
    for p in permutations(apps, 2):
        if p in appDict.keys():
            appDict[p] += 1
        else:
            appDict[p] = 1

for i in data.appHistory:
    permutator(i, appDict)
for i in list(appDict):
    if appDict[i] < 15:
        appDict.pop(i)

data = pd.DataFrame.from_dict(appDict, orient='index').reset_index()

data = data.dropna()

def columner(x, side):
    return x[side]

data['t1_app'] = data['index'].apply(lambda x: columner(x, 0))
data['t2_app'] = data['index'].apply(lambda x: columner(x, 1))
data = data[['t1_app', 't2_app', 0]]
data.columns = ['t1_app', 't2_app', 'NumOverlaps']
data.to_csv('raw_ready.csv')
data = None
data_ = None

# # os.chdir('/mnt/499f5b4c-132a-4775-a414-ce6eba504896/data/clustering/tripeaksRedo/2018-07-12')
data = pd.read_csv('raw_ready.csv', index_col = 0).dropna()[['t1_app', 't2_app', 'NumOverlaps']]
data.columns = ['t1_app', 't2_app', 'NumOverlaps']

data.t1_app = data.t1_app.astype(str)
data.t2_app = data.t2_app.astype(str)

app_popularity = data.groupby('t2_app')['NumOverlaps'].sum()
apps = np.array(app_popularity.sort_values(ascending=False).index)

index_map = dict(np.vstack([apps, np.arange(apps.shape[0])]).T)

count_matrix = ss.coo_matrix((data.NumOverlaps, 
                              (data.t2_app.map(index_map),
                                data.t1_app.map(index_map))),
                              shape=(apps.shape[0], apps.shape[0]),
                              dtype=np.float64)

data = None
conditional_prob_matrix = count_matrix.tocsr()
conditional_prob_matrix = normalize(conditional_prob_matrix, norm='l2', copy=False)

reduced_vectors = TruncatedSVD(n_components=500, random_state=1).fit_transform(conditional_prob_matrix)
reduced_vectors = normalize(reduced_vectors, norm='l2', copy=False)

from sklearn.manifold import TSNE
app_map =TSNE(n_components=2, init = 'pca').fit_transform(reduced_vectors)

app_map_df = pd.DataFrame(app_map, columns=('x', 'y'))

app_map_df['app_bundle_id'] = apps

import hdbscan
clusterer = hdbscan.HDBSCAN(min_samples=40, min_cluster_size=40).fit(app_map)
cluster_ids = clusterer.labels_
app_map_df['cluster'] = cluster_ids
app_map_df.to_csv('map.csv')

app_map_df = pd.read_csv('map.csv')

data = pd.read_parquet("/mnt/499f5b4c-132a-4775-a414-ce6eba504896/Parquet/"+files[1]+".parquet").fillna(0)#[['apphistory', 'impressions', 'installs']]
data = data[data.impressions > 0]
data.columns = ['apphistory', 'impressions', 'installs']
clutsers = app_map_df[['app_bundle_id', 'cluster']].set_index('app_bundle_id').transpose().to_dict()

def clusterizer(x):
    try:
        return clutsers[x]['cluster']
    except:
        return -2

def cleaner(x):
    if ', ' in x:
        _ = x.replace('[', '').replace(']', '').split(', ')
        _ = [clusterizer(y) for y in _]
        return sorted(set(_))
    else:
        _ = x.replace('[', '').replace(']', '')
        _ = [clusterizer(_)]
        return sorted(set(_))

data['clusters'] = data.apphistory.apply(lambda x: cleaner(x))

data['clusters'] = data['clusters'].astype(str)
# data['counter'] = 1
x = data.groupby('clusters').sum().reset_index()
x['cvr'] = x['installs']/x['impressions']
# x['purchases per person'] = x['number of purchases']/x['counter']
# x['dollars per person'] = x['total purchases']/x['counter']

x.to_csv('basicGroupBy.csv')

data.clusters = data.clusters.apply(lambda x: x.replace('[', '').replace(']', '').split(', '))

flat_list = set([item for sublist in data['clusters'] for item in sublist])
ung = []
for q, i in enumerate(flat_list):#output[output.install_y > 1000].sort_values(['relative'], ascending = False).reset_index(drop = True)['clusters']:
    print(q / len(flat_list))
    x = data[data.clusters.apply(lambda x: 1 if i in x else 0) == 1]
    ung.append([i, x['impressions'].sum(), x['installs'].sum(),  len(x),  x['installs'].sum() / len(x)])
ung = pd.DataFrame(ung)
ung.columns = ['cluster', 'impressions', 'installs', 'users', 'installPerUnique']
ung['cvr'] = ung['installs'] /ung['impressions'] 
ung.to_csv('SingleClusterValues.csv')
