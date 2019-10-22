
# coding: utf-8

"""
Takes a history of app visits by user, and creates community clusters by running a hierarchical clustering model 
on a conditional probability matrix of app connecitons
"""

import pandas as pd
import scipy.sparse as ss
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.base import BaseEstimator
from sklearn.utils import check_array
from os.path import isfile
from itertools import permutations
import subprocess
import json

#the directories of files will obviously need to changed, but the rest should work as is
data = pd.read_csv('/home/jr/Downloads/7aa01a00-b7d0-4cfc-b4c0-10919bc6d83f.csv')
data = data[data._col1.str.contains(',')]
data = data.reset_index(drop = True)

#take out top 20 most popular apps that probably won't be that informative
apps = pd.read_csv('/home/jr/Downloads/16e63bdb-c68a-4c9d-bf84-0c2e5e530d08.csv')
apps = apps['app_bundle_id'][0:20].unique()

#nebulous cleaning
def cleaner(x):
    _ = x.split(', ')
    _ = [x for x in _ if x not in apps]
    return ','.join(_)
data._col1 = data._col1.apply(lambda x: cleaner(x))
data = data[data._col1.str.contains(',')]
data.to_csv('/home/jr/Downloads/cleanedApps.csv')

#counts how many people have visited any permutation of 2 apps
data_ = pd.read_csv('/home/jr/Downloads/cleanedApps.csv', index_col = 0).dropna()
appDict = {}
for i in data_._col1:
    apps = i.replace('[', '').replace(']', '').split(',')
    for z in permutations(apps, 2):
        if z in appDict.keys():
            appDict[z] += 1
        else:
            appDict[z] = 1
data = pd.DataFrame.from_dict(appDict, orient='index').reset_index()

def columner(x, side):
    return x[side]
data['appA'] = data['index'].apply(lambda x: columner(x, 0))
data['appB'] = data['index'].apply(lambda x: columner(x, 1))
data = data[['appA', 'appB', 0]]
data.columns = ['appA', 'appB', 'NumOverlaps']

data.to_csv('/home/jr/Downloads/raw_ready.csv')

data = None
data_ = None

#clean load of the data, makes sure that at least 15 different people have visited and combination of apps
raw_data = pd.read_csv('/home/jr/Downloads/raw_ready.csv', index_col = 0).dropna()
raw_data.columns = ['t1_app', 't2_app', 'NumOverlaps']

raw_data = raw_data[raw_data.NumOverlaps > 15]

app_popularity = raw_data.groupby('t2_app')['NumOverlaps'].sum()
apps = np.array(app_popularity.sort_values(ascending=False).index)

index_map = dict(np.vstack([apps, np.arange(apps.shape[0])]).T)

#coordinate matrix of the probability of visiting any given app given from another particular app
count_matrix = ss.coo_matrix((raw_data.NumOverlaps, 
                              (raw_data.t2_app.map(index_map),
                               raw_data.t1_app.map(index_map))),
                             shape=(apps.shape[0], apps.shape[0]),
                             dtype=np.float64)

conditional_prob_matrix = count_matrix.tocsr()
conditional_prob_matrix = normalize(conditional_prob_matrix, norm='l2', copy=False)

#pulls the singular values out of the above matrix, forces it into a number space of size 'n_components'
reduced_vectors = TruncatedSVD(n_components=200, random_state=1).fit_transform(conditional_prob_matrix)
reduced_vectors = normalize(reduced_vectors, norm='l2', copy=False)

import os, datetime as dt
from sklearn.manifold import TSNE

#Stochastic Neighbor Embedding for visualizing network in two dimensions.  I'm really not sure that this part is necessary for the hdbscan, but ran into memory errors trying to 
#run the hdbscan in the dimensions set at the SVD step.
app_map =TSNE(n_components=2).fit_transform(reduced_vectors)
app_map_df = pd.DataFrame(app_map, columns=('x', 'y'))
app_map_df['app_bundle_id'] = apps
app_map_df.head()

import hdbscan
#the actual clustering 
clusterer = hdbscan.HDBSCAN(min_samples=10, min_cluster_size=40).fit(app_map)
cluster_ids = clusterer.labels_

app_map_df['cluster'] = cluster_ids

#everything bokeh related is just for visualizing.  Kind of cool, but not important 
from bokeh.plotting import figure, show, output_notebook, output_file
from bokeh.models import HoverTool, ColumnDataSource, CustomJS, value
from bokeh.models.mappers import LinearColorMapper
from bokeh.palettes import plasma
from collections import OrderedDict

output_notebook()

palette = ['#777777'] + plasma(cluster_ids.max())
colormap = LinearColorMapper(palette=palette, low=-1, high=cluster_ids.max())
color_dict = {'field': 'cluster', 'transform': colormap}

app_map_df['fill_alpha'] = np.exp((app_map.min() - 
                                         app_map.max()) / 5.0) + 0.05

plot_data = ColumnDataSource(app_map_df)

jscode="""
    var data = source.data;
    var start = cb_obj.start;
    var end = cb_obj.end;
    alpha = data['fill_alpha']
    for (i = 0; i < alpha.length; i++) {
         alpha[i] = Math.exp((start - end) / 5.0) + 0.05;
    }
    source.trigger('change');
"""

bokeh_figure = figure(title='A Map of Apps',
                   plot_width = 700,
                   plot_height = 700,
                   tools= ('pan, wheel_zoom, box_zoom,'
                           'box_select, resize, reset'),
                   active_scroll=u'wheel_zoom')

bokeh_figure.add_tools( HoverTool(tooltips = OrderedDict([('app_bundle_id', '@app_bundle_id'),
                                                       ('cluster', '@cluster')])))

bokeh_figure.circle(u'x', u'y', source=plot_data,
                 fill_color=color_dict, line_color=None, fill_alpha='fill_alpha',
                 size=10, hover_line_color=u'black')

bokeh_figure.x_range.callback = CustomJS(args=dict(source=plot_data), code=jscode)
bokeh_figure.y_range.callback = CustomJS(args=dict(source=plot_data), code=jscode)

bokeh_figure.title.text_font_size = value('18pt')
bokeh_figure.title.align = 'center'
bokeh_figure.xaxis.visible = False
bokeh_figure.yaxis.visible = False
bokeh_figure.grid.grid_line_color = None
bokeh_figure.outline_line_color = '#222222'

show(bokeh_figure);

#reloading data from a campaign and looking at how useful these clusters were in identifying high value users
data = pd.read_csv('/home/jr/Downloads/7aa01a00-b7d0-4cfc-b4c0-10919bc6d83f.csv').dropna()
installs = pd.read_csv('/home/jr/Downloads/e5ea4219-d1de-4d8b-811e-5bd01c763300.csv').user_id.unique()
dat_ = data[data.user_id.isin(installs)]
dat_['install'] = 1.0
_dat = data[~data.user_id.isin(installs)]
_dat['install'] = 0.0
data = pd.concat([_dat, dat_])

clusters = app_map_df[['app_bundle_id', 'cluster']].set_index('app_bundle_id').transpose().to_dict()

def clusterizer(x):
    try:
        return clusters[x]['cluster']
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
#runs through all the apps each user has visited and returns the set of clusters we have seen them in
data['_col2'] = data._col1.apply(lambda x: cleaner(x))

data['_col2'] = data['_col2'].astype(str)

#returns a data frame that lists all observed cluster combinations, how many people were observed in each combination, and what the conversion rate was for that cluster.  There were a handful of clusters
#that outperformed the average conversion rate by quite a bit
output = pd.merge(data.groupby('_col2').mean().reset_index(), data.groupby('_col2').count().reset_index(), on = '_col2')

output[output.user_id > 1000].sort_values('user_id', ascending = False).reset_index(drop = True)