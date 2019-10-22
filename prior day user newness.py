"""
For some parquet of dates and counts, parallelize the process of sifting through to find how new various user were to a given experience
"""

# coding: utf-8

import pandas as pd
file = 'a0cda1f4-9b74-422d-9d5f-7f3a261a5537'
data = pd.read_parquet("/mnt/499f5b4c-132a-4775-a414-ce6eba504896/Parquet/"+file+".parquet")
data.columns = ['user_id', 'dates']

import json 
data['dates'] = data['dates'].apply(lambda x: json.loads(x))

import datetime as dt
data['dates'] = data['dates'].apply(lambda x: [dt.datetime.strptime(str(y), "%Y%m%d").date() for y in x])

data['first'] = data['dates'].apply(lambda x: min(x))
data['last'] = data['dates'].apply(lambda x: max(x))

data = data.sort_values(['first', 'last']).reset_index(drop = True)

base = dt.datetime.today()
numdays = 335
date_list = [(base - dt.timedelta(days=x)).date() for x in range(0, numdays)]
date_list = list(reversed(date_list))

import statistics

def doodlybob(df, li):
    allHolder = []
    recentHolder = []
    for date in li:
        print(date)
        mini = []
        temp = df[df['first'] <= date]
        tempHolder = []
        for z in temp.iterrows():
            views = len([x for x in z[1]['dates'] if x <= date])
            tempHolder.append(views)
        allHolder.append([date, len(temp), sum(tempHolder), statistics.stdev(tempHolder)])
        temp = df[df['last'] >= (date - dt.timedelta(days=30))]
        temp = temp[temp['first'] <= date]
        newHolder = []
        for z in temp.iterrows():
            views = len([x for x in z[1]['dates'] if x <= date])
            newHolder.append(views)
        recentHolder.append([date, len(temp), sum(newHolder), statistics.stdev(newHolder)])
    return allHolder, recentHolder

def chunks(l, n):
    for i in range(0, len(l), n)
       yield l[i:i + n]

holdy = []
for i in chunks(date_list, 43):
    holdy.append(i)

def multi_run_wrapper(args):
    return doodlybob(*args)

from multiprocessing import Pool
    pool = Pool(6)

data = data.sample(100)
results1, results2 = pool.map(multi_run_wrapper,[(data, x) for x in holdy])

results1.to_csv('/mnt/499f5b4c-132a-4775-a414-ce6eba504896/data/outputData/df1.csv')
results2.to_csv('/mnt/499f5b4c-132a-4775-a414-ce6eba504896/data/outputData/df2.csv')
