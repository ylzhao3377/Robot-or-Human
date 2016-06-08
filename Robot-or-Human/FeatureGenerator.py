import pandas as pd
import numpy as np
from collections import defaultdict
from fraud_score import fraud_score
from times_four_sec import times_four_sec


bidder = pd.read_csv('train.csv', delimiter=',')
bid = pd.read_csv('bids.csv',delimiter=',')
feature = pd.read_csv('features.csv',delimiter=',')
train = pd.merge(bidder,bid,how='outer',on='bidder_id')
'''
outcome1 = train[train.outcome==1]
outcome0 = train[train.outcome==0]

device_score = fraud_score(train,outcome0,'device')
#ip_score = fraud_score(train,outcome0,'ip')
auction_score = fraud_score(train,outcome0,'auction')
country_score = fraud_score(train,outcome0,'country')
merchandise_score = fraud_score(train,outcome0,'merchandise')
#times_four_sec = times_four_sec(train,'time')
#url_score = fraud_score(train,outcome0,'url')


#print(train.head(5))
train['device_score'] = train['device'].map(device_score)
train['country_score'] = train['country'].map(country_score)
train['auction_score'] = train['auction'].map(auction_score)
train['merchandise_score'] = train['merchandise'].map(merchandise_score)
header_train = train.columns.values.tolist()
print(header_train)



my_bidder_id = np.unique(train['bidder_id'])
mytrain = pd.DataFrame(index = my_bidder_id, columns=['device_score','auction_score','merchandise_score','country_score'])
mytrain= mytrain.fillna(0)
counter = 0
for bidder_id in my_bidder_id:
    x = np.mean(train.loc[train['bidder_id'] == bidder_id,'device_score'].values[0])
    y = np.mean(train.loc[train['bidder_id'] == bidder_id,'auction_score'].values[0])
    z = np.mean(train.loc[train['bidder_id'] == bidder_id,'merchandise_score'].values[0])
    w = np.mean(train.loc[train['bidder_id'] == bidder_id,'country_score'].values[0])
    mytrain.loc[bidder_id] = pd.Series({'device_score':x, 'auction_score':y, 'merchandise_score':z,'country_score':w})
    counter += 1
    if counter%100==0:
        print(counter/float(len(my_bidder_id)), 'processed')
print(mytrain.head(5))
mytrain.to_csv('mytrain.csv', sep='\t', encoding='utf-8')

'''
mytrain = pd.read_csv('mytrain.csv',delimiter='\t')
myfeature = pd.merge(mytrain,feature,how='outer',on='bidder_id')
myfeature.to_csv('myfeature.csv', sep=',', encoding='utf-8')
#print(times_four_sec)
