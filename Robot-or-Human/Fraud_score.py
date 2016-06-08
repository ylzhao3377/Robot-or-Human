import pandas as pd
from collections import defaultdict
def fraud_score(train,outcome0,input):
    allcount = defaultdict(int)
    for device in train[input]:
        allcount[device] += 1
    goodcount = defaultdict(int)
    for device in outcome0[input]:
        goodcount[device] += 1
    device_score = defaultdict(float)
    for key in allcount.keys():
        if key in goodcount.keys():
            device_score[key] = round(float(goodcount[key])/float(allcount[key]),5)
        else:
            device_score[key] = 0
    return(device_score)
