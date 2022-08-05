import numpy as np
import pandas as pd
import methods.correlation.DCCA.dcca_calc as dc
import scipy.stats as st
from ast import literal_eval as ev

def z_norm(data):
    z_data = []
    for d in range(0,data.__len__()):
        z_data.append(st.zscore(data[d]))
    return np.asarray(z_data)

def find_pairs(time_series, params, norm, threshold):
    data = np.asarray(time_series.iloc[:, 3:])
    params = ev(params)
    methods = params[0]
    if norm:
        data = z_norm(data)
    N = data.shape[0]
    D = data.shape[1]
    mean_array= data.mean(axis=1)
    mean_array = np.repeat(mean_array.reshape([N,1]),D,axis=1)
    macro_data = np.where(data > mean_array, True, False)
    macro_data = np.asarray(macro_data)
    t_minus_1 = np.delete(np.concatenate((np.zeros((N,1)), data), axis=1),-1,1)
    micro_data = np.where(data > t_minus_1, True, False)
    micro_data[:,0] = np.ones([N])
    L = data.shape[1]
    pairs = {}
    for method in methods:
        pairs[method] = []
    ctr_mi = 0
    ctr_ma = 0
    for i in range(0,N-1):
        print(i)
        for j in range(i+1, N):
            if time_series.iloc[i]['node'] == time_series.iloc[j]['node']:
                continue
            macro_corr = 1 - (macro_data[i] ^ macro_data[j]).sum()/L
            micro_corr = 1 - (micro_data[i] ^ micro_data[j]).sum() / L
            pair = time_series.iloc[i]['node'] + ';' + time_series.iloc[j]['node']
            if macro_corr > threshold:
                ctr_ma += 1
                if micro_corr > threshold:
                    ctr_mi += 1
                    df = pd.DataFrame([data[i],data[j]]).transpose()
                    if "pearson" in methods:
                        pearson = df.corr(method = 'pearson')[0][1]
                        if pearson > threshold:
                            pairs['pearson'].append({'pair': pair, 'correlation': pearson})
                    if "spearman" in methods:
                        spearman = df.corr(method = 'spearman')[0][1]
                        if spearman > threshold:
                            pairs['spearman'].append({'pair': pair, 'correlation': spearman})
                    if "kendall" in methods:
                        kendall = df.corr(method = 'kendall')[0][1]
                        if kendall > threshold:
                            pairs['kendall'].append({'pair': pair, 'correlation': kendall})
                    if "dcca" in methods:
                        dcca = dc.executor(np.asarray(df).transpose(),6)[0][1]
                        if dcca > threshold:
                            pairs['dcca'].append({'pair': pair, 'correlation': dcca})
    return pairs