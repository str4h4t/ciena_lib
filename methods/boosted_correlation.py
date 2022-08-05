import methods.correlation_calc as cc
import methods.clustering.kshape_cluster as ks
import numpy as np
import scipy.stats as st
from ast import literal_eval as ev

def z_norm(data):
    z_data = []
    for d in range(0,data.__len__()):
        z_data.append(st.zscore(data[d]))
    return np.asarray(z_data)

def calculate_correlation(time_series, params, norm, threshold):
    print("Starting Procedure for Boosted Correlation Method...")
    params = ev(params)
    methods = params[0]
    if params.__len__() == 3:
        window_size = params[1]
        dcca_k = params[2]
    elif params.__len__() <= 2:
        if "dcca" in methods:
            dcca_k = params[1]
        else:
            window_size = params[1]
    time_series = time_series.reset_index()

    data = np.asarray(time_series.iloc[:, 4:])
    if norm:
        data = z_norm(data)
    print("Running K-Means...")
    k = int(data.__len__() / 20)
    assignments = ks.clusterer(data, k)
    config = []
    pairs = {}
    for method in methods:
        pairs[method] = []
        if method == "dcca":
            # noinspection PyUnboundLocalVariable
            config.append(["dcca", dcca_k[0]])
        else:
            # noinspection PyUnboundLocalVariable
            config.append([method, window_size])

    for p in config:
        method = p[0]
        print("Mehthod : " + method)
        for index, row in assignments.iterrows():
            print("Analyzing cluster : " + str(index))
            members = row[1]
            member_data = data[members]
            member_details = time_series.iloc[members]
            corr = cc.correlation_calc(member_data, p[0], p[1])
            curr_result = corr.execute()
            np.fill_diagonal(curr_result, 0)
            indices = member_details.index
            nodes = member_details['node']
            for ctr in range(0,indices.__len__()):
                max_cor = max(curr_result[ctr])
                if max_cor > threshold:
                    n1 = nodes[indices[ctr]]
                    n2 = nodes[indices[np.argmax(curr_result[ctr])]]
                    n = [n1,n2]
                    n.sort()
                    pair = ';'.join(n)
                    if pair not in pairs[method] and n1 != n2:
                        pairs[method].append({'pair': pair, 'correlation': max_cor})
                ctr += 1
    return pairs