import pandas as pd
import numpy as np
import multiprocessing as mp
from functools import partial

def crosscorr(datax, datay, meth, lag=0):
    return datax.corr(datay.shift(lag), method = meth)

def pearson(data):
    return data

def spearman(data):
    return data

def kendall(data):
    return data

def executor(data, method, param):

    if method != 'all':
        window_size = param[0]
        correlation_matrix = np.zeros([data.shape[0],data.shape[0]])
        #np.fill_diagonal(correlation_matrix,99)
        pool = mp.Pool(mp.cpu_count()-1)
        for i in range(0, data.shape[0]):
            #print(i)
            if (i != (data.shape[0] - 1)):
                t_1 = pd.Series(data[i])
                for j in range(i + 1, data.shape[0]):
                    t_2 = pd.Series(data[j])
                    func = partial(crosscorr, t_1, t_2, method)
                    result = pool.map(func, [lag for lag in range(-window_size, window_size)])
                    result = list(np.nan_to_num(result))
                    correlation_matrix[i,j] = max(max(result),abs(min(result)))
        correlation_matrix = correlation_matrix.transpose() + correlation_matrix
        pool.terminate()
        return correlation_matrix
    else:
        N = data.shape[0]
        adj_m_pred_pearson = np.zeros([data.shape[0], data.shape[0]])
        adj_m_pred_spearman = np.zeros([data.shape[0], data.shape[0]])
        adj_m_pred_kendall = np.zeros([data.shape[0], data.shape[0]])
        for i in range(0, N - 1):
            for j in range(i + 1, N):
                df = pd.DataFrame([data[i], data[j]]).transpose()
                pearson = df.corr(method='pearson')[0][1]
                spearman = df.corr(method='spearman')[0][1]
                kendall = df.corr(method='kendall')[0][1]
                adj_m_pred_pearson[i][j] = pearson
                adj_m_pred_pearson[j][i] = pearson
                adj_m_pred_spearman[i][j] = spearman
                adj_m_pred_spearman[j][i] = spearman
                adj_m_pred_kendall[i][j] = kendall
                adj_m_pred_kendall[j][i] = kendall
        return adj_m_pred_pearson, adj_m_pred_spearman, adj_m_pred_kendall
