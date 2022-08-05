# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 14:23:31 2022

@author: Ornela Bregu
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.gaussian_process import GaussianProcessClassifier
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar import vecm
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.tsatools import lagmat, add_trend
from statsmodels.tsa.adfvalues import mackinnonp
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix


        
def adf(ts):
   """
   Augmented Dickey-Fuller unit root test
   """
   # make sure we are working with an array, convert if necessary
   ts = np.asarray(ts)

   # Get the dimension of the array
   nobs = ts.shape[0]

   # We use 1 as maximum lag in our calculations
   maxlag = 1

   # Calculate the discrete difference
   tsdiff = np.diff(ts)

   # Create a 2d array of lags, trim invalid observations on both sides
   tsdall = lagmat(tsdiff[:, None], maxlag, trim='both', original='in')
   # Get dimension of the array
   nobs = tsdall.shape[0]

   # replace 0 xdiff with level of x
   tsdall[:, 0] = ts[-nobs - 1:-1]
   tsdshort = tsdiff[-nobs:]

   # Calculate the linear regression using an ordinary least squares model
   results = OLS(tsdshort, add_trend(tsdall[:, :maxlag + 1], 'c')).fit()
   adfstat = results.tvalues[0]

   # Get approx p-value from a precomputed table (from stattools)
   pvalue = mackinnonp(adfstat, 'c', N=1)
   return pvalue


def adfuller_test(series, signif=0.05, name='', verbose=False):
    """Perform ADFuller to test for Stationarity of given series and print report"""
    r = adfuller(series, autolag='AIC')
    output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}
    p_value = output['pvalue']
    def adjust(val, length= 6): return str(val).ljust(length)

    # Print Summary
    print(f'    Augmented Dickey-Fuller Test on "{name}"', "\n   ", '-'*47)
    print(f' Null Hypothesis: Data has unit root. Non-Stationary.')
    print(f' Significance Level    = {signif}')
    print(f' Test Statistic        = {output["test_statistic"]}')
    print(f' No. Lags Chosen       = {output["n_lags"]}')

    for key,val in r[4].items():
        print(f' Critical value {adjust(key)} = {round(val, 3)}')

    if p_value <= signif:
        print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
        print(f" => Series is Stationary.")
    else:
        print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")
        print(f" => Series is Non-Stationary.")
    return p_value

def stationarity(rd):
    count_OPOUT =0
    count_OPIN=0
    count_OPOUT_stat =0
    count_OPOUT_nnstat=0
    count_OPIN_stat=0
    count_OPIN_nnstat=0
    p_values=np.zeros((len(rd)))
    OUT_nn_ts = []
    IN_nn_ts = []
    OUT_st_ts = []
    IN_st_ts = []

    for i in range(0,len(rd)):
    #     x = df[df.index== (topo_nodes.node.values[i]+"_AMP_OPOUT-OTS")]
        x = rd[rd.index== rd.index[i]]
        if ("AMP_OPOUT-OTS") in rd.index[i]:
            if (len(x)==0):
                i=i+1
            else:
                count_OPOUT = count_OPOUT+1
                x=x.values.reshape(-1)
                p_values[i] = adf(x)
                if adf(x)<0.05:
                    count_OPOUT_stat = count_OPOUT_stat+1
                    OUT_st_ts.append(rd.index[i])
                else:
                    count_OPOUT_nnstat= count_OPOUT_nnstat+1
                    OUT_nn_ts.append(rd.index[i])
        else:
            x = rd[rd.index== rd.index[i]]
            if (len(x)==0):
                i=1+1
            else:
                count_OPIN = count_OPIN+1
                x=x.values.reshape(-1)
                p_values[i] = adf(x)
                if adf(x)<0.05:
                    count_OPIN_stat = count_OPIN_stat+1
                    IN_st_ts.append(rd.index[i])
                else:
                    count_OPIN_nnstat= count_OPIN_nnstat+1
                    IN_nn_ts.append(rd.index[i])
    return IN_nn_ts,OUT_nn_ts, OUT_st_ts,IN_st_ts

# ADF Test on each column
def stationary_predicted(nnst_ts):
    stationality_pred_ts = pd.DataFrame([])
    for i in range(0,nnst_ts.shape[0],5):
        train = nnst_ts.iloc[i:i+5].T
        test = nnst_ts.iloc[i:i+5].T
        # VECM
        if train.values.shape[1] < 2:
            continue
        model = vecm.VECM(train.values, k_ar_diff =2, coint_rank = 5, deterministic='co')

        """
        k_ar_diff = lag length
        set the deterministic argument to "co", meaning that there
        is a constant inside the cointegrating relationship
        """
        res = model.fit()
        out = durbin_watson(res.resid)
        for col, val in zip(nnst_ts.columns, out):
            print((col), ':', round(val, 2))
        residuals = res.resid
        residuals = pd.DataFrame(residuals, columns = train.columns)

        model_fit = res

        # make predictions
        predictions = model_fit.predict(len(test))
        predictions = pd.DataFrame(predictions, columns = test.columns)


        for name, column in predictions.iteritems():
            p_value = adfuller_test(column, name=column.name)
            if p_value <= 0.05:
                stationality_pred_ts = stationality_pred_ts.append(column)
    return stationality_pred_ts

def GP_prep_dataset(topo,topo_nodes,time_series_data):
    target_values = {}
    data_ts = {}
    for curr_node in range(len(topo_nodes)):
        for next_node in range(curr_node +1, len(topo_nodes)):
            tid_curr = topo_nodes.values[curr_node][0].split('_')[0]
            tid_next = topo_nodes.values[next_node][0].split('_')[0]
            name = topo_nodes.values[curr_node][0]+";"+topo_nodes.values[next_node][0]
            df1 = time_series_data[time_series_data.index == topo_nodes.values[curr_node][0]+"_AMP_OPOUT-OTS"]
            df2 = time_series_data[time_series_data.index == topo_nodes.values[next_node][0]+"_AMP_OPOUT-OTS"]
    #         df1 = time_series_data[time_series_data.index == topo_nodes.values[curr_node][0]]
    #         df2 = time_series_data[time_series_data.index == topo_nodes.values[next_node][0]]
            data_ts[name] = np.concatenate((df1.values,df2.values)).flatten()
            if tid_curr==tid_next:
                target_values[name] = 1
            elif topo[topo.node_1 == curr_node].empty & topo[topo.node_2 == curr_node].empty:
                target_values[name] = 0
            elif topo[topo.node_1 == curr_node].empty & (topo[topo.node_2 == curr_node].node_1 == next_node).bool():
                target_values[name] = 1
            elif topo[topo.node_2 == curr_node].empty & (topo[topo.node_1 == curr_node].node_2 == next_node).bool():
                target_values[name] = 1
            else:
                target_values[name] = 0
    return target_values, data_ts

def calculate_metrics(target_values,data_ts, sample_size):
    y = pd.DataFrame.from_dict(target_values,orient='index', columns =['target'])

    X = pd.DataFrame.from_dict(data_ts,orient = 'index')
    X = X.iloc[:,:148].dropna()

    dataset = pd.merge(X, y, left_index=True, right_index=True)

    X_train, X_test, y_train, y_test = train_test_split(dataset[dataset['target']==1].iloc[:,:148], dataset[dataset['target']==1].iloc[:,148], test_size=0.3, random_state=0)
    sampled_dataset = dataset[dataset['target']==0].sample(n=sample_size)

    X_trn, X_tst, y_trn, y_tst = train_test_split(sampled_dataset[sampled_dataset['target']==0].iloc[:,:148], sampled_dataset[sampled_dataset['target']==0].iloc[:,148], test_size=0.3, random_state=0)

    training_X = pd.concat([X_train, X_trn])
    testing_X = pd.concat([X_test, X_tst])
    training_y = pd.concat([y_train, y_trn])
    testing_y = pd.concat([y_test, y_tst])
    model = GaussianProcessClassifier().fit(training_X, training_y)
    training_scores = model.score(training_X, training_y)
    y_pred_training = model.predict(training_X)
    training_scores = precision_recall_fscore_support(training_y, y_pred_training, average='macro')
    y_pred= model.predict(testing_X)
    testing_scores = precision_recall_fscore_support(testing_y, y_pred, average='macro')
    pairs =  testing_y.iloc[np.where(y_pred==1)]
    precision = testing_scores[0]
    recall = testing_scores[1]
    fscore = 2 * (precision * recall) / (precision + recall)
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))
    print("F1-score: " + str(fscore))
    return pairs.index
        
    
  
def gp_exe(time_series_data, topo, sample_size):
    time_series_data = time_series_data.iloc[:,3:].squeeze()
    IN_nn_ts,OUT_nn_ts, OUT_st_ts,IN_st_ts = stationarity(time_series_data)
    OUT_nnst_ts = pd.DataFrame(OUT_nn_ts,columns=["First"])
    OUT_nnst_ts.index = OUT_nnst_ts.First
    OUT_nnst_ts = pd.merge(time_series_data, OUT_nnst_ts, left_index=True, right_index=True).iloc[:,:74].squeeze()
    OUT_nnst_ts = OUT_nnst_ts.sort_index(ascending=True)
    OUT_stationaly_pred_ts = stationary_predicted(OUT_nnst_ts)
    topo_nodes = pd.concat([topo.node_1, topo.node_2],ignore_index=True).drop_duplicates().to_frame(name="node")
    topo_nodes = topo_nodes.reset_index(drop=True)
    OPOUT_data = pd.merge(time_series_data, OUT_nnst_ts, left_index=True, right_index=True).iloc[:,:74].squeeze()
    OUT_stationaly_pred_ts = pd.DataFrame(OUT_stationaly_pred_ts.values, columns=OPOUT_data.columns, index = OUT_stationaly_pred_ts.index )
    OPOUT_stationary_ts = pd.concat([OUT_stationaly_pred_ts,OPOUT_data])
    target_values, data_ts = GP_prep_dataset(topo,topo_nodes,OPOUT_stationary_ts)
    
    pairs = calculate_metrics(target_values,data_ts, sample_size)
    pairs = pd.DataFrame(pairs)
    pairs = pairs[0].str.split(';', expand=True)
    return pairs