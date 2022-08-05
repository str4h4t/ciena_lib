import pandas as pd
import numpy as np
import pickle
import glob
import scipy.stats as st
from scipy.special import softmax




def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name, 'rb') as f:
        return pickle.load(f)

class PROBTHRESDESCEND:
    """ Threshold descending correlation algorithm.
    """

    MODEL_FILENAME = "PROBABILISTIC THRESDESCEND_model"
    

    def __init__(self, n_inter=20, corr_algor='Spearman', normalize=True, cormax_name=None):
        self.n_inter = n_inter
        self.corr = corr_algor
        print(self.corr)
        self.normalize = normalize # whether z-score normalizationis applied
        self.cormax_name = cormax_name # name of the correlation matrix saved in advanced
        
    def correlated_value(self, ts_node_A, ts_node_B):
        corr_values = []
        if self.corr == 'spearman':
            corr_values = [st.spearmanr(ts_a, ts_b)[0] for ts_a in ts_node_A for ts_b in ts_node_B]
        elif self.corr == 'kendall':
            corr_values = [st.kendalltau(ts_a, ts_b)[0] for ts_a in ts_node_A for ts_b in ts_node_B]
        elif self.corr == 'pearson':
            corr_values = [st.pearsonr(ts_a, ts_b)[0] for ts_a in ts_node_A for ts_b in ts_node_B]
            
        return corr_values, np.max(corr_values)
    
    def convert_ts(self, df_pmvalues): # convert form of df_pmvalues to calculate correlation value
        list_ts = []
        for i in range(len(df_pmvalues)):
            series = df_pmvalues.iloc[i][5:]
            z_score = st.zscore(np.asarray(series, dtype='float32'))
            list_ts.append(
                {'node': df_pmvalues.node.iloc[i], 'fac': df_pmvalues.fac.iloc[i],
                 'pm': df_pmvalues.pm.iloc[i], 'raw': np.asarray(series), 'z_score': z_score})
        df_ts = pd.DataFrame(list_ts)
        return df_ts
            
    
    def con_rescon(self, df_pmvalues, threshold): # Connections Reconstruction
        df_ts = self.convert_ts(df_pmvalues)
        self.nodes = nodes = df_ts.node.drop_duplicates().to_list()
        
        # Calculate the score matrix of nodes
        if self.cormax_name is None:
            print('calculating score matrix...')
            n = nodes.__len__()
            score_matrix = np.zeros(shape=(n, n))

            for i in range(n):
                for j in range(i, n):
                    node_A = nodes[i]
                    node_B = nodes[j]
                    if self.normalize:
                        ts_node_A = df_ts.loc[df_ts.node.isin([node_A])]['z_score']
                        ts_node_B = df_ts.loc[df_ts.node.isin([node_B])]['z_score']
                    else:
                        ts_node_A = df_ts.loc[df_ts.node.isin([node_A])]['raw']
                        ts_node_B = df_ts.loc[df_ts.node.isin([node_B])]['raw']

                    _, corr_value_max = self.correlated_value(ts_node_A, ts_node_B)
                    score_matrix[i, j] = corr_value_max

            for i in range(1, n):
                for j in range(i):
                    score_matrix[i,j] = score_matrix[j, i]
                    
            print('saving score matrix...')
            save_obj( score_matrix, 'score_matrix')
            
        else:
            print('score maxtrix is calculated in advanced')
            score_matrix = load_obj(self.cormax_name)
            # print(score_matrix)
        
        # From the score matrix, reconstruct connections
            
        connections_list = []
        for n in range(self.n_inter):
            print("Iteration: ", n)
            Set_Ori = nodes.copy()
            Set_A = nodes.copy()
            np.random.shuffle(Set_A)
            Set_B = []
            thres = 1
            res_connections = []
            i = 0
            while (len(Set_A) > 1 and thres >= threshold):
                thres = thres - 0.05
                Set_C = []
                while Set_A:
                    i = i+1
                    node_1 = Set_A[0]
                    info1 = node_1.split('_')
                    TID1 = info1[0]
                    ind = Set_Ori.index(node_1)
                    row = score_matrix[ind,:].copy()
                    inds_B = [Set_Ori.index(node) for node in Set_B]
                    inds_C = [Set_Ori.index(node) for node in Set_C]
                    row[inds_B] = float('-inf') 
                    row[inds_C] = float('-inf') 
                    row[ind] = float('-inf')
                    ind_sorted = np.argsort(row)[::-1]
                    row_sorted = np.sort(row)[::-1]
                    node_sorted = [nodes[j] for j in ind_sorted]
                    node_2 = node_sorted[0]
                    info2 = node_2.split('_')
                    TID2 = info2[0]
                    if row[ind_sorted[0]] > thres and TID1!=TID2:
                        res_connections.append({"node_1": node_1, "node_2": node_2})
                        Set_A.remove(node_1)
                        Set_A.remove(node_2)
                        Set_B.append(node_1)
                        Set_B.append(node_2)
                    else:
                        Set_A.remove(node_1)
                        Set_C.append(node_1)
                Set_A = Set_C.copy()
            connections_list.append(pd.DataFrame(res_connections))

        df_connections_all = pd.concat(connections_list, ignore_index=True)
        n = np.round(df_connections_all.__len__()/self.n_inter).astype('int32')
        # print(n)
        df_connections_sorted = pd.DataFrame({'node_1': df_connections_all.min(axis=1), 'node_2': df_connections_all.max(axis=1)}).sort_values(by='node_1')
        df_connections_counted = df_connections_sorted.value_counts().reset_index()
        df_connections_counted.rename({0: 'counts'}, axis=1, inplace=True)
        
        prob_matrix = score_matrix.copy()
        np.fill_diagonal(prob_matrix, -np.inf)
        prob_matrix = softmax(prob_matrix, axis=0)
        
        prob = []
        for i in df_connections_counted.index:
            a = df_connections_counted.node_1[i]
            idx_a = Set_Ori.index(a)
            b = df_connections_counted.node_2[i]
            idx_b = Set_Ori.index(b)
            prob.append(prob_matrix[idx_a, idx_b])
            
        df_connections_counted['prob'] = np.array(prob)
        total_prob = df_connections_counted.counts*df_connections_counted.prob
        df_connections_counted['total_prob'] = total_prob
        df_connections_counted.sort_values(by='total_prob', ascending=False, inplace=True)
        # print('Reconstructed Connections: \n',  df_connections_counted)
        df_connections = df_connections_counted.drop(['counts', 'prob', 'total_prob'], axis=1)[:n]
        self.df_connections = df_connections.sort_values(by='node_1').reset_index(drop=True)

        print("Reconstructed Topology: \n", self.df_connections)
        
        return self.df_connections
    
    def validate(self, df_topos, df_connections):
        n = df_topos.__len__()
        # df_connections = self.df_connections

        df_TP = pd.merge(df_topos, df_connections, on=['node_1', 'node_2'])
        TP = df_TP.shape[0]
        FP = df_connections.shape[0] - TP
        FN = df_topos.shape[0] - TP
        Precision = TP/(TP+FP)
        Recall = TP/(TP+FN)
        F1_score = 2*Precision*Recall/(Precision+Recall)
        print("Performance metric:\n{:20}{}\n{:20}{}\n{:20}{}\n{:20}{}\n{:20}{}\n{:20}{}"
              .format('TruePositive', TP, 'FalsePositive', FP, "FalseNegative", FN,
                      'Precision', Precision, 'Recall', Recall, "F1_score", F1_score))
        # print("Reconstructed Topology: \n", self.df_connections)