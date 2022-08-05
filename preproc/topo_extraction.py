import pandas as pd
import numpy as np
import pickle
import glob
import scipy.stats as st


def convert_ts(df_pmvalues): # convert form of df_pmvalues to calculate correlation value
    list_ts = []
    for i in range(len(df_pmvalues)):
        series = df_pmvalues.iloc[i][5:]
        z_score = st.zscore(np.asarray(series, dtype='float32'))
        list_ts.append(
            {'node': df_pmvalues.node.iloc[i], 'fac': df_pmvalues.fac.iloc[i],
             'pm': df_pmvalues.pm.iloc[i], 'raw': np.asarray(series), 'z_score': z_score})
    df_ts = pd.DataFrame(list_ts)
    return df_ts

def corval_insert(connections, ts): # insert correlation value collumn into true topology 
    topos = connections.copy()
    corr_values_max = []
    for i in list(topos.index):
        node_A = topos.loc[i]['node_1']
        node_B = topos.loc[i]['node_2']
        ts_node_A = ts.loc[ts.node.isin([node_A])]['raw']
        ts_node_B = ts.loc[ts.node.isin([node_B])]['raw']
        _, corr_value_max = correlated_value_kendall(ts_node_A, ts_node_B)  
        corr_values_max.append(corr_value_max)       
    df_corr_values_max = pd.DataFrame(corr_values_max)
    df_corr_values_max.index = topos.index
    topos.insert(2,'corr_max', df_corr_values_max)
    return topos

def correlated_value_kendall(ts_node_A, ts_node_B, opt_max=True):
    corr_values = [st.kendalltau(ts_a, ts_b)[0] for ts_a in ts_node_A for ts_b in ts_node_B]
    if opt_max:
        return corr_values, np.max(corr_values)
    else:
        return corr_values, np.mean(corr_values)
    



class TopoExtraction():
    """ Extracting topo.
    """

    MODEL_FILENAME = "TopoExtraction"
    
    def __init__(self, corr_thres=0.5):
        self.corr_thres = corr_thres # the variance threshold for satisfied time-series
        
    def topo_extract(self, topo_path, df_pmvalues):
        
        ## STEP 1: Extract all possible topologies
        print('STEP 1: Possible topologies')
        
        topo_files = glob.glob(topo_path)
        list_topos = []
        for file in topo_files:
            df_topo_file= pd.read_csv(file, usecols=['tid_scrambled', 'lim', 'neighbor1_final', 'neighbor1_lim', 'crsin', 'crsout'])
            lim_1 = df_topo_file["lim"].str.split('-', expand=True)
            lim_2 = df_topo_file["neighbor1_lim"].str.split('-', expand=True)
            neighbour = df_topo_file.neighbor1_final.str.split('-', expand=True)
            neighbour = neighbour[1] + '-' + neighbour[2] + '-' + neighbour[3]
            crsin = df_topo_file.crsin.str.split('-', expand=True)
            crsout = df_topo_file.crsout.str.split('-', expand=True)
            ## Add port
            col_1 = df_topo_file['tid_scrambled'] + '_' + lim_1[1] + '_' + lim_1[2] + '_' + str(8)
            col_2 = neighbour + '_' + lim_2[1] + '_' + lim_2[2] + '_' + str(6)
            df = pd.DataFrame({'node_1': col_1, 'node_2': col_2})
            df = pd.DataFrame({'node_1':df.min(axis=1), 'node_2':df.max(axis=1)}).drop_duplicates(ignore_index=True)
            list_topos.append(df)
            ## Add port 
            col_1 = df_topo_file['tid_scrambled'] + '_' + lim_1[1] + '_' + lim_1[2] + '_' + str(6)
            col_2 = neighbour + '_' + lim_2[1] + '_' + lim_2[2] + '_' + str(8)
            df = pd.DataFrame({'node_1': col_1, 'node_2': col_2})
            df = pd.DataFrame({'node_1':df.min(axis=1), 'node_2':df.max(axis=1)}).drop_duplicates(ignore_index=True)
            list_topos.append(df)
            ## Add port 
            col_1 = df_topo_file['tid_scrambled'] + '_' + lim_1[1] + '_' + lim_1[2] + '_' + str(6)
            col_2 = neighbour + '_' + lim_2[1] + '_' + lim_2[2] + '_' + str(6)
            df = pd.DataFrame({'node_1': col_1, 'node_2': col_2})
            df = pd.DataFrame({'node_1':df.min(axis=1), 'node_2':df.max(axis=1)}).drop_duplicates(ignore_index=True)
            list_topos.append(df)
            ## Add port 
            col_1 = df_topo_file['tid_scrambled'] + '_' + lim_1[1] + '_' + lim_1[2] + '_' + str(8)
            col_2 = neighbour + '_' + lim_2[1] + '_' + lim_2[2] + '_' + str(8)
            df = pd.DataFrame({'node_1': col_1, 'node_2': col_2})
            df = pd.DataFrame({'node_1':df.min(axis=1), 'node_2':df.max(axis=1)}).drop_duplicates(ignore_index=True)
            list_topos.append(df)

        # Extract all possible topologies
        df_topos_all = (pd.concat(list_topos, ignore_index=True).drop_duplicates(ignore_index=True)
                        .sort_values(by=['node_1', 'node_2'], ignore_index=True))

        node_1 = df_topos_all.node_1.str.split('_', expand=True)
        node_2 = df_topos_all.node_2.str.split('_', expand=True)
        df_node = pd.DataFrame({'node_1': node_1[0], 'node_2': node_2[0]})
        df_topos_all = df_topos_all[df_node.node_1 != df_node.node_2].reset_index(drop=True) # Apply condition TID_1 != TID_2
        df_topos_all = pd.DataFrame({'node_1':df_topos_all.min(axis=1), 'node_2':df_topos_all.max(axis=1)}
                                   ).drop_duplicates(ignore_index=True)
        
        ## STEP 2: Filter topologies with respect to available data set
        print('STEP 2: Filter topologies by available data set')
        
        # number of nodes in pmvalues dataset
        pm_nodes = df_pmvalues.node.drop_duplicates(keep='first').to_list()

        # filter invalid connections
        df_filter_topos = df_topos_all.where(df_topos_all.isin(pm_nodes)).dropna().reset_index(drop=True)
        remaining_nodes = pd.concat([df_filter_topos.node_1, df_filter_topos.node_2], 
                                    axis=0,ignore_index=True).drop_duplicates().to_numpy()
        print("There are only {} remaining connection and {} different nodes from the topo files"
              .format(df_filter_topos.shape[0], remaining_nodes.shape[0]))

        df_pmvalues = df_pmvalues.loc[df_pmvalues.node.isin(remaining_nodes)]
        df_topos_remain = df_filter_topos.sort_values(by='node_1')
        
        ## STEP 3: Filter topologies that satisfy the conditions
        print('STEP 3: Filter topologies by criteria')
        
        df_ts = convert_ts(df_pmvalues) # Convert form of df_pmvalues
        
        # Filter unsatisfied connections (Factor 1)
        # Select connections with higher correlation values

        df_topos_ext = corval_insert(df_topos_remain, df_ts)
        df_1 = df_topos_ext.copy()
        df_2 = pd.DataFrame({'node_1': df_1.node_2, 'node_2': df_1.node_1, 'corr_max': df_1.corr_max})
        df_topos_ext = pd.concat((df_1, df_2), axis=0).reset_index(drop=True)
        
        df_topos_ext = df_topos_ext.groupby('node_1', group_keys=False).apply(lambda x: x.loc[x.corr_max.idxmax()])
        df_topos_ext = df_topos_ext.groupby('node_2', group_keys=False).apply(lambda x: x.loc[x.corr_max.idxmax()])
        df_topos_ext = df_topos_ext.drop(columns='corr_max').reset_index(drop=True)
        df_topos_ext = pd.DataFrame({'node_1': df_topos_ext.min(axis=1), 'node_2': df_topos_ext.max(axis=1)})
        df_topos_ext = df_topos_ext.drop_duplicates(keep='first').sort_values(by='node_1').reset_index(drop=True)
        
        # Filter unsatisfied connections (Factor 2)
        # Select connections with correlation values higher than 0.5
        df_topos_ext = corval_insert(df_topos_ext, df_ts)
        df_topos_ext = df_topos_ext.loc[df_topos_ext.corr_max > 0.5]
        df_topos_ext = df_topos_ext.drop(columns='corr_max')
        df_topos_ext = corval_insert(df_topos_ext, df_ts)
        
        df_topos = df_topos_ext.drop(columns='corr_max')
        
        return df_topos
