import pandas as pd
import configparser
from preproc.data_processing import DataExtraction
from preproc.topo_extraction import TopoExtraction

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('data_config.ini')
    ts_path = config['data'].get('ts_path')
    topo_path = config['data'].get('topo_path')
    topo = bool('True' == config['data'].get('topo'))
    pm_types = ['OPIN-OTS', 'OPOUT-OTS']
    data_extractor = DataExtraction(pm_types=pm_types)
    # data_path = 'data/results_anonymised_Oct_30/df_nhresult*/*.csv'
    # topo_path = 'data/topo_with_amp_ports_anonymised/df_topo_*.csv'
    df_pmvalues, df_riskvalues = data_extractor.ts_extract(ts_path)

    # To find the time_series with respect to PMs

    df_pmvalues = df_pmvalues.loc[df_pmvalues.pm.isin(pm_types)]
    if topo:
        topo_extractor = TopoExtraction()
        df_topos = topo_extractor.topo_extract(topo_path, df_pmvalues)
        a = pd.concat((df_topos.node_1, df_topos.node_2)).unique()

        nodes_topo = pd.concat((df_topos.node_1, df_topos.node_2)).unique()
        df_pmvalues_topo = df_pmvalues.loc[df_pmvalues.node.isin(nodes_topo)]
        df_topos.to_pickle('data_procsd/topo_ground_truth.pkl')

    df_riskvalues.to_pickle('data_procsd/risk_values.pkl')
    df_pmvalues.to_pickle('data_procsd/pm_values.pkl')