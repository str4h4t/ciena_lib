import pandas as pd
import glob
from tqdm import tqdm



class DataExtraction():
    """ Extracting data.
    """

    MODEL_FILENAME = "DataExtraction"
    
    def __init__(self, pm_types, mis_limit=10, var_thres=0.05):
        self.pm_types = pm_types
        self.mis_limit = mis_limit # the maximum missing points permited in a time-series
        self.var_thres = var_thres # the variance threshold for satisfied time-series

        
    def ts_extract(self, data_path):
        
        ## STEP 1: Load data
        print('STEP 1: Data loading')
        nh_files = glob.glob(data_path)
        
        # Load data files
        list_nh_files = []
        for file in tqdm(nh_files):
            df = pd.read_csv(file)
            list_nh_files.append(df)
        df_pm = pd.concat(list_nh_files, ignore_index=True)
        
        ## STEP 2: Filter by PM types
        print('STEP 2: Filter pm values by pm types')

        df_pm = df_pm.loc[df_pm.pm.isin(self.pm_types)].reset_index()
        
        ## STEP 3: Group pm values by ts-id
        
        print('STEP 3: Group pm values into time-series')
        df_pm['fac'] = df_pm.port_key_anonymised.str.split('::', expand=True)[1].str.split('-', expand=True)[0]
        df_pm['pm'] = df_pm['pm'].str.replace('_','-')
        df_pm['ts_id'] = (df_pm['mename_anonymised'] + '_' + df_pm['shelf'].astype(str) + 
                                               '_'+df_pm['slot'].astype(str) + '_' + df_pm['port'].astype(str) + 
                                               '_'+df_pm['fac']+'_'+df_pm['pm'])
        # time-series construction
        df_reformated = (df_pm[['ts_id', 'pmtime', 'pmvalue', 'riskvalue']].
                         groupby(['ts_id']).agg(lambda x: list(x)))
        
        ## STEP 4: Expanding data into pm time-series and risk time-series
        
        print('STEP 4: Pm time-series and risk time-series extraction')
        list_pmvalues = []
        list_riskvalues = []

        for ts in tqdm(df_reformated.iterrows(), total=df_reformated.shape[0]):
            list_pmvalues.append(pd.DataFrame({'time':pd.to_datetime(ts[1].pmtime, unit='s').normalize(), 
                                               ts[0]:ts[1].pmvalue}).groupby('time').mean())
            list_riskvalues.append(pd.DataFrame({'time':pd.to_datetime(ts[1].pmtime, unit='s').normalize(), 
                                                 ts[0]:ts[1].riskvalue}).groupby('time').mean())
            
        ## STEP 5: Apply criteria to get satisfied time-series
        
        print('STEP 5: Filtering time-series')
 
        pmvalues = pd.concat(list_pmvalues, axis=1)
        riskvalues = pd.concat(list_riskvalues, axis=1)

        # Remove trivial outlier points
        riskvalues = riskvalues.where(pmvalues > -100)
        pmvalues = pmvalues.where(pmvalues > -100)
        
        # Apply missing values limit
        riskvalues = riskvalues.loc[:, (pmvalues.isna().sum(axis=0) < self.mis_limit)]
        pmvalues = pmvalues.loc[:, (pmvalues.isna().sum(axis=0) < self.mis_limit)]
        
        # Apply variance threshold 
        riskvalues = riskvalues.loc[:, (pmvalues.std(0) > self.var_thres)]
        pmvalues = pmvalues.loc[:, (pmvalues.std(0) > self.var_thres)]
        
        ## STEP 6: Interpolation
        
        print('STEP 6: Times-series interpolation')
        
        # Select the time period for time-series
        pmvalues_reindex = pmvalues.reindex(pd.date_range("2020-01-17", "2020-03-30"))
        riskvalues_reindex = riskvalues.reindex(pd.date_range("2020-01-17", "2020-03-30"))

        pmvalues_reindex.interpolate(inplace=True)
        riskvalues_reindex.interpolate(inplace=True)


        riskvalues_reindex = riskvalues_reindex.loc[:, (pmvalues_reindex.isna().sum() == 0) & 
                                                    (pmvalues_reindex.std(0) > self.var_thres)]
        pmvalues_reindex = pmvalues_reindex.loc[:, (pmvalues_reindex.isna().sum() == 0) & 
                                                (pmvalues_reindex.std(0) > self.var_thres)]

        pmvalues_reindex = pmvalues_reindex.transpose()
        riskvalues_reindex = riskvalues_reindex.transpose()
        
        ## STEP 7: Add necessary information
        
        print('STEP 7: Add important information into time-series')
        
        info = pmvalues_reindex.index.to_series().str.split('_', expand=True)
        df_pmvalues = pd.concat([pd.DataFrame({'node': info[0]+'_'+info[1]+'_'+info[2]+'_'+info[3],
                                            'fac': info[4], 'pm': info[5]}), pmvalues_reindex], axis=1)
        
        df_riskvalues = pd.concat([pd.DataFrame({'node': info[0]+'_'+info[1]+'_'+info[2]+'_'+info[3], 
                                            'fac': info[4], 'pm': info[5]}), riskvalues_reindex], axis=1)
        
        return df_pmvalues, df_riskvalues
