import configparser

import pandas as pd

import switch as sw
import pickle as pkl
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('model_config.ini')
    method = config['main'].get('model_name')
    validate = bool('True' == config['main'].get('validation'))
    params = config['main'].get('parameters')
    norm = bool(config['main'].get('normalization'))
    ts_loc = config['main'].get('ts_file_loc')
    threshold = float(config['main'].get('threshold'))
    ts_data = pd.read_pickle(ts_loc)
    if validate:
        val_loc = config['main'].get('val_file_loc')
        val_data = pd.read_pickle(val_loc)
        meth = sw.switch(ts_data, method, validate, norm, threshold, params, val_data)
    else:
        meth = sw.switch(ts_data, method, validate, norm, threshold, params)
    meth.execute_method()
    print("Done")