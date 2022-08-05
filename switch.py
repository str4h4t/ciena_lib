import methods.boosted_correlation as bcm
import methods.layered_boolean_correlation as lbc
import methods.validation.validation_bcm_lbc as bcmb
import methods.validation.thresholder as tr
from methods.PTDAlgorithm import PROBTHRESDESCEND
import methods.Gaussian_Processes.GaussianProcesses as gpr
from ast import literal_eval as ev


class switch:
    def __init__(self, ts_data, method, val_flag, norm, threshold, params = None, val_data = None):
        self.ts_data = ts_data
        self.method = method
        self.params = params
        self.val_flag = val_flag
        self.norm = norm
        self.val_data = val_data
        self.threshold = threshold

    def execute_method(self):
        if self.method == "bcm":
            pairs = bcm.calculate_correlation(self.ts_data, self.params, self.norm, self.threshold)
            if self.val_flag:
                bcmb.validate(self.ts_data, self.val_data, pairs, self.threshold, self.method)
            else:
                tr.threshold(self.threshold, pairs)
            print("done")
        if self.method == "lbc":
            pairs = lbc.find_pairs(self.ts_data, self.params, self.norm, self.threshold)
            if self.val_flag:
                bcmb.validate(self.ts_data, self.val_data, pairs, self.threshold, self.method)
            else:
                tr.threshold(self.threshold, self.method, pairs)
        if self.method == "pthredes":
            parameters = ev(self.params)
            if parameters[2][0] == 'None':
                cormax = None
            else:
                cormax = parameters[2][0]
            pthredes = PROBTHRESDESCEND(n_inter= parameters[0][0], corr_algor=parameters[1][0], normalize= self.norm, cormax_name=cormax)
            pairs = pthredes.con_rescon(self.ts_data, self.threshold)
            pairs.to_csv('output/result_ ' + self.method + '_' + str(self.threshold) + '_out.csv')
            if self.val_flag:
                pthredes.validate(self.val_data, pairs)
        if self.method == "gp":
            if not self.val_flag:
                print("This algorithm cannot be executed without a validation set")
            else:
                parameters = ev(self.params)
                pairs = gpr.gp_exe(self.ts_data, self.val_data, parameters[0][0])
                pairs.to_csv('output/gaussian_process_out.csv')
                print("done")
