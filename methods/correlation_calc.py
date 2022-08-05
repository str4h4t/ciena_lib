import methods.correlation.basic.basic_3 as b3
from methods.correlation.DCCA import dcca_calc

class correlation_calc:
    result = 0
    def __init__(self, data, model_name, param = None):
        self.data = data
        self.model_name = model_name
        self.param = param

    def basic_cor(self):
        result = b3.executor( self.data, self.model_name, self.param)
        return result

    def dcca(self):
        return dcca_calc.executor(self.data, self.param)

    # def dtw(self):
    #     dtw_instance = dtw_calc.dtw_calc(self.data, self.param, self.boost)
    #     result = dtw_instance.executor()
    #     return result

    def execute(self):
        if self.model_name == "pearson" or self.model_name == "spearman" or self.model_name == "kendall" or self.model_name == "all":
            return self.basic_cor()
        if self.model_name == "dcca":
            return self.dcca()

        print("hello")

    

