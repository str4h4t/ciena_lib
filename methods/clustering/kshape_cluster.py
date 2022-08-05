from methods.clustering.kshapemaster.kshape.core import kshape, zscore
import pandas as pd

def clusterer(data, k):
    clusters = kshape(zscore(data, axis=1), k)
    assignments = pd.DataFrame(clusters)
    return assignments