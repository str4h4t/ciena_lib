import numpy as np
import pandas as pd

def validate(data_set, df, pairs, threshold, model):
    total_positives = 0
    nh_nodes = data_set['node'].unique()
    df_combined = []
    for row in df.iterrows():
        # if row[1]['node1'].split('_')[0] == row[1]['node2'].split('_')[0]:
            df_combined.append('_'.join([row[1]['node_1'],row[1]['node_2']]))
            if (row[1]['node_1'] in nh_nodes) and (row[1]['node_2'] in nh_nodes):
                total_positives += 1
    print(total_positives)
    for meth in pairs.keys():
        results =[]
        for thres in np.arange(threshold, 1, 0.01):
            pairs_found = pd.DataFrame(pairs[meth])
            TP = 0
            predicted = []
            for row in pairs_found.iterrows():
                nodes = row[1]['pair'].split(';')
                nodes.sort()
                nodes = '_'.join(nodes)
                corr = row[1]['correlation']
                if nodes not in predicted and corr > thres:
                    predicted.append(nodes)
            for p in predicted:
                if p in df_combined:
                    TP += 1
            FP = predicted.__len__() - TP
            FN = total_positives - TP
            if TP + FN == 0 or TP + FP == 0 or TP == 0:
                Precision = 1
                Recall = 1
                fscore = 2 * (Precision * Recall) / (Precision + Recall)
                results.append(
                    {'threshold': thres, 'tp': TP, 'fp': FP, 'fn': FN, 'precision': Precision, 'recall': Recall,
                     'f1_score': fscore})
                continue
            Recall = TP/(TP+FN)
            Precision = TP/(TP+FP)
            fscore = 2 * (Precision * Recall) / (Precision + Recall)
            results.append({'threshold': thres, 'tp': TP, 'fp': FP, 'fn': FN, 'precision': Precision, 'recall': Recall,
                            'f1_score': fscore})
        results = pd.DataFrame(results)
        optimal_threshold = results.loc[results['f1_score'].idxmax()]['threshold']
        pairs_found = pd.DataFrame(pairs[meth])
        predicted = []
        for row in pairs_found.iterrows():
            nodes = row[1]['pair']
            corr = row[1]['correlation']
            if nodes not in predicted and corr > optimal_threshold:
                predicted.append(nodes)
        results.to_csv('output/result_' + meth + '_' + model + '_complete.csv')
        best_pairs = pd.DataFrame(predicted)
        best_pairs = best_pairs[0].str.split(';', expand=True)
        best_pairs.to_csv('output/result_' + model + '_' + meth + '_' + str(thres) + '_best_out.csv')