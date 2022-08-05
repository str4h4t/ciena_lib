import pandas as pd


def threshold(thres, model, pairs):
    for meth in pairs.keys():
        pairs_found = pd.DataFrame(pairs[meth])
        predicted = []
        for row in pairs_found.iterrows():
            nodes = row[1]['pair'].split(';')
            nodes.sort()
            nodes = ';'.join(nodes)
            corr = row[1]['correlation']
            if nodes not in predicted and corr > thres:
                predicted.append(nodes)
        curr_pairs = pd.DataFrame(predicted)
        curr_pairs = curr_pairs[0].str.split(';', expand=True)
        curr_pairs.to_csv('output/' + model + '_' + meth + '_' + str(thres) + '_out.csv')
    print("done")