import os
import matplotlib
import matplotlib.cm as cm
import numpy as np
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd


def extract_dims(embed_data):
    x, y = [], []
    for example in embed_data:
        x.append(float(example[0]))
        y.append(float(example[1]))
    return x, y


if __name__ == '__main__':
    in_fp = 'data/prototype_contexts_w_embeddings_reduced.pk'
    print('Loading data')
    with open(in_fp, 'rb') as fd:
        data = pickle.load(fd)

    acronyms_fp = '../context_extraction/data/merged_prototype_expansions_w_counts.csv'
    acronyms_df = pd.read_csv(acronyms_fp)
    sfs = acronyms_df['sf'].unique()

    lf_to_sf_map = {}
    for row_idx, row in acronyms_df.iterrows():
        row = row.to_dict()
        lf_to_sf_map[row['lf']] = row['sf']

    forms = data['form']
    embeddings = data['embeddings']

    sf_chart_data = {}

    for sf in sfs:
        sf_chart_data[sf] = {
            'sf_embeddings': [],
            'lf_embeddings': defaultdict(list),
        }

    print('Finished loading data.  Organizing data...')
    for idx, (form, embedding) in enumerate(zip(forms, embeddings)):
        if form in sfs:
            sf = form
            sf_chart_data[sf]['sf_embeddings'].append(embedding)
        else:
            assert form in lf_to_sf_map.keys()
            sf = lf_to_sf_map[form]
            sf_chart_data[sf]['lf_embeddings'][form].append(embedding)

    print('Now Plotting Charts loading data.  Organizing data...')
    for sf in sfs:
        plt.figure()
        this_lfs = list(sf_chart_data[sf]['lf_embeddings'].keys())
        colors = cm.rainbow(np.linspace(0, 1, len(this_lfs) + 1))

        values = [sf] + this_lfs

        xs, ys = [], []
        for value in values:
            if value == sf:
                embeddings = sf_chart_data[sf]['sf_embeddings']
            else:
                embeddings = sf_chart_data[sf]['lf_embeddings'][value]

            x, y = extract_dims(embeddings)
            xs.append(x)
            ys.append(y)
        for x, y, value, color in zip(xs, ys, values, colors):
            plt.scatter(x, y, color=color, label=value)

        plt.xlabel('First Component')
        plt.ylabel('Second Component')
        plt.title('Scatter of First Two Principal Components for {}'.format(sf))
        plt.legend(bbox_to_anchor=(1.04,1), loc='upper left', fontsize='x-small')

        out_png = 'visualizations/{}.png'.format(sf)
        print('Saving image to {}'.format(out_png))
        plt.savefig(out_png, bbox_inches='tight')
