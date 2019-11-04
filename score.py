import numpy as np
import pandas as pd
import distance
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


def read_csv(path):
    return pd.read_csv(path)

def predict_merge(df, model):
    df = df[['lf_1', 'lf_2', 'Should Merge?']]
    y_score = []
    for index, row in df.iterrows():
        y_score.append(1 - model.aligned_edit_distance(row['lf_1'].split(), row['lf_2'].split())[0])
        # y_score.append(model.jaccard_overlap(row['lf_1'].split(), row['lf_2'].split()))
    y_true = df['Should Merge?'].to_numpy()
    return y_true, np.array(y_score)


if __name__ == '__main__':
    df = read_csv('./expansion_etl/data/ml/merge_dataset_annotated.csv')
    print(df.columns)
    dis = distance
    y_true, y_score = predict_merge(df, dis)
    roc_auc = roc_auc_score(y_true, y_score)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()




