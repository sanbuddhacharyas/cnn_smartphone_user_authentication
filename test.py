import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from utils.load_dataset import load_cnn_training_dataset
from models.cnn_model import cnn
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report



import sys
sys.path.insert(0, '../')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pass required variables')
    parser.add_argument('--id', type=str, help = 'Legitimate id')
    parser.add_argument('--X', type = str, help="Data path")
    parser.add_argument('--Y',  type = str, help="label path")
    parser.add_argument('--model_path',  type = str, help="label path")
    args = parser.parse_args()

    X     = load_cnn_training_dataset(args.X)
    Y     = load_cnn_training_dataset(args.Y)

    classifier       = load_model(args.model_path)
    pred             = classifier.predict(X)
    pred             = np.squeeze(pred>0.8).astype(np.int8)
    accuracy         = accuracy_score(Y.astype(np.int8), pred.astype(np.int8))
    report           = classification_report(Y.astype(np.int8), pred.astype(np.int8))
    print(report)

    print(Y, pred)

    target_names=['Intruder', 'Legitimate']
    cm = confusion_matrix(Y.astype(np.int8), pred.astype(np.int8))
    print(np.array(cm).shape)
    df_cm = pd.DataFrame(cm, index = target_names, columns = target_names)
    plt.figure(figsize=(8,8))
    ax = sns.heatmap(df_cm, cmap ="BuGn", annot=True, cbar=False, fmt='d', annot_kws={"size": 20})
    plt.setp(ax.get_xticklabels(), fontsize=16)
    plt.setp(ax.get_yticklabels(), fontsize=16)
    plt.xlabel("Predicted labels", fontweight='bold', fontsize=18)
    plt.ylabel("True labels", fontweight='bold', fontsize=18)
    plt.show()
    plt.savefig('CNN_model.png')