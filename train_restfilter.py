from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import argparse
import pandas as pd
from utils.load_dataset import load_data

if __name__=='__main__':
    #XGBClassifier
    parser = argparse.ArgumentParser(description='Pass required variables')
    parser.add_argument('--rest_train_path', type = str, help='Root path for cnn training datset', default='./dataset/rest_data/train')
    parser.add_argument('--save_model_path', type=str, help='Path to the pre-process rest features', default='./weights/Rest/Model/model.sav')

    args = parser.parse_args()

    X_train = load_data(args.rest_train_path + '/X_train.csv')
    X_test = load_data(args.rest_train_path + '/X_test.csv')
    Y_test = load_data(args.rest_train_path + '/Y_test.csv')
    Y_train = load_data(args.rest_train_path + '/Y_train.csv')
    
    classifier = XGBClassifier(learning_rate = 0.2, reg_lambda = 0.01)
    classifier.fit(X_train, Y_train)
    pred   = classifier.predict(X_test)

    print(pred)
    print(accuracy_score(Y_test, pred))
    report = classification_report(Y_test, pred)
    print(report)

    with open("xgboost_rest_filter.txt", 'a') as f:
        f.write(str(report))

    target_names=['MOTION', 'REST']
    cm = confusion_matrix(Y_test, pred)
    df_cm = pd.DataFrame(cm, index = target_names, columns = target_names)
    plt.figure(figsize=(8,8))
    ax = sns.heatmap(df_cm, cmap ="BuGn", annot=True, cbar=False, fmt='d', annot_kws={"size": 20})
    plt.setp(ax.get_xticklabels(), fontsize=16)
    plt.setp(ax.get_yticklabels(), fontsize=16)
    plt.xlabel("Predicted labels", fontweight='bold', fontsize=18)
    plt.ylabel("True labels", fontweight='bold', fontsize=18)
    plt.show()
    plt.savefig('Xgboost.png')


    #XGBoost classifier model saved
    with open(args.save_model_path, "wb") as f:
        pickle.dump(classifier, f)