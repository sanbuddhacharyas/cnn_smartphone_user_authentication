from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc, RocCurveDisplay
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns
import pandas as pd

def compute_metrics(y_truth, y_pred):
    '''
    Returns
        fpr : Increasing false positive rates such that element i is the false positive rate of predictions with score >= thresholds[i]
        tpr : Increasing true positive rates such that element i is the true positive rate of predictions with score >= thresholds[i].
        thresholds : Decreasing thresholds on the decision function used to compute fpr and tpr. thresholds[0] represents no instances being predicted and is arbitrarily set to max(y_score) + 1
        auc : Area Under the ROC Curve metric
    '''
    # calculate AUC
    auc = roc_auc_score(y_truth, y_pred)
    
    # calculate roc curve
    fpr, tpr, thresholds = roc_curve(y_truth, y_pred)
    
    return fpr, tpr, thresholds, auc


def find_optimal_threhold(Y_test, pred):
    #Plot ROC curve and find the threshold
    fpr, tpr, thresholds_roc, auc     = compute_metrics(Y_test, pred)
    precision, recall, thresholds_pre = precision_recall_curve(Y_test, pred)
    display                           = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc,  estimator_name='Siamese Triplet')

    gmean             = np.sqrt(tpr * (1 - fpr))
    optimal_idx       = np.argmax(gmean)
    optimal_threshold = thresholds_roc[optimal_idx]
    string            = '{:.3f}'.format(optimal_threshold)

    y_pred_logic = np.array(np.array(pred)>=optimal_threshold, dtype=np.int)
    acc          = accuracy_score(Y_test, y_pred_logic)

    display.plot()
    print(tpr[optimal_idx], fpr[optimal_idx])
    plt.title(f'Threshold=>{optimal_threshold}')
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], c='Red', edgecolors='face')
    plt.text(fpr[optimal_idx]+0.01, tpr[optimal_idx]-0.06, s=f'{string}')
    plt.show()
    plt.savefig('optimal_threshold.png')

    index,       = np.where(optimal_threshold==thresholds_pre)
    print(f"ACC==>{acc}| prec=>{precision[index]} | Recall={recall[index]} | Threshold=>{optimal_threshold} | TPR=>{tpr[optimal_idx]} | FPR=>{fpr[optimal_idx]}")

    # Report
    report           = classification_report(Y_test, y_pred_logic)
    print(report)
    # Confusion matrix
    target_names=['Intruder', 'Legitimate']
    cm = confusion_matrix(Y_test, y_pred_logic)
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