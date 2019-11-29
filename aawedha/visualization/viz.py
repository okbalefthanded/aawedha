from sklearn.metrics import auc
from scipy import interp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


def plot_train_val_curve(model_history={}):
    '''
    '''
    ky = list(model_history.keys())
    for k in range(len(ky) // 2):
        plt.figure(k)
        plt.plot(model_history[ky[k]])
        plt.plot(model_history[ky[k + len(ky) // 2]])
        plt.title(ky[k].upper())
        plt.ylabel(ky[k].upper())
        plt.xlabel('Epoch')
        plt.legend(['Train', 'val'], loc='upper left')
        plt.show()


def plot_roc_curve(results={}, nfolds=4):
    '''
    '''
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    #
    fg = plt.figure()
    for fld in range(nfolds):
        fpr = results['fpr'][0][fld]
        tpr = results['tpr'][0][fld]
        roc_auc = results['auc'][0, fld]  # (subject, fold)
        tprs.append(interp(mean_fpr, fpr, tpr))
        fg.plot(fpr, tpr, lw=1, alpha=0.3,
                label=f'ROC fold {fld} AUC = {roc_auc}')

    fg.plot([0, 1], [0, 1], linestyle='--', lw=2,
            color='r', label='Chance', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(results['auc'])
    fg.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    fg.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    fg.xlim([-0.05, 1.05])
    fg.ylim([-0.05, 1.05])
    fg.xlabel('False Positive Rate')
    fg.ylabel('True Positive Rate')
    fg.title('Receiver operating characteristic example')
    fg.legend(loc="lower right")
    fg.show()
    return fg


def plot_confusion_matrix(cm, target_names, title='Confusion matrix',
                          cmap="Blues"):
    '''
    '''
    cm = 100 * cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    df = pd.DataFrame(data=cm, columns=target_names, index=target_names)
    g = sns.heatmap(df, annot=True, fmt=".1f", linewidths=.5, vmin=0, vmax=100,
                    cmap=cmap)
    g.set_title(title)
    g.set_ylabel('True label')
    g.set_xlabel('Predicted label')
    return g
