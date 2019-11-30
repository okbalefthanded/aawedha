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
    for fld in range(nfolds):
        fpr = results['fpr'][0][fld]
        tpr = results['tpr'][0][fld]
        roc_auc = results['auc'][0, fld]  # (subject, fold)
        tprs.append(interp(mean_fpr, fpr, tpr))
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label=f'ROC fold {fld} AUC = {roc_auc}')

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2,
             color='r', label='Chance', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(results['auc'])
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


def plot_confusion_matrix(cm, target_names, title='Confusion matrix',
                          cmap="Blues"):
    '''
    '''
    cm = 100 * cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    df = pd.DataFrame(data=cm, columns=target_names, index=target_names)
    plt.figure()
    g = sns.heatmap(df, annot=True, fmt=".1f", linewidths=.5, vmin=0, vmax=100,
                    cmap=cmap)
    g.set_title(title)
    g.set_ylabel('True label')
    g.set_xlabel('Predicted label')
    plt.show()
    return g


def plot_subjects_perf(results={}):
    '''
    '''
    plt.rcParams["figure.figsize"] = (10, 5)
    labels = ['acc_mean_per_subj', 'auc_mean_per_subj']
    x = np.arange(len(results['acc_mean_per_subj']))  # the label locations
    width = 0.5  # the width of the bars
    y = iter([-1, 1])
    fig, ax = plt.subplots()
    for lb in labels:
        rects = ax.bar(x + (width/2)*next(y), results[lb], width, label=lb)
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{0:.2f}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('')
    ax.set_title('Performance per subject')
    ax.set_xticks(x)
    ax.set_xticklabels(x)
    ax.legend()

    fig.tight_layout()
    plt.show()
    return fig, ax
