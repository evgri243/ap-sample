"""
Library with print functions for mlhack
"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_roc(target, scores):
    """
    Plot ROC curve and print AUC

    Arguemnts:
    target (list float) - class labels (e.g., 0, 1)
    scores (list float) - predictor scores
    """

    false_positives, true_positives, thresholds = roc_curve(target, scores)
    roc_auc = auc(false_positives, true_positives)

    plt.title("ROC")
    plt.plot(false_positives, true_positives, 'b', label='AUC = {0:0.2f}'.format(roc_auc))
    plt.legend(loc='lower right')
    plt.plot([0,1], [0,1], 'r--')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    return (true_positives, false_positives, thresholds)

def plot_classes_scatter(features, target):
    """
    Plots PCA-based scatter plot for classes

    Arguemnts:
    features (list list float) - features matrix
    target (list float) - class labels (e.g., 0, 1)    
    """

    pca = PCA(n_components=2)
    features_pc = pca.fit_transform(features)

    features_pc_0 = features_pc[target == 0, :]
    plt.scatter(features_pc_0[:,0], features_pc_0[:,1])

    features_pc_1 = features_pc[target == 1, :]
    plt.scatter(features_pc_1[:,0], features_pc_1[:,1])

    plt.title("ROC")
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    
    plt.show()