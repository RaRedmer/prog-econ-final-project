import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold


# Display/plot feature importance
def visualize_importances(importances, filename='feature_importances', main=False):
    """ Visualize shapley values
        Args:
            importances(pd.DataFrame): Features and its Importance-Values
     """
    best_features = importances[["feature", "importance"]].groupby("feature")["importance"].agg(['mean', 'std']) \
        .sort_values(by="mean", ascending=False).head(40).reset_index()
    best_features.columns = ["feature", "mean importance", "err"]

    plt.figure(figsize=(8, 10))
    sns.barplot(x="mean importance", y="feature", xerr=best_features['err'], data=best_features)
    plt.title('LightGBM Features (average over folds)')
    plt.tight_layout()
    plot_path = 'output/{}.png'.format(filename)
    if main:
        plot_path = '../test/{}.png'.format(filename)
    plt.savefig(plot_path, bbox_inches="tight")
    plt.clf()


# Display/plot shapley values
def visualize_shapley_values(feat_importance, filename='feature_shap_importances', main=False):
    """ Visualize shapley values
        Args:
            feat_importance(pd.DataFrame): Features and its SHAP-Values
     """
    best_features = feat_importance[["feature", "shap_values"]].groupby("feature")["shap_values"].agg(['mean', 'std']) \
                                                               .sort_values(by="mean", ascending=False).head(40) \
                                                               .reset_index()
    best_features.columns = ["feature", "mean SHAP values", "err"]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="mean SHAP values", y="feature", xerr=best_features['err'], data=best_features)
    plt.title('LightGBM SHAP values (average over folds)')
    plt.tight_layout()
    plot_path = 'output/{}.png'.format(filename)
    if main:
        plot_path = '../test/{}.png'.format(filename)
    plt.savefig(plot_path, bbox_inches="tight")
    plt.clf()


def visualize_roc_curve(actuals, predictions, folds_ids, filename='roc_curve', main=False):
    """ Plot ROC curve
        Args:
            actuals (pd.DataFrame): Actual training data
            predictions(pd.Series): Predictions from training data
            folds_ids(list): IDs from folds as tuples
        """
    plt.figure(figsize=(6, 6))
    scores = []
    for n_fold, (train_id, target_id) in enumerate(folds_ids):

        false_pos, true_pos, thresholds = roc_curve(actuals.iloc[target_id], predictions[target_id])
        score = roc_auc_score(actuals.iloc[target_id], predictions[target_id])
        scores.append(score)
        plt.plot(false_pos, true_pos, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.4f)' % (n_fold + 1, score))

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
    false_pos, true_pos, thresholds = roc_curve(actuals, predictions)
    score = roc_auc_score(actuals, predictions)
    plt.plot(false_pos, true_pos, color='b', label='Avg ROC (AUC = %0.4f $\pm$ %0.4f)' % (score, np.std(scores)), lw=2, alpha=.8)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('LightGBM ROC Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()

    plot_path = 'output/{}.png'.format(filename)
    if main:
        plot_path = '../test/{}.png'.format(filename)
    plt.savefig(plot_path, bbox_inches="tight")
    plt.clf()
