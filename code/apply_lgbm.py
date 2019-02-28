import numpy as np
import pandas as pd
import shap
import gc
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from code.visualizations import *
from code.initiate_logging import *


def kfold_lightgbm(train_df, train_target, params, num_folds):
    """  Apply LightGBM with KFold
        Arg:
            train_df(pd.DataFrame): Training-Data
            train_df(pd.Series): Test-Data
            params(dict): Parameter-Values for LightGBM
            num_folds(int): Number of Folds

        Returns:
            feat_importance(pd.DataFrame): Table with importances, SHAP-values, folds for each feature
            models(list): List of LightGBM-Model-objects
            scores(list): List of ROC-AUC-scores
            oof_preds(np.ndarray): Predictions
        """

    logger_lgbm = init_logging()
    logger_lgbm.info("Starting LightGBM. Train shape: {}".format(train_df.shape))
    # Cross validation model
    folds = KFold(n_splits=num_folds, shuffle=True, random_state=100)
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    feat_importance = pd.DataFrame()
    scores = []
    models = []
    # loop through folds
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df, train_target)):
        # index ids of fold for training
        train_x, train_y = train_df.iloc[train_idx], train_target.iloc[train_idx]
        # index ids of fold for validation
        valid_x, valid_y = train_df.iloc[valid_idx], train_target.iloc[valid_idx]

        clf = LGBMClassifier(**params)
        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)],
                eval_metric='auc', verbose=1000, early_stopping_rounds=300)

        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]

        # feature importance
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = train_df.columns.values
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["shap_values"] = abs(shap.TreeExplainer(clf).shap_values(valid_x)[:, :train_df.shape[1]]).mean(axis=0).T
        fold_importance_df["fold"] = n_fold + 1
        # add feature importances
        feat_importance = pd.concat([feat_importance, fold_importance_df], axis=0)

        scores.append(roc_auc_score(valid_y, oof_preds[valid_idx]))
        logger_lgbm.info('Fold %2d AUC : %.6f' % (n_fold + 1, scores[n_fold]))
        models.append(clf)
        del clf, train_x, train_y, valid_x, valid_y, fold_importance_df
        gc.collect()

    score = roc_auc_score(train_target, oof_preds)
    logger_lgbm.info('Full AUC score %.6f' % score)
    logger_lgbm.info('Mean AUC score %.6f' % np.mean(scores))
    return feat_importance, models, scores, oof_preds


if __name__ == "__main__":
    from code.data_functions import *
    import json
    from code.initiate_logging import *

    logger = init_logging()
    num_rows = 10000
    data = get_application_train(num_rows, main=True)

    bureau = get_bureau_and_balance(num_rows, main=True)
    data = data.join(bureau, how='left', on='SK_ID_CURR')
    del bureau

    prev = get_previous_applications(num_rows, main=True)
    data = data.join(prev, how='left', on='SK_ID_CURR')
    del prev

    pos = get_pos_cash(num_rows, main=True)
    data = data.join(pos, how='left', on='SK_ID_CURR')
    del pos

    ins = get_installments_payments(num_rows, main=True)
    data = data.join(ins, how='left', on='SK_ID_CURR')
    del ins

    cc = get_credit_card_balance(num_rows, main=True)
    data = data.join(cc, how='left', on='SK_ID_CURR')
    del cc

    # non-id columns
    feats = [col for col in data.columns if col not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]
    train_df = data[data['TARGET'].notnull()][feats]
    train_target = data[data['TARGET'].notnull()]['TARGET']
    del data

    folds = KFold(n_splits=5, shuffle=True, random_state=1001)
    folds_idx = [(trn_idx, val_idx) for trn_idx, val_idx in folds.split(train_df, train_target)]

    with open('../utilities/lgbm_bayes_params.json') as json_file:
        params = json.loads(json_file.read())

    feat_importance, models, scores, oof_preds = kfold_lightgbm(train_df, train_target, params,  num_folds=5)
