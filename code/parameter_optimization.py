import numpy as np
import gc
import warnings


warnings.simplefilter(action='ignore', category=FutureWarning)


def optimize_lgbm_params(train_df, target_df):
    """Apply Bayesian Optimization to LightGBM parameters
        Args:
            train_df(pd.DataFrame): Training data
            target_df(pd.Series):  Target/ Test data

        Returns:
            best_params(dict): Optimized parameters for LGBM
    """
    def _lgbm_evaluate(**params):
        """Wrapper for KFold LGBM parameter evaluation
            Args:
                params(dict): Parameter to evaluate based on LGBM outcome

            Returns:
                roc_auc_score(float): ROC-AUC-value to optimize by Bayesian optimization
        """

        warnings.simplefilter('ignore')
        params['num_leaves'] = int(params['num_leaves'])
        params['max_depth'] = int(params['max_depth'])

        clf = LGBMClassifier(**params, n_estimators=10000, nthread=4)

        folds = KFold(n_splits=2, shuffle=True, random_state=1001)
        test_pred_proba = np.zeros(train_df.shape[0])
        for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df, target_df)):
            train_x, train_y = train_df[feats].iloc[train_idx], target_df.iloc[train_idx]
            valid_x, valid_y = train_df[feats].iloc[valid_idx], target_df.iloc[valid_idx]

            clf.fit(train_x, train_y,
                    eval_set=[(train_x, train_y), (valid_x, valid_y)], eval_metric='auc',
                    verbose=False, early_stopping_rounds=100)

            test_pred_proba[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]

            del train_x, train_y, valid_x, valid_y
            gc.collect()

        return roc_auc_score(target_df, test_pred_proba)

    # parameter ranges for optimization
    params = {'colsample_bytree': (0.8, 1),
              'learning_rate': (.015, .025),
              'num_leaves': (33, 35),
              'subsample': (0.8, 1),
              'max_depth': (7, 9),
              'reg_alpha': (.03, .05),
              'reg_lambda': (.06, .08),
              'min_split_gain': (.01, .03),
              'min_child_weight': (38, 40)}

    bo = BayesianOptimization(_lgbm_evaluate, params)
    bo.maximize(init_points=5, n_iter=5)

    best_params = bo.max['params']
    best_params['n_estimators'] = 10000
    best_params['nthread'] = 4
    best_params['num_leaves'] = int(best_params['num_leaves'])
    best_params['max_depth'] = int(best_params['max_depth'])

    return best_params


if __name__ == "__main__":
    import json
    from code.data_functions import *
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import KFold
    from lightgbm import LGBMClassifier
    from bayes_opt import BayesianOptimization

    data = get_application_train(main=True)

    bureau = get_bureau_and_balance(main=True)
    data = data.join(bureau, how='left', on='SK_ID_CURR')
    del bureau

    prev = get_previous_applications(main=True)
    data = data.join(prev, how='left', on='SK_ID_CURR')
    del prev

    pos = get_pos_cash(main=True)
    data = data.join(pos, how='left', on='SK_ID_CURR')
    del pos

    ins = get_installments_payments(main=True)
    data = data.join(ins, how='left', on='SK_ID_CURR')
    del ins

    cc = get_credit_card_balance(main=True)
    data = data.join(cc, how='left', on='SK_ID_CURR')
    del cc

    # non-id columns
    feats = [col for col in data.columns if col not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]
    train_df = data[data['TARGET'].notnull()][feats]
    train_target = data[data['TARGET'].notnull()]['TARGET']
    del data

    best_params = optimize_lgbm_params(train_df, train_target)
    with open('../utilities/lgbm_bayes_params.json', 'w') as file:
        json.dump(best_params, file)
