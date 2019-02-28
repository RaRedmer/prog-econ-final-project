# coding: utf-8

import os.path
import time
from contextlib import contextmanager
from datetime import datetime
from code.parameter_optimization import *
from code.apply_lgbm import *
from code.data_functions import *
from code.initiate_logging import *
from code.visualizations import *
import json
import warnings
from bayes_opt import BayesianOptimization


# warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

logger = init_logging()

@contextmanager
def runtime(title):
    start = time.time()
    yield
    logger.info("{} - runtime: {:.0f}s".format(title, time.time() - start))
    gc.collect()


num_rows = 20000
# num_rows = None
with runtime("Get training data"):
    data = get_application_train(num_rows)

with runtime("Enrich by Credit Bureau Data"):
    bureau = get_bureau_and_balance(num_rows)
    logger.info("Credit Bureau data shape: {}".format(bureau.shape))
    data = data.join(bureau, how='left', on='SK_ID_CURR')
    del bureau

with runtime("Enrich by Previous applications"):
    prev = get_previous_applications(num_rows)
    logger.info("Previous applications shape: {}".format(prev.shape))
    data = data.join(prev, how='left', on='SK_ID_CURR')
    del prev

with runtime("Enrich by Pos-cash balance"):
    pos = get_pos_cash(num_rows)
    logger.info("Pos-cash balance shape: {}".format(pos.shape))
    data = data.join(pos, how='left', on='SK_ID_CURR')
    del pos

with runtime("Enrich by installments payments"):
    ins = get_installments_payments(num_rows)
    logger.info("Installments payments shape: {}".format(ins.shape))
    data = data.join(ins, how='left', on='SK_ID_CURR')
    del ins

with runtime("Enrich by credit card balance"):
    cc = get_credit_card_balance(num_rows)
    logger.info("Credit card balance shape: {}".format(cc.shape))
    data = data.join(cc, how='left', on='SK_ID_CURR')
    del cc

# non-id columns
feats = [col for col in data.columns if col not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]
train_df = data[data['TARGET'].notnull()][feats]
train_target = data[data['TARGET'].notnull()]['TARGET']
del data


params_path = 'utilities/lgbm_bayes_params.json'
if os.path.isfile(params_path):
    with open(params_path) as json_file:
        params = json.loads(json_file.read())
else:
    params = optimize_lgbm_params(train_df, train_target)

with runtime('KFold LightGBM'):
    feat_importance, models, scores, oof_preds = kfold_lightgbm(train_df, train_target, params,  num_folds=5)

with open("output/auc_scores.txt", "a") as text_file:
    text_file.write(r"{time} - AUC scores before selection: {scores}; mean={mean}; variables={vars}" \
                    .format(time=datetime.now().strftime("%H:%M:%S"),
                            scores=scores,
                            mean=np.mean(scores),
                            vars=train_df.shape[1]))
    text_file.write("\n")

# create Folds
folds = KFold(n_splits=5, shuffle=True, random_state=1000)
folds_idx = [(train_id, target_id) for train_id, target_id in folds.split(train_df, train_target)]

logger.info('Visualize Feature Importances')
visualize_roc_curve(actuals=train_target, predictions=oof_preds, folds_ids=folds_idx, filename='roc_curve_all_variables')
visualize_shapley_values(feat_importance)
visualize_importances(feat_importance)

explainer = shap.TreeExplainer(models[np.argmax(scores)])
shap_values = explainer.shap_values(train_df)

percentile = 0.1
best_features = feat_importance[["feature", "shap_values"]].groupby("feature")["shap_values"].agg(['mean']) \
                                                           .sort_values(by="mean", ascending=False).reset_index()

logger.info('Generate SHAP-Summary Plot')
shap.summary_plot(shap_values, train_df, max_display=30, show=False)
plt.savefig('output/shap_summary_plot.png', bbox_inches="tight")
plt.clf()

logger.info('Generate SHAP-Dependency Plots')
for placing, feat in enumerate(best_features[:5]["feature"]):
    shap.dependence_plot(feat, shap_values, train_df, show=False)
    plt.savefig('output/shap_depend_{top_num}_{feature}_.png'.format(feature=feat.split(' ')[0], top_num=str(placing + 1)),
                bbox_inches="tight")
    plt.clf()

best_features = best_features[:int(best_features.shape[0]*percentile)]["feature"].values
with runtime("Run LightGBM with selected features"):
    train_df = train_df[best_features]
    feat_importance, models, scores, oof_preds = kfold_lightgbm(train_df, train_target, params, num_folds=5)

with open("output/auc_scores.txt", "a") as text_file:
    text_file.write(r"{time} - AUC scores after selection: {scores}; mean={mean}; variables={vars}; new variables={new_vars}" \
                    .format(time=datetime.now().strftime("%H:%M:%S"),
                            scores=scores,
                            mean=np.mean(scores),
                            vars=train_df.shape[1],
                            new_vars=len([feat for feat in best_features if 'new' in feat.lower()])))
    text_file.write("\n")
