# coding: utf-8

import numpy as np
import pandas as pd
import gc


def get_application_train(num_rows=None, main=False):
    """ Load application data and add new features to it
        Args:
            num_rows(int): Fix number of rows to load (optional)
            main(bool): Indicate that its being run as mainscript

        Returns:
            train_data(pd.DataFrame): Processed training data
    """
    path_train = 'original_data/application_train.csv'
    if main:
        path_train = '../' + path_train

    train_data = pd.read_csv(path_train, nrows=num_rows)

    # remove 4 applications with XNA in CODE_GENDER
    train_data = train_data[train_data['CODE_GENDER'] != 'XNA']
    # FLAG_DOC features on which kurtosis gets calculated
    docs = [col for col in train_data.columns if 'FLAG_DOC' in col]
    # features which are flagged as live
    live = [col for col in train_data.columns if ('FLAG_' in col) & ('FLAG_DOC' not in col) & ('_FLAG_' not in col)]

    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    train_data['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    # median income by organization, later used for map
    inc_by_org = train_data[['AMT_INCOME_TOTAL', 'ORGANIZATION_TYPE']].groupby('ORGANIZATION_TYPE').median()['AMT_INCOME_TOTAL']
    # create new variables
    train_data['NEW_ANNUITY_TO_INCOME_RATIO'] = train_data['AMT_ANNUITY'] / (1 + train_data['AMT_INCOME_TOTAL'])
    train_data['NEW_BIRTH_TO_EMPLOY_RATIO'] = train_data['DAYS_BIRTH'] / (1 + train_data['DAYS_EMPLOYED'])
    train_data['NEW_CAR_TO_BIRTH_RATIO'] = train_data['OWN_CAR_AGE'] / train_data['DAYS_BIRTH']
    train_data['NEW_CAR_TO_EMPLOY_RATIO'] = train_data['OWN_CAR_AGE'] / train_data['DAYS_EMPLOYED']
    train_data['NEW_CREDIT_TO_ANNUITY_RATIO'] = train_data['AMT_CREDIT'] / train_data['AMT_ANNUITY']
    train_data['NEW_CREDIT_TO_GOODS_RATIO'] = train_data['AMT_CREDIT'] / train_data['AMT_GOODS_PRICE']
    train_data['NEW_CREDIT_TO_INCOME_RATIO'] = train_data['AMT_CREDIT'] / train_data['AMT_INCOME_TOTAL']
    train_data['NEW_DOC_IND_KURT'] = train_data[docs].kurtosis(axis=1)
    train_data['NEW_EMPLOY_TO_BIRTH_RATIO'] = train_data['DAYS_EMPLOYED'] / train_data['DAYS_BIRTH']
    train_data['NEW_EMPLOY_TO_BIRTH-18_RATIO'] = train_data['DAYS_EMPLOYED'] / (train_data['DAYS_BIRTH'] + 18 * 365)
    train_data['NEW_EXT_SOURCES_GEO'] = train_data[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].apply(lambda x: np.exp(np.log(x[x > 0]).mean()), axis=1)
    train_data['NEW_EXT_SOURCES_MAD'] = train_data[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mad(axis=1, skipna=True)
    train_data['NEW_EXT_SOURCES_MAX'] = train_data[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].max(axis=1, skipna=True)
    train_data['NEW_EXT_SOURCES_MEAN'] = train_data[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1, skipna=True)
    train_data['NEW_EXT_SOURCES_MEDIAN'] = train_data[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].median(axis=1, skipna=True)
    train_data['NEW_EXT_SOURCES_MIN'] = train_data[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].min(axis=1, skipna=True)
    train_data['NEW_EXT_SOURCES_PROD'] = train_data[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].prod(axis=1, skipna=True, min_count=1)
    train_data['NEW_EXT_SOURCES_STD'] = train_data[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1, skipna=True)
    train_data['NEW_INC_BY_ORG'] = train_data['ORGANIZATION_TYPE'].map(inc_by_org)
    train_data['NEW_INC_PER_CHLD'] = train_data['AMT_INCOME_TOTAL'] / (1 + train_data['CNT_CHILDREN'])
    train_data['NEW_INCOME_CREDIT_PERC'] = train_data['AMT_INCOME_TOTAL'] / train_data['AMT_CREDIT']
    train_data['NEW_INCOME_PER_PERSON'] = train_data['AMT_INCOME_TOTAL'] / train_data['CNT_FAM_MEMBERS']
    train_data['NEW_INCOME_TO_ANNUITY_RATIO'] = train_data['AMT_INCOME_TOTAL'] / (1 + train_data['AMT_ANNUITY'])
    train_data['NEW_LIVE_IND_SUM'] = train_data[live].sum(axis=1)
    train_data['NEW_PAYMENT_RATE'] = train_data['AMT_ANNUITY'] / train_data['AMT_CREDIT']
    train_data['NEW_PHONE_TO_BIRTH_RATIO'] = train_data['DAYS_LAST_PHONE_CHANGE'] / train_data['DAYS_BIRTH']
    train_data['NEW_PHONE_TO_EMPLOYED_RATIO'] = train_data['DAYS_LAST_PHONE_CHANGE'] / train_data['DAYS_EMPLOYED']

    # categorical features with binary encode
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        train_data[bin_feature], uniques = pd.factorize(train_data[bin_feature])

    # categorical features as dummies
    categorical_columns = [col for col in train_data.columns if train_data[col].dtype == 'object']
    train_data = pd.get_dummies(train_data, columns=categorical_columns, dummy_na=True)
    # unused features
    dropcolum = ['FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_4',
                 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7',
                 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10',
                 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13',
                 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16',
                 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19',
                 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21']
    train_data = train_data.drop(dropcolum, axis=1)
    return train_data


def get_bureau_and_balance(num_rows=None, main=False):
    """ Load and aggregate bureau/balance data
        (Monthly balances of previous credits in Credit Bureau)
        Args:
            num_rows(int): Fix number of rows to load (optional)
            main(bool): Indicate that its being run as mainscript

        Returns:
            bureau_agg(pd.DataFrame): Aggregated bureau data
     """
    path_bureau = 'original_data/bureau.csv'
    path_bb = 'original_data/bureau_balance.csv'
    if main:
        path_bureau, path_bb = '../' + path_bureau, '../' + path_bb

    bureau = pd.read_csv(path_bureau, nrows=num_rows)
    bb = pd.read_csv(path_bb, nrows=num_rows)
    # create dummies for categorical features for bureau_balance
    bb_columns = list(bb.columns)
    bb_cat_columns = [col for col in bb.columns if bb[col].dtype == 'object']
    bb = pd.get_dummies(bb, columns=bb_cat_columns, dummy_na=True)
    bb_cat = [col for col in bb.columns if col not in bb_columns]
    # create dummies for categorical features for bureau
    bureau_columns = list(bureau.columns)
    bureau_categorical_columns = [col for col in bureau.columns if bureau[col].dtype == 'object']
    bureau = pd.get_dummies(bureau, columns=bureau_categorical_columns, dummy_na=True)
    bureau_cat = [col for col in bureau.columns if col not in bureau_columns]

    # aggregations for bureau balance
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    # set index to aggregation columns
    bb_agg.columns = pd.Index([col[0] + "_" + col[1].upper() for col in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace=True)
    del bb, bb_agg

    # bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': ['mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }
    # bureau and bureau_balance categorical features
    cat_aggregations = {cat: ['mean'] for cat in bureau_cat}
    for cat in bb_cat:
        cat_aggregations[cat + "_MEAN"] = ['mean']

    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + col[0] + "_" + col[1].upper() for col in bureau_agg.columns.tolist()])
    # aggregating active credits
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    # set index to aggregation columns
    active_agg.columns = pd.Index(['ACTIVE_' + col[0] + "_" + col[1].upper() for col in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg

    # aggregating closed credits
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    # set index to aggregation columns
    closed_agg.columns = pd.Index(['CLOSED_' + col[0] + "_" + col[1].upper() for col in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg, bureau
    return bureau_agg


def get_previous_applications(num_rows=None, main=False):
    """ Load and aggregate previous application data and add new features to it
        (All previous applications for Home Credit loans of clients who have loans)
        Args:
            num_rows(int): Fix number of rows to load (optional)
            main(bool): Indicate that its being run as mainscript

        Returns:
            prev_agg(pd.DataFrame): Aggregated previous application data
     """
    path_train = 'original_data/previous_application.csv'
    if main:
        path_train = '../' + path_train

    prev = pd.read_csv(path_train, nrows=num_rows)

    # create dummies for categorical features
    original_columns = list(prev.columns)
    categorical_columns = [col for col in prev.columns if prev[col].dtype == 'object']
    prev = pd.get_dummies(prev, columns=categorical_columns, dummy_na=True)
    cat_cols = [c for c in prev.columns if c not in original_columns]

    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)
    #  new feature
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    # features to perform aggregations on
    num_aggregations = {
        'AMT_ANNUITY': ['max', 'mean'],
        'AMT_APPLICATION': ['max', 'mean'],
        'AMT_CREDIT': ['max', 'mean'],
        'APP_CREDIT_PERC': ['max', 'mean'],
        'AMT_DOWN_PAYMENT': ['max', 'mean'],
        'AMT_GOODS_PRICE': ['max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['max', 'mean'],
        'RATE_DOWN_PAYMENT': ['max', 'mean'],
        'DAYS_DECISION': ['max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    # previous applications categorical features
    cat_aggregations = {cat: ['mean'] for cat in cat_cols}
    # aggregation based on dict
    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + col[0] + "_" + col[1].upper() for col in prev_agg.columns.tolist()])
    # approved applications
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + col[0] + "_" + col[1].upper() for col in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
    # refused applications
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + col[0] + "_" + col[1].upper() for col in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    del refused, refused_agg, approved, approved_agg, prev
    return prev_agg


def get_pos_cash(num_rows=None, main=False):
    """ Load and aggregate point of sales and cash loans balance data
        (Monthly balance snapshots of previous credit cards that the applicant has with Home Credit)
        Args:
            num_rows(int): Fix number of rows to load (optional)
            main(bool): Indicate that its being run as mainscript

        Returns:
            pos_agg(pd.DataFrame): Aggregated pos_cash_balance data
     """
    path_pos = 'original_data/POS_CASH_balance.csv'
    if main:
        path_pos = '../' + path_pos
    pos = pd.read_csv(path_pos, nrows=num_rows)
    # create dummies for categorical features
    original_columns = list(pos.columns)
    categorical_columns = [col for col in pos.columns if pos[col].dtype == 'object']
    pos = pd.get_dummies(pos, columns=categorical_columns, dummy_na=True)
    cat_cols = [c for c in pos.columns if c not in original_columns]

    # features to perform aggregation on
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    # conduct aggregation
    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + col[0] + "_" + col[1].upper() for col in pos_agg.columns.tolist()])
    # count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos
    return pos_agg


def get_installments_payments(num_rows=None, main=False):
    """ Load and aggregate installment payment data, and add further features
        (Repayment history for the previously disbursed credits in Home Credit related to the loans)
        Args:
            num_rows(int): Fix number of rows to load (optional)
            main(bool): Indicate that its being run as mainscript

        Returns:
            ins_agg(pd.DataFrame): Aggregated installment payment data
     """
    # Read original_data and merge
    path_ins = 'original_data/installments_payments.csv'
    if main:
        path_ins = '../' + path_ins
    ins = pd.read_csv(path_ins, nrows=num_rows)

    original_columns = list(ins.columns)
    categorical_columns = [col for col in ins.columns if ins[col].dtype == 'object']
    ins = pd.get_dummies(ins, columns=categorical_columns, dummy_na=True)
    cat_cols = [c for c in ins.columns if c not in original_columns]

    # percentage and difference paid in each installment
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    # days past due (DPD) and days before due (DBD)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    # conduct aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['mean', 'var'],
        'PAYMENT_DIFF': ['mean', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    # conduct aggregation based on dict
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + col[0] + "_" + col[1].upper() for col in ins_agg.columns.tolist()])
    # count installment accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins
    return ins_agg


def get_credit_card_balance(num_rows=None, main=False):
    """ Load and aggregate credit card balance data
        (Monthly balance snapshots of previous credit cards that the applicant has with Home Credit)
        Args:
            num_rows(int): Fix number of rows to load (optional)
            main(bool): Indicate that its being run as mainscript

        Returns:
            cc_agg(pd.DataFrame): Aggregated credit card balance data
     """
    path_cc = 'original_data/credit_card_balance.csv'
    if main:
        path_cc = '../' + path_cc

    cc = pd.read_csv(path_cc, nrows=num_rows)
    categorical_columns = [col for col in cc.columns if cc[col].dtype == 'object']
    cc = pd.get_dummies(cc, columns=categorical_columns, dummy_na=True)

    # generic aggregations
    cc.drop(['SK_ID_PREV'], axis=1, inplace=True)
    cc_agg = cc.groupby('SK_ID_CURR').agg(['max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + col[0] + "_" + col[1].upper() for col in cc_agg.columns.tolist()])
    # count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    gc.collect()
    return cc_agg


if __name__ == "__main__":
    rows = 10000
    # test
    joined_data = get_application_train(num_rows=rows, main=True)
    joined_data.to_csv('../test/train_test.csv', index=False)

    bureau_and_balance = get_bureau_and_balance(num_rows=rows, main=True)
    bureau_and_balance.to_csv('../test/bureau_and_balance_test.csv', index=False)
    joined_data = joined_data.join(bureau_and_balance, how='left', on='SK_ID_CURR')
    del bureau_and_balance

    previous_applications = get_previous_applications(num_rows=rows, main=True)
    previous_applications.to_csv('../test/previous_application_test.csv', index=False)
    joined_data = joined_data.join(previous_applications, how='left', on='SK_ID_CURR')
    del previous_applications

    pos_cash = get_pos_cash(num_rows=rows, main=True)
    pos_cash.to_csv('../test/pos_cash_balance_test.csv', index=False)
    joined_data = joined_data.join(pos_cash, how='left', on='SK_ID_CURR')
    del pos_cash

    installments_payments = get_installments_payments(num_rows=rows, main=True)
    installments_payments.to_csv('../test/installments_payments_test.csv', index=False)
    joined_data = joined_data.join(installments_payments, how='left', on='SK_ID_CURR')
    del installments_payments

    credit_card_balance = get_credit_card_balance(num_rows=rows, main=True)
    credit_card_balance.to_csv('../test/credit_card_balance_test.csv', index=False)
    joined_data = joined_data.join(credit_card_balance, how='left', on='SK_ID_CURR')
    del credit_card_balance

    joined_data.to_csv('../test/joined_data_test.csv', index=False)
