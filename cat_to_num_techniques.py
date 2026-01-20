import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sys
import os
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from log_file import setup_logging
logger = setup_logging('cat_to_num_techniques')

from sklearn.preprocessing import LabelEncoder
import category_encoders as ce

class CAT_TO_NUM_TECHNIQUES:
    @staticmethod
    def label_encoding(X_train, X_test):
        try:
            logger.debug('Label Encoding')
            logger.info(f'Before X_train: {X_train.head()}')
            logger.info(f'Before X_test: {X_test.head()}')

            X_tr = X_train.copy()
            X_te = X_test.copy()

            for col in X_tr.columns:
                le = LabelEncoder()
                le.fit(pd.concat([X_tr[col], X_te[col]]).astype(str))
                X_tr[col] = le.transform(X_tr[col].astype(str))
                X_te[col] = le.transform(X_te[col].astype(str))

            logger.debug('After Label Encoding')
            logger.info(f'After X_train: {X_tr.head()}')
            logger.info(f'After X_test: {X_te.head()}')

            return X_tr, X_te

        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    @staticmethod
    def one_hot_encoding(X_train, X_test):
        try:
            logger.debug('One Hot Encoding')
            logger.info(f'Before X_train: {X_train.head()}')
            logger.info(f'Before X_test: {X_test.head()}')

            X = pd.concat([X_train, X_test], axis=0)
            X = pd.get_dummies(X, drop_first=True)

            logger.debug('After OneHot Encoding')
            logger.info(f'After X_train: {X.iloc[:len(X_train)]}')
            logger.info(f'After X_test: {X.iloc[len(X_train):]}')

            return X.iloc[:len(X_train)], X.iloc[len(X_train):]

        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    @staticmethod
    def frequency_encoding(X_train, X_test):
        try:
            logger.debug('Frequency Encoding')
            logger.info(f'Before X_train: {X_train.head()}')
            logger.info(f'Before X_test: {X_test.head()}')

            X_tr = X_train.copy()
            X_te = X_test.copy()

            for col in X_tr.columns:
                freq = X_tr[col].value_counts()
                X_tr[col] = X_tr[col].map(freq)
                X_te[col] = X_te[col].map(freq).fillna(0)

            logger.debug('After Frequency Encoding')
            logger.info(f'After X_train: {X_tr.head()}')
            logger.info(f'After X_test: {X_te.head()}')

            return X_tr, X_te

        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    @staticmethod
    def binary_encoding(X_train, X_test):
        try:
            logger.debug('Binary Encoding')
            logger.info(f'Before X_train: {X_train.head()}')
            logger.info(f'Before X_test: {X_test.head()}')

            nominal_cols = X_train.select_dtypes(include='object').columns
            encoder = ce.BinaryEncoder(cols=nominal_cols)

            X_tr = encoder.fit_transform(X_train)
            X_te = encoder.transform(X_test)

            logger.debug('After Binary Encoding')
            logger.info(f'After X_train: {X_tr.head()}')
            logger.info(f'After X_test: {X_te.head()}')

            return X_tr, X_te

        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    @staticmethod
    def ordinal_encoding(X_train, X_test):
        try:
            logger.debug('Ordinal Encoding')
            logger.info(f'Before X_train: {X_train.head()}')
            logger.info(f'Before X_test: {X_test.head()}')

            X_tr = X_train.copy()
            X_te = X_test.copy()

            ordinal_cols = {
                'Contract': ['Month-to-month', 'One year', 'Two year'],
                'InternetService': ['No', 'DSL', 'Fiber optic'],
                'PaymentMethod': ['Electronic check','Mailed check','Bank transfer (automatic)'],
                'DeviceType': ['Old Device', 'New Device'],
                'Region': ['Rural', 'Sub Urban', 'Urban']
            }

            for col, order in ordinal_cols.items():
                if col in X_tr.columns:
                    mapping = {v: i for i, v in enumerate(order)}
                    X_tr[col] = X_tr[col].map(mapping)
                    X_te[col] = X_te[col].map(mapping)

            logger.debug('After Ordinal Encoding')
            logger.info(f'After X_train: {X_tr.head()}')
            logger.info(f'After X_test: {X_te.head()}')

            return X_tr, X_te

        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')