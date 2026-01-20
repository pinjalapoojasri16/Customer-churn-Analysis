import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sys
import os
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.impute import SimpleImputer, KNNImputer

from log_file import setup_logging
logger = setup_logging('missing_value_techniques')

class MISSING_VALUE_TECHNIQUES:
    @staticmethod
    def mean_imputation(X_train, X_test):
        try:
            logger.info(f'Mean imputation for missing values')
            logger.info(f'Before imputation X_train: {X_train.isnull().sum()}')
            logger.info(f'Before imputation X_test: {X_test.isnull().sum()}')

            X_tr = X_train.copy()
            X_te = X_test.copy()

            cols = X_tr.select_dtypes(exclude='object').columns
            imp = SimpleImputer(strategy='mean')

            self.X_train = pd.DataFrame(
                imputer.fit_transform(self.X_train),
                columns=self.X_train.columns
            )
            X_te[cols] = imp.transform(X_te[cols])

            logger.info(f'Mean imputation completed')
            logger.info(f'After imputation X_train: {X_tr.isnull().sum()}')
            logger.info(f'After imputation X_test: {X_te.isnull().sum()}')

            return X_tr, X_te

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')


    @staticmethod
    def median_imputation(X_train, X_test):
        try:
            logger.info(f'Median imputation for missing values')
            logger.info(f'Before imputation X_train: {X_train.isnull().sum()}')
            logger.info(f'Before imputation X_test: {X_test.isnull().sum()}')

            X_tr = X_train.copy()
            X_te = X_test.copy()

            cols = X_tr.select_dtypes(exclude='object').columns
            imp = SimpleImputer(strategy='median')

            X_tr[cols] = imp.fit_transform(X_tr[cols])
            X_te[cols] = imp.transform(X_te[cols])

            logger.info(f'Median imputation completed')
            logger.info(f'After imputation X_train: {X_tr.isnull().sum()}')
            logger.info(f'After imputation X_test: {X_te.isnull().sum()}')

            return X_tr, X_te

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')


    @staticmethod
    def mode_imputation(X_train, X_test):
        try:
            logger.info(f'Mode imputation for missing values')
            logger.info(f'Before imputation X_train: {X_train.isnull().sum()}')
            logger.info(f'Before imputation X_test: {X_test.isnull().sum()}')

            X_tr = X_train.copy()
            X_te = X_test.copy()

            imp = SimpleImputer(strategy='most_frequent')

            X_tr[:] = imp.fit_transform(X_tr)
            X_te[:] = imp.transform(X_te)

            logger.info(f'Mode imputation completed')
            logger.info(f'After imputation X_train: {X_tr.isnull().sum()}')
            logger.info(f'After imputation X_test: {X_te.isnull().sum()}')

            return X_tr, X_te

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')


    @staticmethod
    def knn_imputation(X_train, X_test):
        try:
            logger.info(f'KNN imputation for missing values')
            logger.info(f'Before imputation X_train: {X_train.isnull().sum()}')
            logger.info(f'Before imputation X_test: {X_test.isnull().sum()}')

            X_tr = X_train.copy()
            X_te = X_test.copy()

            cols = X_tr.select_dtypes(exclude='object').columns
            imp = KNNImputer(n_neighbors=5)

            X_tr[cols] = imp.fit_transform(X_tr[cols])
            X_te[cols] = imp.transform(X_te[cols])

            logger.info(f'KNN imputation completed')
            logger.info(f'After imputation X_train: {X_tr.isnull().sum()}')
            logger.info(f'After imputation X_test: {X_te.isnull().sum()}')

            return X_tr, X_te

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')


    @staticmethod
    def forward_fill(X_train, X_test):
        try:
            logger.info(f'Forward fill for missing values')
            logger.info(f'Before imputation X_train: {X_train.isnull().sum()}')
            logger.info(f'Before imputation X_test: {X_test.isnull().sum()}')

            X_tr = X_train.copy().ffill()
            X_te = X_test.copy().ffill()

            logger.info(f'Forward fill completed')
            logger.info(f'After imputation X_train: {X_tr.isnull().sum()}')
            logger.info(f'After imputation X_test: {X_te.isnull().sum()}')

            return X_tr, X_te

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    @staticmethod
    def backward_fill(X_train, X_test):
        try:
            logger.info(f'Backward fill for missing values')
            logger.info(f'Before imputation X_train: {X_train.isnull().sum()}')
            logger.info(f'Before imputation X_test: {X_test.isnull().sum()}')

            X_tr = X_train.copy().bfill()
            X_te = X_test.copy().bfill()

            logger.info(f'Backward fill completed')
            logger.info(f'After imputation X_train: {X_tr.isnull().sum()}')
            logger.info(f'After imputation X_test: {X_te.isnull().sum()}')

            return X_tr, X_te

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    @staticmethod
    def random_sample_imputation(X_train, X_test):
        try:
            logger.info('Random Sample Imputation for missing values')
            logger.info(f'Before imputation X_train: {X_train.isnull().sum()}')
            logger.info(f'Before imputation X_test: {X_test.isnull().sum()}')

            X_tr = X_train.copy()
            X_te = X_test.copy()

            for col in X_tr.columns:
                if X_tr[col].isnull().sum() > 0:
                    # sample from non-null values
                    random_samples = X_tr[col].dropna()

                    if len(random_samples) == 0:
                        continue

                    X_tr[col] = X_tr[col].apply(lambda x: np.random.choice(random_samples) if pd.isnull(x) else x)
                    X_te[col] = X_te[col].apply(lambda x: np.random.choice(random_samples) if pd.isnull(x) else x)

            logger.info('Random Sample Imputation Completed')
            logger.info(f'After imputation X_train: {X_tr.isnull().sum()}')
            logger.info(f'After imputation X_test: {X_te.isnull().sum()}')

            return X_tr, X_te

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')