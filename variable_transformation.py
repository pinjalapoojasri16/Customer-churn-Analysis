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
logger = setup_logging('variable_transformation')

from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler,PowerTransformer,QuantileTransformer
from scipy.stats import boxcox

class VARIABLE_TRANSFORMATION:
    @staticmethod
    def standard_scaling(X_train, X_test):
        try:
            logger.info('Standard Scaling')
            logger.info(f'Before X_train Data : {X_train.head()}')
            logger.info(f'Before X_test Data : {X_test.head()}')

            sc = StandardScaler()
            X_tr = pd.DataFrame(sc.fit_transform(X_train),columns=X_train.columns,index=X_train.index)
            X_te = pd.DataFrame(sc.transform(X_test),columns=X_test.columns,index=X_test.index)

            logger.info(f'Standard Scaling Completed')
            logger.info(f'After X_train Data : {X_tr.head()}')
            logger.info(f'After X_test Data : {X_te.head()}')

            return X_tr, X_te

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    @staticmethod
    def minmax_scaling(X_train, X_test):
        try:
            logger.info('MinMax Scaling')
            logger.info(f'Before X_train Data : {X_train.head()}')
            logger.info(f'Before X_test Data : {X_test.head()}')

            sc = MinMaxScaler()
            X_tr = pd.DataFrame(sc.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
            X_te = pd.DataFrame(sc.transform(X_test), columns=X_test.columns, index=X_test.index)

            logger.info(f'MinMax Scaling Completed')
            logger.info(f'After X_train Data : {X_tr.head()}')
            logger.info(f'After X_test Data : {X_te.head()}')

            return X_tr, X_te

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    @staticmethod
    def robust_scaling(X_train, X_test):
        try:
            logger.info('Robust Scaling')
            logger.info(f'Before X_train Data : {X_train.head()}')
            logger.info(f'Before X_test Data : {X_test.head()}')

            sc = RobustScaler()
            X_tr = pd.DataFrame(sc.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
            X_te = pd.DataFrame(sc.transform(X_test), columns=X_test.columns, index=X_test.index)

            logger.info(f'Robust Scaling Completed')
            logger.info(f'After X_train Data : {X_tr.head()}')
            logger.info(f'After X_test Data : {X_te.head()}')

            return X_tr, X_te

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    @staticmethod
    def log_transform(X_train, X_test):
        try:
            logger.info('Log Transformation')
            logger.info(f'Before X_train Data : {X_train.head()}')
            logger.info(f'Before X_test Data : {X_test.head()}')

            X_tr = X_train.copy()
            X_te = X_test.copy()

            for col in X_tr.columns:
                if (X_tr[col] >= 0).all():
                    X_tr[col] = np.log1p(X_tr[col])
                    X_te[col] = np.log1p(X_te[col])

            logger.info(f'Log Transformation Completed')
            logger.info(f'After X_train Data : {X_tr.head()}')
            logger.info(f'After X_test Data : {X_te.head()}')

            return X_tr, X_te

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')


    @staticmethod
    def power_transform(X_train, X_test):
        try:
            logger.info('Power Transformation')
            logger.info(f'Before X_train Data : {X_train.head()}')
            logger.info(f'Before X_test Data : {X_test.head()}')

            pt = PowerTransformer()
            X_tr = pd.DataFrame(pt.fit_transform(X_train),columns=X_train.columns,index=X_train.index)
            X_te = pd.DataFrame(pt.transform(X_test),columns=X_test.columns,index=X_test.index)

            logger.info(f'Power Transformation Completed')
            logger.info(f'After X_train Data : {X_tr.head()}')
            logger.info(f'After X_test Data : {X_te.head()}')

            return X_tr, X_te

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    @staticmethod
    def boxcox_transform(X_train, X_test):
        try:
            logger.info('Boxcox Transformation')
            logger.info(f'Before X_train Data : {X_train.head()}')
            logger.info(f'Before X_test Data : {X_test.head()}')

            X_tr = X_train.copy()
            X_te = X_test.copy()

            for col in X_tr.columns:
                if (X_tr[col] > 0).all():
                    X_tr[col], lam = boxcox(X_tr[col])
                    X_te[col] = boxcox(X_te[col], lmbda=lam)

            logger.info(f'Boxcox Transformation Completed')
            logger.info(f'After X_train Data : {X_tr.head()}')
            logger.info(f'After X_test Data : {X_te.head()}')

            return X_tr, X_te

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    @staticmethod
    def yeojohnson_transform(X_train, X_test):
        try:
            logger.info('Yeojohnson Transformation')
            logger.info(f'Before X_train Data : {X_train.head()}')
            logger.info(f'Before X_test Data : {X_test.head()}')

            pt = PowerTransformer(method='yeo-johnson')
            X_tr = pd.DataFrame(pt.fit_transform(X_train),columns=X_train.columns)
            X_te = pd.DataFrame(pt.transform(X_test),columns=X_test.columns)

            logger.info(f'Yeojohnson Transformation Completed')
            logger.info(f'After X_train Data : {X_tr.head()}')
            logger.info(f'After X_test Data : {X_te.head()}')

            return X_tr, X_te

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')


    @staticmethod
    def quantile_transform(X_train, X_test):
        try:
            logger.info('Quantile Transformation')
            logger.info(f'Before X_train Data : {X_train.head()}')
            logger.info(f'Before X_test Data : {X_test.head()}')

            qt = QuantileTransformer(output_distribution='normal')
            X_tr = pd.DataFrame(qt.fit_transform(X_train),columns=X_train.columns,index=X_train.index)
            X_te = pd.DataFrame(qt.transform(X_test),columns=X_test.columns,)

            logger.info(f'Quantile Transformation Completed')
            logger.info(f'After X_train Data : {X_tr.head()}')
            logger.info(f'After X_test Data : {X_te.head()}')

            return X_tr, X_te

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')