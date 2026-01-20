import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sys
import os
import seaborn as sns
import warnings
from scipy import stats

warnings.filterwarnings('ignore')

from log_file import setup_logging
logger = setup_logging('outlier_handling')

from sklearn.preprocessing import RobustScaler

class OUTLIER_HANDLING:
    # Create directory to save plots
    plot_dir = "plot_outliers"
    os.makedirs(plot_dir, exist_ok=True)

    @staticmethod
    def iqr_method(X_train, X_test):
        try:
            logger.info('IQR Method Started')
            X_tr = X_train.copy()
            X_te = X_test.copy()

            numeric_cols = X_tr.select_dtypes(include='number').columns
            Q1 = X_tr[numeric_cols].quantile(0.25)
            Q3 = X_tr[numeric_cols].quantile(0.75)
            IQR = Q3 - Q1

            mask = ~((X_tr[numeric_cols] < (Q1 - 1.5 * IQR)) | (X_tr[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)

            logger.info('IQR Method Finished')
            return X_tr.loc[mask], X_te

        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')
            return None, None

    @staticmethod
    def zscore_method(X_train, X_test):
        try:
            logger.info('ZScore Method Started')
            X_tr = X_train.copy()
            X_te = X_test.copy()

            numeric_cols = X_tr.select_dtypes(include='number').columns
            z_scores = np.abs(stats.zscore(X_tr[numeric_cols]))
            mask = (z_scores < 3).all(axis=1)

            logger.info('ZScore Method Finished')
            return X_tr.loc[mask], X_te

        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')
            return None, None

    @staticmethod
    def winsorization(X_train, X_test):
        try:
            logger.info('Winsorization Method Started')
            X_tr = X_train.copy()
            X_te = X_test.copy()

            numeric_cols = X_tr.select_dtypes(include='number').columns
            lower = X_tr[numeric_cols].quantile(0.05)
            upper = X_tr[numeric_cols].quantile(0.95)

            X_tr[numeric_cols] = X_tr[numeric_cols].clip(lower=lower, upper=upper, axis=1)
            X_te[numeric_cols] = X_te[numeric_cols].clip(lower=lower, upper=upper, axis=1)

            logger.info('Winsorization Method Finished')
            return X_tr, X_te

        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')
            return None, None

    @staticmethod
    def clipping(X_train, X_test):
        try:
            logger.info('Clipping Method Started')
            X_tr = X_train.copy()
            X_te = X_test.copy()

            numeric_cols = X_tr.select_dtypes(include='number').columns
            X_tr[numeric_cols] = X_tr[numeric_cols].clip(-3, 3)
            X_te[numeric_cols] = X_te[numeric_cols].clip(-3, 3)

            logger.info('Clipping Method Finished')
            return X_tr, X_te

        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')
            return None, None

    @staticmethod
    def log_outlier(X_train, X_test):
        try:
            logger.info('LogOutlier Method Started')
            X_tr = X_train.copy()
            X_te = X_test.copy()

            numeric_cols = X_tr.select_dtypes(include='number').columns

            # Shift to handle negative values
            min_val = min(X_tr[numeric_cols].min().min(), X_te[numeric_cols].min().min())
            if min_val < 0:
                shift = abs(min_val) + 1
                X_tr[numeric_cols] = X_tr[numeric_cols] + shift
                X_te[numeric_cols] = X_te[numeric_cols] + shift

            X_tr[numeric_cols] = np.log1p(X_tr[numeric_cols])
            X_te[numeric_cols] = np.log1p(X_te[numeric_cols])

            logger.info('LogOutlier Method Finished')
            return X_tr, X_te

        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')
            return None, None

    @staticmethod
    def robust_scaling(X_train, X_test):
        try:
            logger.info('Robust Scaling Started')

            X_tr = X_train.copy()
            X_te = X_test.copy()

            numeric_cols = X_tr.columns
            scaler = RobustScaler()

            X_tr[numeric_cols] = scaler.fit_transform(X_tr[numeric_cols])
            X_te[numeric_cols] = scaler.transform(X_te[numeric_cols])

            logger.info('Robust Scaling Finished')
            return X_tr, X_te

        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(
                f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}'
            )
            return None, None

    @staticmethod
    def no_outlier(X_train, X_test):
        try:
            logger.info('No Outlier Method Started')
            logger.info('No Outlier Method Finished')
            return X_train, X_test

        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')
            return None, None

    @staticmethod
    def save_outlier_plot(X_train, X_test, technique_name):
        try:
            numeric_cols = X_train.select_dtypes(include='number').columns
            for col in numeric_cols:
                plt.figure(figsize=(12, 5))

                # Train plot
                plt.subplot(1, 2, 1)
                sns.boxplot(x=X_train[col])
                plt.title(f"Train - {col} ({technique_name})")

                # Test plot
                plt.subplot(1, 2, 2)
                sns.boxplot(x=X_test[col])
                plt.title(f"Test - {col} ({technique_name})")

                # Save figure
                filename = f"{OUTLIER_HANDLING.plot_dir}/{technique_name}_{col}.png"
                print(f"Saving plot: {filename}")  # Debug
                plt.savefig(filename)
                plt.close()

        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    @staticmethod
    def apply_all_techniques(X_train, X_test):
        techniques = ['iqr_method', 'zscore_method', 'winsorization', 'clipping', 'log_outlier', 'no_outlier']
        results = {}
        for tech in techniques:
            func = getattr(OUTLIER_HANDLING, tech)
            X_tr, X_te = func(X_train, X_test)
            if X_tr is not None:
                OUTLIER_HANDLING.save_outlier_plot(X_tr, X_te, tech)
                results[tech] = (X_tr, X_te)
        return results