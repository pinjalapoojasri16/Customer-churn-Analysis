'''
In this file we are going to load the data and other ML pipeline techniques which are needed
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sys
import os
import seaborn as sns
import pickle
import logging
import warnings
warnings.filterwarnings('ignore')

from log_file import setup_logging
logger = setup_logging('main')

from sklearn.model_selection import train_test_split
from scipy.stats import skew
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler,MaxAbsScaler,Normalizer

from missing_value_techniques import MISSING_VALUE_TECHNIQUES
from variable_transformation import VARIABLE_TRANSFORMATION
from outlier_handling import OUTLIER_HANDLING
from feature_selection import FEATURE_SELECTION
from cat_to_num_techniques import CAT_TO_NUM_TECHNIQUES
from data_balancing import DATA_BALANCING
from All_models import common

class Task1:
    def __init__(self,path):
        try:
            self.path = path
            self.df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
            self.df.isnull().sum()
            file_handler = logging.FileHandler("logs/main.log")

            logger.info('Data loaded successfully')

            logger.info(f'Total rows in the data : {self.df.shape[0]}')
            logger.info(f'Total columns in the data : {self.df.shape[1]}')
            logger.info(f'{self.df.tail(5)}')

            self.y = self.df['Churn'].map({'Yes': 1, 'No': 0})
            self.X = self.df.drop(columns=['Churn', 'customerID'])

            logger.info(f'Independent Column(X): {self.X.shape}')
            logger.info(f'Dependent Column(y): {self.y.shape}')

            self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(self.X, self.y, test_size = 0.2, random_state = 42)

            logger.info(f'X_train Columns:{self.X_train.columns}')
            logger.info(f'X_test Columns:{self.X_test.columns}')

            logger.info(f'y_train Sample Data: {self.y_train.sample(5)}')
            logger.info(f'y_test Sample Data: {self.y_test.sample(5)}')

            logger.info(f'Training data size : {self.X_train.shape}')
            logger.info(f'Testing data size : {self.X_test.shape}')

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    def missing_values(self):
        try:
            logger.info('Selecting Missing Value Technique')
            logger.info(f'Before X_train: {self.X_train.isnull().sum()}')
            logger.info(f'Before X_test : {self.X_test.isnull().sum()}')

            techniques = {
                'mean': MISSING_VALUE_TECHNIQUES.mean_imputation,
                'median': MISSING_VALUE_TECHNIQUES.median_imputation,
                'mode': MISSING_VALUE_TECHNIQUES.mode_imputation,
                'knn': MISSING_VALUE_TECHNIQUES.knn_imputation,
                'ffill': MISSING_VALUE_TECHNIQUES.forward_fill,
                'bfill': MISSING_VALUE_TECHNIQUES.backward_fill,
                'random': MISSING_VALUE_TECHNIQUES.random_sample_imputation
            }

            X_train_filled = self.X_train.copy()
            X_test_filled = self.X_test.copy()

            best_technique_per_column = {}

            # Only columns that contain missing values
            missing_cols = self.X_train.columns[self.X_train.isnull().any()]

            for col in missing_cols:
                scores = {}

                for name, func in techniques.items():
                    try:
                        # Apply technique ONLY to the selected column
                        X_tr_col, _ = func(self.X_train[[col]].copy(),self.X_test[[col]].copy())

                        # Variance preservation metric (numeric columns only)
                        if pd.api.types.is_numeric_dtype(self.X_train[col]):
                            original_var = self.X_train[col].var()
                            new_var = X_tr_col[col].var()
                            var_diff = abs(original_var - new_var)
                        else:
                            # For categorical columns, use missing count change
                            var_diff = abs(self.X_train[col].isnull().sum() -X_tr_col[col].isnull().sum())

                        scores[name] = var_diff

                    except Exception:
                        continue  # skip incompatible techniques

                # Best technique for this column
                best_tech = min(scores, key=scores.get)
                best_technique_per_column[col] = best_tech

                # Apply best technique to the column
                X_train_filled[col], X_test_filled[col] = techniques[best_tech](self.X_train[[col]].copy(),self.X_test[[col]].copy())

                logger.info(f'Best imputation for {col}: {best_tech}')

            # Replace original datasets
            self.X_train = X_train_filled
            self.X_test = X_test_filled

            logger.info(f'After X_train missing:\n{self.X_train.isnull().sum()}')
            logger.info(f'After X_test missing:\n{self.X_test.isnull().sum()}')

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    def vt_out(self):
        try:
            #logger.info(f'{self.X_train.info()}')
            logger.info('Selecting Variable Transformation Technique')
            logger.info(f'X_train Columns : {self.X_train.columns}')
            logger.info(f'X_test Columns: {self.X_test.columns}')

            self.X_train_num = self.X_train.select_dtypes(exclude='object')
            self.X_train_cat = self.X_train.select_dtypes(include='object')
            self.X_test_num = self.X_test.select_dtypes(exclude='object')
            self.X_test_cat = self.X_test.select_dtypes(include='object')

            logger.info(f'X_train Numerical columns : {self.X_train_num.columns}')
            logger.info(f'X_train Categorical columns: {self.X_train_cat.columns}')
            logger.info(f'X_test Numerical columns: {self.X_test_num.columns}')
            logger.info(f'X_test Categorical columns: {self.X_test_cat.columns}')

            logger.info(f'X_train Numerical Shape: {self.X_train_num.shape}')
            logger.info(f'X_train Categorical Shape: {self.X_train_cat.shape}')
            logger.info(f'X_test Numerical Shape: {self.X_test_num.shape}')
            logger.info(f'X_test Categorical Shape: {self.X_test_cat.shape}')

            techniques = {
                'standard': VARIABLE_TRANSFORMATION.standard_scaling,
                'minmax': VARIABLE_TRANSFORMATION.minmax_scaling,
                'robust': VARIABLE_TRANSFORMATION.robust_scaling,
                'log': VARIABLE_TRANSFORMATION.log_transform,
                'power': VARIABLE_TRANSFORMATION.power_transform,
                'boxcox': VARIABLE_TRANSFORMATION.boxcox_transform,
                'yeojohnson': VARIABLE_TRANSFORMATION.yeojohnson_transform,
                'quantile': VARIABLE_TRANSFORMATION.quantile_transform
            }

            best_technique_per_feature = {}

            X_train_transformed = self.X_train_num.copy()
            X_test_transformed = self.X_test_num.copy()

            for col in self.X_train_num.columns:
                skew_scores = {}

                for name, func in techniques.items():
                    try:
                        # Apply transformation to SINGLE COLUMN
                        X_tr_col, _ = func(self.X_train_num[[col]].copy(),self.X_test_num[[col]].copy())
                        skew_val = skew(X_tr_col[col], nan_policy='omit')
                        skew_scores[name] = abs(skew_val)

                    except Exception:
                        continue  # skip failed transformation

                # Select best technique for this feature
                best_tech = min(skew_scores, key=skew_scores.get)
                best_technique_per_feature[col] = best_tech

                # Apply best transformation to that column
                X_train_transformed[col], X_test_transformed[col] = techniques[best_tech](self.X_train_num[[col]].copy(),self.X_test_num[[col]].copy())

                logger.info(f'Best transformation for {col}: {best_tech} | Skew: {skew_scores[best_tech]:.4f}')

            # Replace original data with transformed data
            self.X_train_num = X_train_transformed
            self.X_test_num = X_test_transformed

        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    def outlier(self):
        try:
            logger.info('Selecting Outlier Technique (Column-wise)')

            techniques = {
                'winsor': OUTLIER_HANDLING.winsorization,
                'clip': OUTLIER_HANDLING.clipping,
                'log': OUTLIER_HANDLING.log_outlier,
                'robust': OUTLIER_HANDLING.robust_scaling,
                'none': OUTLIER_HANDLING.no_outlier
            }

            X_train_out = self.X_train_num.copy()
            X_test_out = self.X_test_num.copy()

            for col in self.X_train_num.columns:
                losses = {}

                for name, func in techniques.items():
                    X_tr_col, X_te_col = func(self.X_train_num[[col]].copy(),self.X_test_num[[col]].copy())

                    # SAVE PLOTS
                    #OUTLIER_HANDLING.save_outlier_plot(X_tr_col,X_te_col,technique_name=f"{name}_{col}")

                    # Metric: distribution distortion
                    loss = abs(self.X_train_num[col].var() - X_tr_col[col].var())

                    losses[name] = loss

                # Select best technique
                best_tech = min(losses, key=losses.get)

                X_tr_col, X_te_col = techniques[best_tech](self.X_train_num[[col]].copy(),self.X_test_num[[col]].copy())

                X_train_out[col] = X_tr_col[col].values
                X_test_out[col] = X_te_col[col].values

                logger.info(f'Best outlier method for {col}: {best_tech}')

            self.X_train_num = X_train_out
            self.X_test_num = X_test_out

        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    def f_selection(self):
        try:
            logger.info(f'Before Feature Selection : {list(self.X_train_num.columns)} 'f'--> {self.X_train_num.shape}')
            logger.info('Selecting Feature Selection Technique')

            self.X_train_num = self.X_train_num.replace([np.inf, -np.inf], np.nan)
            self.X_test_num = self.X_test_num.replace([np.inf, -np.inf], np.nan)

            if self.X_train_num.isnull().sum().sum() > 0:
                logger.info('NaN detected → applying median imputation')
                self.X_train_num = self.X_train_num.fillna(self.X_train_num.median())
                self.X_test_num = self.X_test_num.fillna(self.X_train_num.median())

            techniques = {
                'variance': FEATURE_SELECTION.variance_threshold,
                'correlation': FEATURE_SELECTION.correlation_filter,
                'kbest': FEATURE_SELECTION.select_k_best,
                'rfe': FEATURE_SELECTION.rfe,
                'lasso': FEATURE_SELECTION.lasso,
                'tree': FEATURE_SELECTION.tree_based
            }

            feature_counts = {}

            for name, func in techniques.items():
                try:
                    result = func(self.X_train_num, self.X_test_num, self.y_train)

                    if result is None:
                        logger.info(f'{name} returned None → skipped')
                        continue

                    X_tr, _ = result

                    feature_counts[name] = X_tr.shape[1]
                    logger.info(f'{name} selected {X_tr.shape[1]} features')

                except Exception as e:
                    logger.info(f'{name} failed → skipped | Reason: {e}')
                    continue

            if not feature_counts:
                raise ValueError('No feature selection technique succeeded')

            best = min(feature_counts, key=feature_counts.get)
            logger.info(f'Best Feature Selection: {best}')

            self.X_train_num, self.X_test_num = techniques[best](self.X_train_num, self.X_test_num, self.y_train)

            logger.info(f'After Feature Selection : {list(self.X_train_num.columns)} 'f'--> {self.X_train_num.shape}')

        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    def cat_to_num(self):
        try:
            logger.info('Selecting Encoding (Categorical to Numerical) Technique Column-wise')

            cat_cols = self.X_train.select_dtypes(include='object').columns

            X_train_num = self.X_train.drop(columns=cat_cols)
            X_test_num = self.X_test.drop(columns=cat_cols)

            X_train_cat = self.X_train[cat_cols]
            X_test_cat = self.X_test[cat_cols]

            techniques = {
                'label': CAT_TO_NUM_TECHNIQUES.label_encoding,
                'onehot': CAT_TO_NUM_TECHNIQUES.one_hot_encoding,
                'frequency': CAT_TO_NUM_TECHNIQUES.frequency_encoding,
                'binary': CAT_TO_NUM_TECHNIQUES.binary_encoding,
                'ordinal': CAT_TO_NUM_TECHNIQUES.ordinal_encoding
            }

            X_train_enc = pd.DataFrame(index=X_train_cat.index)
            X_test_enc = pd.DataFrame(index=X_test_cat.index)

            for col in cat_cols:
                scores = {}

                for name, func in techniques.items():
                    try:
                        X_tr_col, _ = func(X_train_cat[[col]], X_test_cat[[col]])
                        scores[name] = X_tr_col.shape[1]
                    except:
                        continue

                best = min(scores, key=scores.get)
                logger.info(f'Best encoding for {col}: {best}')

                X_tr_col, X_te_col = techniques[best](X_train_cat[[col]],X_test_cat[[col]])

                X_train_enc = pd.concat([X_train_enc, X_tr_col], axis=1)
                X_test_enc = pd.concat([X_test_enc, X_te_col], axis=1)

            self.X_train = pd.concat([X_train_num, X_train_enc], axis=1)
            self.X_test = pd.concat([X_test_num, X_test_enc], axis=1)

            self.y_train = self.y_train.loc[self.X_train.index]
            self.y_test = self.y_test.loc[self.X_test.index]

            logger.info(f'Encoding completed. X_train shape: {self.X_train.shape}')

        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    def data_balance(self):
        try:
            logger.info('Selecting Data Balancing Technique (ROW-WISE)')

            X = self.X_train.reset_index(drop=True)
            y = self.y_train.reset_index(drop=True)

            # HARD ASSERT
            assert len(X) == len(y), "X and y are misaligned before balancing"

            techniques = {
                'none': DATA_BALANCING.no_balance,
                'ros': DATA_BALANCING.random_over_sampling,
                'rus': DATA_BALANCING.random_under_sampling,
                'smote': DATA_BALANCING.smote,
                'smote_tomek': DATA_BALANCING.smote_tomek,
                'smote_enn': DATA_BALANCING.smote_enn
            }

            scores = {}

            for name, func in techniques.items():
                X_res, y_res = func(X, y)

                if len(X_res) != len(y_res):
                    logger.info(f'{name} skipped due to mismatch')
                    continue

                model = LogisticRegression(max_iter=1000)

                f1 = cross_val_score(model,X_res,y_res,scoring='f1',cv=3).mean()

                scores[name] = f1
                logger.info(f'{name} F1 score: {round(f1, 4)}')

            best = max(scores, key=scores.get)
            logger.info(f'Best Balancing Method Selected: {best}')


            self.X_train_bal, self.y_train_bal = techniques[best](X, y)

            logger.info(f'After Balancing:\n{self.y_train_bal.value_counts()}')
            return self.X_train_bal, self.y_train_bal

        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    def feature_scaling(self,a,b):
        try:
            self.X_train_bal = a
            self.y_train_bal = b
            logger.info('Selecting Best Feature Scaling Technique')

            scalers = {
                'standard': StandardScaler(),
                'minmax': MinMaxScaler(),
                'robust': RobustScaler(),
                'maxabs': MaxAbsScaler(),
                'normalize': Normalizer()
            }

            scores = {}

            for name, scaler in scalers.items():
                try:
                    X_scaled = scaler.fit_transform(self.X_train_bal)

                    model = LogisticRegression(max_iter=1000)

                    f1 = cross_val_score(model,X_scaled,self.y_train_bal,scoring='f1',cv=3).mean()

                    scores[name] = f1
                    logger.info(f'{name} scaler -> F1 score: {round(f1, 4)}')

                except Exception as e:
                    logger.info(f'{name} scaler failed → skipped | {e}')

            best = max(scores, key=scores.get)
            logger.info(f'Best Scaling Technique Selected: {best}')

            # APPLY BEST SCALER
            best_scaler = scalers[best]

            self.X_train_scaled = pd.DataFrame(best_scaler.fit_transform(self.X_train_bal),columns=self.X_train_bal.columns)

            self.X_test_scaled = pd.DataFrame(best_scaler.transform(self.X_test),columns=self.X_test.columns)

            logger.info('Feature scaling applied successfully')

            with open('scaler_path.pkl', 'wb') as f:
                pickle.dump(best_scaler, f)

            logger.info(f'X_train columns : {self.X_train_scaled.columns}')
            logger.info(f'X_test columns : {self.X_test_scaled.columns}')
            # Calling common function
            common(self.X_train_scaled, self.y_train_bal, self.X_test_scaled, self.y_test)

        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

if __name__ == '__main__':
    try:
        obj = Task1('C:\\Users\\pushp\\Downloads\\Task 1')
        obj.missing_values()
        obj.vt_out()
        obj.outlier()
        obj.cat_to_num()
        obj.f_selection()
        a,b = obj.data_balance()
        obj.feature_scaling(a,b)

    except Exception as e:
        error_type,error_msg,error_line = sys.exc_info()
        logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

