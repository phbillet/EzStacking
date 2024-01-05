import numpy as np
import pandas as pd
import sys
from pandas.api.types import is_numeric_dtype
from sklearn.base import BaseEstimator, is_classifier, is_regressor, TransformerMixin
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.inspection import partial_dependence
from sklearn.inspection import PartialDependenceDisplay
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, recall_score 

class Decorrelator(BaseEstimator, TransformerMixin):
    """
    Decorrelator is a class used to eliminate too correlated columns depending on a threshold during preprocessing.

    Parameters
    ----------
    threshold_corr
    """  
    def __init__(self, threshold):
        self.threshold = threshold
        self.correlated_columns = None

    def fit(self, X, y=None):
        correlated_features = set()  
        if not isinstance(X, pd.DataFrame):
           X = pd.DataFrame(X)
        corr_matrix = X.corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > self.threshold: # we are interested in absolute coeff value
                    colname = corr_matrix.columns[i]  # getting the name of column
                    correlated_features.add(colname)
        self.correlated_features = correlated_features
        return self

    def transform(self, X, y=None, **kwargs):
        return (pd.DataFrame(X)).drop(labels=self.correlated_features, axis=1)
    
class ColumnsDropper(BaseEstimator, TransformerMixin):
    """
    ColumnsDropper is a class used to drop columns from a dataset.

    Parameters
    ----------
    cols : list of columns dropped by the transformer
    """  
    def __init__(self, cols):
        if not isinstance(cols, list):
            self.cols = [cols]
        else:
            self.cols = cols

    def fit(self, X: pd.DataFrame, y: pd.Series):
        # there is nothing to fit
        return self

    def transform(self, X:pd.DataFrame):
        X = X.copy()
        return X[self.cols]    
    
