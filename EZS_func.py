import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy import stats
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, recall_score 

# Technical functions

def plot_dataframe_structure(df):
    plt.figure()
    df.dtypes.value_counts().plot.pie(ylabel='')
    plt.title('Data types')
    plt.show()

def plot_categorical(df):
    plt.figure()
    df.nunique().plot.bar()
    plt.title('Number of different values')
    plt.show()
    
def duplicates(df):
    duplicate_rows_df = df[df.duplicated()]
    if duplicate_rows_df.shape[0] > 0:
       print('Number of rows before removing:', df.count()[0])
       print('Number of duplicate rows:', duplicate_rows_df.shape[0])
       df = df.drop_duplicates()
       print('Number of rows after removing:', df.count()[0])
    else:
       print('No duplicate rows.')
    return df

def drop_na(df, threshold_NaN):
    isna_stat = (df.isna().sum()/df.shape[0]).sort_values(ascending=True)
    drop_cols = []
    if isna_stat.max() > 0.0:
       drop_cols = np.array(isna_stat[isna_stat > threshold_NaN].index)
       print('Drop columns containing more than', threshold_NaN*100,'% of NaN:', drop_cols)
       df = df.drop(drop_cols, axis=1)
    else:
       print('No need to drop columns.')
    return df, drop_cols

def encoding(df, threshold_cat):
    encoded_cols = []
    for c in df.columns:
        if df[c].dtypes == 'object' or df[c].dtypes.name == 'category': 
           print('Encoding object column:', c)
           df[c] = df[c].factorize()[0].astype(np.int)
           encoded_cols.append([c, 'cat'])
        elif is_numeric_dtype(df[c]): 
             if df[c].unique().shape[0] > threshold_cat: 
                print('Encoding numeric column:', c)
                df[c]=(df[c]-df[c].mean())/df[c].std()
                encoded_cols.append([c, 'num'])
             else:
                print('Column ', c,' is categorical.')
                encoded_cols.append([c, 'cat'])
        else: 
             print('Unknown type ', df[c].dtypes,' for column:',c) 
             df = df.drop(c, axis=1)
             drop_cols = np.unique(np.concatenate((drop_cols, c)))
    encoded_cols = np.array(encoded_cols).T.tolist()
    return df, encoded_cols

def imputation(df):
    isna_stat = (df.isna().sum()/df.shape[0]).sort_values(ascending=True) 
    if isna_stat.max() > 0.0: 
       print('Imputing NaN using IterativeImputer') 
       df = pd.DataFrame(IterativeImputer(random_state=0).fit_transform(df), columns = df.columns)  
    else: 
       print('No need to impute data.')
    return df

def outliers(df, threshold_Z):
    Z_score = np.abs(stats.zscore(df)) 
    df_o_Z = df[(Z_score < threshold_Z).all(axis=1)]
    if df_o_Z.shape[0] != 0:
       print('Using Z_score, ', str(df.shape[0] - df_o_Z.shape[0]) ,' rows will be suppressed.') 
       df = df_o_Z
    else:
       print('Possible problem with outliers treatment, check threshold_Z') 
    return df

def plot_sns_corr_class(df, target_col):
    g = sns.PairGrid(df, hue=target_col) 
    g.map_upper(sns.scatterplot) 
    g.map_lower(sns.kdeplot) 
    g.map_diag(sns.kdeplot, lw=3, legend=False) 
    g.add_legend() 
    g.fig.suptitle('Pairwise data relationships', y=1.01) 
    plt.show()
    
def plot_sns_corr_regre(df, target_col):
    g = sns.PairGrid(df)
    g.map_upper(sns.scatterplot)
    g.map_lower(sns.kdeplot)
    g.map_diag(sns.kdeplot, lw=3, legend=False)
    g.fig.suptitle('Pairwise data relationships', y=1.01)
    plt.show()
    
def split(X, y, test_size=0.33, stratify=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=stratify, shuffle=True)
    return X_train, X_test, y_train, y_test
    
def downcast_dtypes(df):
    start_mem = df.memory_usage().sum() / 1024**2
    print(('Memory usage of dataframe is {:.2f}' 
                     'MB').format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')
    end_mem = df.memory_usage().sum() / 1024**2
    print(('Memory usage after optimization is: {:.2f}' 
                              'MB').format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) 
                                             / start_mem))
    return df

def shannon_entropy(y):
    from collections import Counter
    from numpy import log
    
    n = len(y)
    classes = [(clas,float(count)) for clas,count in Counter(y).items()]
    k = len(classes)
    
    H = -sum([ (count/n) * log((count/n)) for clas,count in classes]) #shannon entropy
    return H/log(k)

def format_test(df, dropped_cols, encoded_cols):
    df = df.drop(dropped_cols, axis=1)
    for c in encoded_cols:
        df[c] = df[c].factorize()[0]
    df = downcast_dtypes(df)
    return df_copy

def score_stacking_c(model, X_train, y_train, X_test, y_test):
    nb_estimators = len(model.estimators_)
    res_stack = np.empty((nb_estimators + 1, 3), dtype='object')
    for j in range(nb_estimators):
        res_stack [j, 0] = [*model.named_estimators_.keys()][j]
        m_t_x_train = model.transform(X_train)
        if model.transform(X_train).shape[1] == nb_estimators: 
           res_stack [j, 1] = accuracy_score(np.rint(m_t_x_train).T[j], y_train)
           res_stack [j, 2] = accuracy_score(np.rint(model.transform(X_test)).T[j], y_test)
        else: 
           res_stack [j, 1] = accuracy_score(m_t_x_train.reshape((X_train.shape[0],\
                                                                  nb_estimators,\
                                                                  y_train.unique().shape[0])).argmax(axis=2).T[j]\
                                             , y_train)
           res_stack [j, 2] = accuracy_score(model.transform(X_test).reshape((X_test.shape[0],\
                                                                              nb_estimators,\
                                                                              y_test.unique().shape[0])).argmax(axis=2).T[j],\
                                             y_test)
    res_stack [len(model.estimators_) , 0] = 'Stack'
    res_stack [len(model.estimators_) , 1] = accuracy_score(model.predict(X_train), y_train)
    res_stack [len(model.estimators_) , 2] = accuracy_score(model.predict(X_test), y_test)  
    models = res_stack.T[0]
    score_train = res_stack.T[1]
    score_test = res_stack.T[2]
    plt.figure(figsize=(8,5))
    plt.scatter(models, score_train, label='Train')
    plt.scatter(models, score_test, label='Test')
    plt.title('Model scores: accuracy')
    plt.xticks(rotation='vertical')
    plt.legend()
    plt.show()
    return res_stack

def score_stacking_r(model, X_train, y_train, X_test, y_test):
    nb_estimators = len(model.estimators_)
    res_stack = np.empty((nb_estimators + 1, 3), dtype='object')
    for j in range(nb_estimators):
        res_stack [j, 0] = [*model.named_estimators_.keys()][j]
        m_t_x_train = model.transform(X_train)
        if m_t_x_train.shape[1] == nb_estimators: 
           res_stack [j, 1] = r2_score(np.rint(m_t_x_train).T[j], y_train)
           res_stack [j, 2] = r2_score(np.rint(model.transform(X_test)).T[j], y_test)
        else: 
           res_stack [j, 1] = r2_score(m_t_x_train.reshape((X_train.shape[0],\
                                                            nb_estimators,\
                                                            y_train.unique().shape[0])).argmax(axis=2).T[j],\
                                       y_train)
           res_stack [j, 2] = r2_score(model.transform(X_test).reshape((X_test.shape[0],\
                                                                        nb_estimators,\
                                                                        y_test.unique().shape[0])).argmax(axis=2).T[j],\
                                       y_test)
    res_stack [len(model.estimators_) , 0] = 'Stack'
    res_stack [len(model.estimators_) , 1] = r2_score(model.predict(X_train), y_train)
    res_stack [len(model.estimators_) , 2] = r2_score(model.predict(X_test), y_test)  
    models = res_stack.T[0]
    score_train = res_stack.T[1]
    score_test = res_stack.T[2]
    plt.figure(figsize=(8,5))
    plt.scatter(models, score_train, label='Train')
    plt.scatter(models, score_test, label='Test')
    plt.title('Model scores: r2')
    plt.xticks(rotation='vertical')
    plt.legend()
    plt.show()
    return res_stack
    
def plot_perm_imp(model, X, y, scoring):
    result = permutation_importance(model, X, y, scoring, n_repeats=10)
    sorted_idx = result.importances_mean.argsort()
    fig, ax = plt.subplots()
    ax.barh(X.columns[sorted_idx], result.importances[sorted_idx].mean(axis=1).T)
    ax.set_title("Permutation Importances")
    fig.tight_layout()
    plt.show()
    
def plot_history(history):
    pd.DataFrame(history.history).plot(figsize=(12, 9))
    plt.title("Learning Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.show()