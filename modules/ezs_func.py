import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import sys
from ipywidgets import interact, interact_manual, fixed, IntText
from scipy import stats
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from pandas.api.types import is_numeric_dtype
from sklearn.base import is_classifier, is_regressor
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score, TimeSeriesSplit
from statsmodels.tsa.seasonal import seasonal_decompose, STL, MSTL
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.inspection import partial_dependence
from sklearn.inspection import PartialDependenceDisplay
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, recall_score 

# Technical functions
def reduce_schema(col, values):
    """
    Internal function used to drop columns from schema after feature elimination.
    """    
    schema = pd.read_csv('./schema.csv')
    schema = schema[~schema[col].isin(values)]
    schema.to_csv('./schema.csv', index=False)
    
def get_features():
    """
    Extract the categorical and numerical feature from the schema.
    Returns:
        features_cat: list of categorical features
        features_num: list of numerical features.
    """      
    schema = pd.read_csv('./schema.csv')
    features_cat = schema[schema['column_type']=='cat'].column_name
    features_num = schema[schema['column_type']=='num'].column_name
    return features_cat, features_num
    
# EDA
def plot_dataframe_structure(df):
    """
    Plot dataframe structure: It shows the different data types in the dataframe.
    Parameters:
        df: Pandas dataframe.
    Returns:
        Plotting.
    """
    plt.figure()
    df.dtypes.value_counts().plot.pie(ylabel='')
    plt.title('Data types')
    plt.show()

def plot_categorical(df):
    """
    Plot the number of different values for each categorical feature in the dataframe.
    Parameters:
        df: Pandas dataframe.
    Returns:
        Plotting.
    """
    plt.figure()
    df.nunique().plot.bar()
    plt.title('Number of different values')
    plt.show()
    
def duplicates(df):
    """
    Remove the duplicate rows from dataframe.
    Parameters:
        df: Pandas dataframe.
    Returns:
        df: Pandas dataframe without duplicate rows.
    """    
    duplicate_rows_df = df[df.duplicated()]
    if duplicate_rows_df.shape[0] > 0:
       print('Number of rows before removing:', df.count()[0])
       print('Number of duplicate rows:', duplicate_rows_df.shape[0])
       df = df.drop_duplicates()
       print('Number of rows after removing:', df.count()[0])
    else:
       print('No duplicate rows.')
    return df

def drop_na(df, target_col, threshold_NaN):
    """
    Remove the columns from dataframe containing NaN depending on threshold_NaN.
    Parameters:
        df: Pandas dataframe
        threshold_NaN: in [0, 1] from GUI.
    Returns:
        df: Pandas dataframe 
        drop_cols: list of dropped columns.
    """    
    isna_stat = (df.isna().sum()/df.shape[0]).sort_values(ascending=True)
    drop_cols = []
    if isna_stat.max() > 0.0:
       drop_cols = np.array(isna_stat[isna_stat > threshold_NaN].index)
       print('Drop columns containing more than', threshold_NaN*100,'% of NaN:', drop_cols)
       df = df.drop(drop_cols, axis=1)
    else:
       print('No need to drop columns.')
    
    return df, drop_cols

def encoding(df, threshold_cat, target_col):
    """
    Encode the data.
    Parameters:
        df: Pandas dataframe
        threshold_cat: integer, if the number of different values of a given column is less than this limit, 
                       this column is considered as categorical. 
    Returns:
        df: Pandas dataframe 
        encoded_cols: Pandas dataframe of columns with their encoding and range.
    """      
    encoded_cols = []
    for c in df.columns:
        if df[c].dtypes == 'object' or df[c].dtypes.name == 'category': 
           encoded_cols.append([c, 'cat', df[c].dropna().unique().tolist()])
           print('Encoding object column:', c)
           df[c] = df[c].factorize()[0].astype(np.int32)
        elif is_numeric_dtype(df[c]): 
             if df[c].unique().shape[0] > threshold_cat: 
                encoded_cols.append([c, 'num', [df[c].min(), df[c].max()]])
                print('Encoding numeric column:', c)
                df[c]=(df[c]-df[c].mean())/df[c].std()
             else:
                print('Column ', c,' is categorical.')
                encoded_cols.append([c, 'cat', df[c].dropna().unique().tolist()])
        else: 
             print('Unknown type ', df[c].dtypes,' for column:',c) 
             df = df.drop(c, axis=1)
             drop_cols = np.unique(np.concatenate((drop_cols, c)))
    encoded_cols = pd.DataFrame(encoded_cols, columns=['column_name', 'column_type', 'column_range'], dtype=object)
    encoded_cols = encoded_cols.loc[encoded_cols['column_name'] != target_col]
    encoded_cols.to_csv('schema.csv', index=False)
    return df, encoded_cols

def imputation(df):
    """
    Impute NaN in the dataframe using IterativeImputer.
    Parameters:
        df: Pandas dataframe.
    Returns:
        df: Pandas dataframe.
    """        
    isna_stat = (df.isna().sum()/df.shape[0]).sort_values(ascending=True) 
    if isna_stat.max() > 0.0: 
       print('Imputing NaN using IterativeImputer') 
       df = pd.DataFrame(IterativeImputer(random_state=0).fit_transform(df), columns = df.columns)  
    else: 
       print('No need to impute data.')
    return df

def outliers(df, threshold_Z):
    """
    Remove the outliers from dataframe according to Z_score.
    Parameters:
        df: Pandas dataframe
        threshold_Z: number from GUI. 
    Returns:
        df: Pandas dataframe. 
    """  
    Z_score = np.abs(stats.zscore(df)) 
    df_o_Z = df[(Z_score < threshold_Z).all(axis=1)]
    if df_o_Z.shape[0] != 0:
       print('Using Z_score, ', str(df.shape[0] - df_o_Z.shape[0]) ,' rows will be suppressed.') 
       df = df_o_Z
    else:
       print('Possible problem with outliers treatment, check threshold_Z') 
    return df

def correlated_columns(df, threshold_corr, target_col):
    """
    Display correlation matrix of features, and returns the list of the too correlated features
    according to threshold_corr.
    Parameters:
        df: Pandas dataframe
        threshold_corr: number from GUI
        target: target column.
    Returns:
        correlated_features: list of the features having a correlation greater than threshold_corr. 
    """  
    df = df.drop(target_col, axis=1)
    corr_matrix = df.corr() 
    correlated_features=[]
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold_corr: # we are interested in absolute coeff value
               colname = corr_matrix.columns[i]  # getting the name of column
               correlated_features.append(colname)
    correlated_features = list(dict.fromkeys(correlated_features))
    return correlated_features

def hierarchical_clustering(df, t):
    """
    Plot the hierarchical clustering of the features based on the Spearman rank-order correlations.
    Parameters:
        df: Pandas dataframe,
        t: distance threshold.
    Returns:
        Plotting.
    """
    # from: https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    corr = spearmanr(df).correlation

    # Ensure the correlation matrix is symmetric
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1)

    # We convert the correlation matrix to a distance matrix before performing
    # hierarchical clustering using Ward's linkage.
    distance_matrix = 1 - np.abs(corr)
    dist_linkage = hierarchy.ward(squareform(distance_matrix))
    dendro = hierarchy.dendrogram(
        dist_linkage, labels=df.columns.tolist(), ax=ax1, leaf_rotation=90
    )
    dendro_idx = np.arange(0, len(dendro["ivl"]))

    ax2.imshow(corr[dendro["leaves"], :][:, dendro["leaves"]])
    ax2.set_xticks(dendro_idx)
    ax2.set_yticks(dendro_idx)
    ax2.set_xticklabels(dendro["ivl"], rotation="vertical")
    ax2.set_yticklabels(dendro["ivl"])
    fig.tight_layout()
    plt.show()
    
    from collections import defaultdict

    cluster_ids = hierarchy.fcluster(dist_linkage, t=t, criterion="distance")
    print('cluster_ids = ', cluster_ids)
    cluster_id_to_feature_ids = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_ids):
        cluster_id_to_feature_ids[cluster_id].append(idx)
    selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
    selected_features_names = df.columns[selected_features].tolist()
    return selected_features_names

def plot_sns_corr_class(df, target_col):
    """
    Plot correlation information for classification problem (if Seaborn option is checked).
    Parameters:
        df: Pandas dataframe
        target_col: name of the target column. 
    Returns:
        Plotting. 
    """     
    g = sns.PairGrid(df, hue=target_col) 
    g.map_upper(sns.scatterplot) 
    g.map_lower(sns.kdeplot) 
    g.map_diag(sns.kdeplot, lw=3, legend=False) 
    g.add_legend() 
    g.fig.suptitle('Pairwise data relationships', y=1.01) 
    plt.show()
    
def plot_sns_corr_regre(df, target_col):
    """
    Plot correlation information for regression problem (if Seaborn option is checked).
    Parameters:
        df: Pandas dataframe
        target_col: name of the target column. 
    Returns:
        Plotting. 
    """      
    g = sns.PairGrid(df)
    g.map_upper(sns.scatterplot)
    g.map_lower(sns.kdeplot)
    g.map_diag(sns.kdeplot, lw=3, legend=False)
    g.fig.suptitle('Pairwise data relationships', y=1.01)
    plt.show()
    
def downcast_dtypes(df):
    """
    Compress the input dataframe.
    Parameters:
        df: Pandas dataframe.
    Returns:
        df: Pandas dataframe.
    """      
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
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df

def shannon_entropy(y):
    """
    Compute Shannon entropy of a dataset.
    Parameters:
        y: univariate Pandas dataframe.
    Returns:
        shannon entropy: float.
    """     
    from collections import Counter
    from numpy import log
    
    n = len(y)
    classes = [(clas,float(count)) for clas,count in Counter(y).items()]
    k = len(classes)
    
    H = -sum([ (count/n) * log((count/n)) for clas,count in classes]) #shannon entropy
    return H/log(k)
    
# Dataset splitting
def split(X, y, random_state, test_size=0.33, threshold_entropy=0.7, undersampling=False, undersampler=None):
    """
    Split dataframe into train and test sets.
    If the Shannon entropy of the target dataset is less than 0.7, RepeatedStratifiedKFold is used.
    Parameters:
        X: feature dataframe
        y: target dataframe.
    Returns:
        X_train: train feature dataframe 
        X_test: test feature dataframe
        y_train: train target dataframe
        y_test: test target dataframe.
    """
    s_e = shannon_entropy(y)
    if s_e < threshold_entropy:
       if undersampling: 
          if undersampler == 'Random': 
             from imblearn.under_sampling import RandomUnderSampler
             us = RandomUnderSampler()
          elif undersampler == 'Centroids': 
             from imblearn.under_sampling import ClusterCentroids
             us = ClusterCentroids()
          elif undersampler == 'AllKNN': 
             from imblearn.under_sampling import AllKNN
             us = AllKNN()
          elif undersampler == 'TomekLinks': 
             from imblearn.under_sampling import TomekLinks
             us = TomekLinks()
          else:
             print("Unknown undersampler")       
          X, y = us.fit_resample(X, y)
          print("Shannon Entropy = {:.4}, split using undersampler {} and RepeatedStratifiedKFold".format(s_e, undersampler)) 
       else: 
          print("Shannon Entropy = {:.4}, split using RepeatedStratifiedKFold".format(s_e)) 
       skfold = RepeatedStratifiedKFold(n_splits=5, random_state = random_state)
       # enumerate the splits and summarize the distributions
       for ind_train, ind_test in skfold.split(X, y):
           X_train, X_test = X.iloc[ind_train], X.iloc[ind_test]
           y_train, y_test = y.iloc[ind_train], y.iloc[ind_test] 
    else:    
       X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=None,\
                                                           shuffle=True, random_state = random_state)
    return X_train, X_test, y_train, y_test
    
# Modelling functions
def model_filtering(level_0, model_imp, nb_model, score_stack, threshold_score):
    """
    Suppress estimators from level 0 having a test score smaller than threshold_score (from score_stack), then 
    keep nb_model best estimators (according to model_imp).
    Parameters:
        level_0: list of estimators of level 0
        model_imp: sorted array of model importance
        nb_model : number of model to keep
        score_stack: accuracy of estimators on train and test sets in a tabular
        threshold_score : minimal score.
    Returns:
        list of filtered estimators of level 0.
    """
    # it is not possible to keep more models than we initially have
    if nb_model > len(level_0):
       nb_model = len(level_0)
    
    # keep model names and test scores
    score_stack = np.delete(np.delete(score_stack, 1, axis =1), -1, axis = 0)
    # keep models having test score greater than threshold_score 
    score_stack = score_stack[score_stack[:,1] > threshold_score]
    
    # it is not possible to keep more models than we have filtered    
    if nb_model > len(score_stack):
       nb_model = len(score_stack)
    
    # keep models (in importance array) having test score greater than threshold_score
    model_imp = model_imp[np.in1d(model_imp[:, 0], score_stack)]
    model_imp_f = model_imp[np.argpartition(model_imp[:,1], -nb_model)[-nb_model:]].T[0]
    
    return list(filter(lambda x: x[0] in model_imp_f, level_0))

def feature_filtering(feature_importance, nb_feature):
    """
    Separate features in two lists, the first one contains the nb_feature most important features, 
    the second one contains the complement.
    Parameters:
        feature_importance: array of features with their importance
        nb_feature: number of features we want to keep.
    Returns
        best_feature: list of nb_feature most important features
        worst_feature: list of the worst important features.
    """
    # check nb_feature
    if nb_feature > feature_importance.shape[0]:
       nb_feature = feature_importance.shape[0] 
    
    best_feature = feature_importance[np.argpartition(feature_importance[:,1], -nb_feature)[-nb_feature:]].T[0]
    worst_feature = list(set(feature_importance.T[0]) - set(best_feature))

    return best_feature, worst_feature

# Model evaluation functions
def score_stacking_c(model, X_train, y_train, X_test, y_test):
    """
    Compute the score of the stacked classification estimator and of each level_0 estimator.
    Parameters:
        model: estimator obtained after fitting
        X_train: train feature dataframe 
        X_test: test feature dataframe
        y_train: train target dataframe
        y_test: test target dataframe.
    Returns:
        plotting: accuracy of estimators on train and test sets
        res_stack: accuracy of estimators on train and test sets in a tabular.
    """       
    nb_estimators = len(model.estimators_)
    res_stack = np.empty((nb_estimators + 1, 3), dtype='object')
    m_t_x_train = model.transform(X_train)
    for j in range(nb_estimators):
        res_stack [j, 0] = [*model.named_estimators_.keys()][j]
        if m_t_x_train.shape[1] == nb_estimators: 
           res_stack [j, 1] = accuracy_score(np.rint(m_t_x_train).T[j], y_train)
           res_stack [j, 2] = accuracy_score(np.rint(model.transform(X_test)).T[j], y_test)
        else: 
           res_stack [j, 1] = accuracy_score(m_t_x_train.reshape((X_train.shape[0],\
                                                                  nb_estimators,\
                                                                  y_train.unique().shape[0])).argmax(axis=2).T[j],\
                                             y_train)
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
    """
    Compute the score of the stacked regression estimator and of each level_0 estimator.
    Parameters:
        model: estimator obtained after fitting
        X_train: train feature dataframe 
        X_test: test feature dataframe
        y_train: train target dataframe
        y_test: test target dataframe.
    Returns:
        plotting: accuracy of estimators on train and test sets
        res_stack: accuracy of estimators on train and test sets in a tabular.
    """        
    nb_estimators = len(model.estimators_)
    res_stack = np.empty((nb_estimators + 1, 3), dtype='object')
    m_t_x_train = model.transform(X_train)
    for j in range(nb_estimators):
        res_stack [j, 0] = [*model.named_estimators_.keys()][j]
        res_stack [j, 1] = r2_score(np.rint(m_t_x_train).T[j], y_train)
        res_stack [j, 2] = r2_score(np.rint(model.transform(X_test)).T[j], y_test)
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

def score_stacking(model, X_train, y_train, X_test, y_test):
    """
    Compute the score of the stacked estimator and of each level_0 estimator.
    Parameters:
        model: estimator obtained after fitting
        X_train: train feature dataframe 
        X_test: test feature dataframe
        y_train: train target dataframe
        y_test: test target dataframe.
    Returns:
        plotting: accuracy of estimators on train and test sets
        res_stack: accuracy of estimators on train and test sets in a tabular
        plotting: model importance according to performance
        mod_imp: model importance in a table.
    """     
    if is_classifier(model):
       res_stack = score_stacking_c(model, X_train, y_train, X_test, y_test)
    else:
       res_stack = score_stacking_r(model, X_train, y_train, X_test, y_test) 
    nb_estimators = len(model.estimators_)
    res_level_0 = res_stack[0:nb_estimators]
    mod_imp = np.delete(res_level_0[res_level_0[:, 2].argsort()], 1, axis=1)
    fig, ax = plt.subplots()
    ax.barh(mod_imp.T[0], mod_imp.T[1])
    ax.set_title("Model Importance according to performance")
    fig.tight_layout()
    plt.show()
    return res_stack, mod_imp

def find_coeff(model):
    """
    Searches the wrapped model for the feature importances parameter.
    """
    for attr in ("feature_importances_", "coef_"):
        try:
           return getattr(model, attr)
        except AttributeError:
           continue

        raise YellowbrickTypeError(
           "could not find feature importances param on {}".format(
                model.__class__.__name__
           )
        )
        
def model_importance_c(model, level_1_model):
    """
    Compute the model importance depending on final estimator coefficients for classification.
    Parameters:
        model: estimator obtained after fitting.
    Returns:
        mod_imp: sorted array of model importance. 
    """        
    level_0 = np.array(list(model.named_estimators_.keys()))
    n_classes = model.classes_.shape[0]
    n_models = len(model.estimators_)
    model_coeff = find_coeff(model.final_estimator_)
    
    if level_1_model == 'tree':
       if len(model_coeff) == n_models:
          coeff = model_coeff.reshape(n_models)  
       else:
          coeff = sum(model_coeff.reshape(n_classes,n_models))
            
    if level_1_model == 'regression':
       if len(model_coeff[0]) == n_models:
          coeff = model_coeff.reshape(n_models)  
       else:
          coeff = sum(model_coeff.reshape(n_classes,n_models,n_classes)[i].T[i] for i in range(n_classes))
            
    model_importance = np.empty((len(level_0), 2), dtype='object')
    for ind in range(len(level_0)):
        model_importance[ind, 0] = level_0[ind]
        model_importance[ind, 1] = coeff[ind]
    return model_importance[model_importance[:, 1].argsort()]

def model_importance_r(model, level_1_model):
    """
    Compute the model importance depending on final estimator coefficients for regression.
    Parameters:
        model: estimator obtained after fitting.
    Returns:
        mod_imp: sorted array of model importance.
    """         
    level_0 = np.array(list(model.named_estimators_.keys()))
    coeff = find_coeff(model.final_estimator_)
    model_importance = np.empty((len(level_0), 2), dtype='object')
    for ind in range(len(level_0)):
        model_importance[ind, 0] = level_0[ind]
        model_importance[ind, 1] = coeff[ind]
    return model_importance[model_importance[:, 1].argsort()]

def plot_model_importance(model, level_1_model):
    """
    Compute the model importance depending on final estimator coefficients.
    Parameters:
        model: estimator obtained after fitting.
    Returns:
        plotting: model importance according to aggregator coefficients
        mod_imp: sorted array of model importance.
    """      
    if is_classifier(model):
       mod_imp = model_importance_c(model, level_1_model)
    else:
       mod_imp = model_importance_r(model, level_1_model)
    fig, ax = plt.subplots()
    ax.barh(mod_imp.T[0], mod_imp.T[1])
    ax.set_title("Model Importance according to aggregator coefficients")
    fig.tight_layout()
    plt.show()
    return mod_imp

def plot_perm_importance(model, X, y, CPU):
    """
    Compute the feature permutation importance.
    Parameters:
        model: estimator obtained after fitting
        X: feature dataframe
        y: target dataframe
        CPU: boolean for CPU training.
    Returns:
        plotting: feature permutation importance
        perm_imp: sorted array of feature permutation importance.
    """       
    if is_classifier(model):
       scoring = 'accuracy'
    else:
       scoring = 'r2'  
    if CPU==True:
       result = permutation_importance(model, X, y, scoring=scoring, n_repeats=10, n_jobs=-1)
    else:
       result = permutation_importance(model, X, y, scoring=scoring, n_repeats=10)
    sorted_idx = result.importances_mean.argsort()
    perm_imp = np.array([X.columns[sorted_idx], result.importances[sorted_idx].mean(axis=1).T]).T
    fig, ax = plt.subplots()
    ax.barh(perm_imp.T[0], perm_imp.T[1])
    ax.set_title("Permutation Importance")
    fig.tight_layout()
    plt.show()
    return perm_imp

def plot_partial_dependence_c(model, X, features, features_cat, CPU, target_encoder):
    """
    Plot partial dependence of features for a given classification estimator and a given dataset.
    Parameters:
        model: estimator obtained after fitting
        X: feature dataframe
        features: list of features
        CPU: boolean for CPU training.
    Returns:
        plotting: partial dependence of input features.
    """      
    target = model.classes_
    for ind in range(len(target)):
        fig, ax = plt.subplots(figsize=(10, 5))
        
        if CPU==True:
           n_jobs = -1
        else:
           n_jobs = None
        
        if np.isin(features, features_cat):
            kind = "average"
        else:
            kind = "both"
        
        display = PartialDependenceDisplay.from_estimator(
                  estimator = model,
                  X = X,
                  features = features,
                  target = target[ind],
                  n_cols = 2,
                  categorical_features = features_cat,
                  kind = kind,
                  subsample=50,
                  n_jobs = n_jobs,
                  grid_resolution = 20,
                  ice_lines_kw = {"color": "tab:blue", "alpha": 0.2, "linewidth": 0.5},
                  pd_line_kw = {"color": "tab:orange", "linestyle": "--"},
                  ax = ax,
                  )

        display.figure_.suptitle("Partial dependence for the class: " + str(target_encoder.inverse_transform([ind])[0]))
        display.figure_.subplots_adjust(hspace=0.3)
        plt.show()
    
def plot_partial_dependence_r(model, X, features, features_cat, CPU):
    """
    Plot partial dependence of features for a given regression estimator and a given dataset.
    Parameters:
        model: estimator obtained after fitting
        X: feature dataframe
        features: list of features
        CPU: boolean for CPU training.
    Returns:
        plotting: partial dependence of input features.
    """      
    fig, ax = plt.subplots(figsize=(10, 5))
    if CPU==True:
       n_jobs = -1
    else:
       n_jobs = None
        
    if np.isin(features, features_cat):
       kind = "average"
    else:
       kind = "both"
        
    display = PartialDependenceDisplay.from_estimator(
              estimator = model,
              X = X,
              features = features,
              categorical_features = features_cat,
              n_cols = 2,
              kind=kind,
              subsample=50,
              n_jobs=n_jobs,
              grid_resolution=20,
              ice_lines_kw={"color": "tab:blue", "alpha": 0.2, "linewidth": 0.5},
              pd_line_kw={"color": "tab:orange", "linestyle": "--"},
              ax = ax,
              )

    display.figure_.suptitle("Partial dependence")
    display.figure_.subplots_adjust(hspace=0.3)
    plt.show() 

def plot_partial_dependence(model, X, features, CPU, target_encoder):
    """
    Plot partial dependence of features for a given estimator and a given dataset.
    Parameters:
        model: estimator obtained after fitting
        X: feature dataframe
        features: list of features, if features = [], partial dependences will be plot for all numeric features
        CPU: boolean for CPU training.
    Returns:
        plotting: partial dependence of input features.
    """
    # if input list of features is empty, we use the list of numeric features
    if features == []:
       features = X.columns.tolist() 
    else:
    #  we keep only numeric features    
       features = np.intersect1d(features, X.columns.tolist()).tolist() 
    
    features_cat, features_num = get_features()
    
    if features_cat.tolist() == []:
       features_cat = None 
        
    if is_classifier(model):
       plot_partial_dependence_c(model, X, features, features_cat, CPU, target_encoder)
    else:
       plot_partial_dependence_r(model, X, features, features_cat, CPU)
            
def pd_ice_plot(model, X, feature, CPU, target_encoder=None):
    """
    Interactively plot partial dependence of features for a given estimator and a given dataset.
    Parameters:
        model: estimator obtained after fitting
        X: feature dataframe
        features: list of features, if features = [], partial dependences will be plot for all numeric features
        CPU: boolean for CPU training.
    Returns:
        plotting: partial dependence of input features.
    """    
    
    def ppd(model, X, feature, CPU, target_encoder):
        plot_partial_dependence(model, X, feature, CPU, target_encoder) 
        
    interact(ppd, model=fixed(model), X=fixed(X), feature=feature, CPU=fixed(CPU), target_encoder=fixed(target_encoder));

def plot_history(history):
    """
    Plot learning curves of Keras neural network.
    Parameters:
        history: history of Keras neural network.
    Returns:
        plotting: learning curves of Keras neural network.
    """     
    pd.DataFrame(history.history).plot(figsize=(12, 9))
    plt.title("Learning Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.show()
    
def K_confusion_matrix(model, X_train, y_train, X_test, y_test):
    """
    Plot confusion matrix of a classification estimator on train and test sets.
    Parameters:
        model: estimator obtained after fitting
        X_train: train feature dataframe 
        X_test: test feature dataframe
        y_train: train target dataframe
        y_test: test target dataframe.
    Returns:
        plotting: confusion matrix on train and test sets.
    """     
    from sklearn.metrics import confusion_matrix
    y_pred = model.predict(X_train)
    if len(y_pred.shape)>1:
       y_pred = np.around(y_pred).astype(int)
       y_pred = np.argmax(y_pred, axis=1)
       y_train = y_train.idxmax(axis=1)
    cm = confusion_matrix(y_train, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title('Confusion matrix on train set')
    plt.show()
    y_pred = model.predict(X_test)
    if len(y_pred.shape)>1:
       y_pred = np.around(y_pred).astype(int)
       y_pred = np.argmax(y_pred, axis=1)
       y_test = y_test.idxmax(axis=1)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title('Confusion matrix on test set')
    plt.show()
    
def K_classification_report(model, X_train, y_train, X_test, y_test):
    """
    Plot classification report of a classification estimator on train and test sets.
    Parameters:
        model: estimator obtained after fitting
        X_train: train feature dataframe 
        X_test: test feature dataframe
        y_train: train target dataframe
        y_test: test target dataframe.
    Returns:
        plotting: classification report on train and test sets.
    """        
    y_pred = model.predict(X_train)
    if len(y_pred.shape)>1:
       y_pred = np.around(y_pred).astype(int)
       y_pred = np.argmax(y_pred, axis=1)
       y_train = y_train.idxmax(axis=1)
    cr=classification_report(y_train, y_pred, output_dict=True)
    display(pd.DataFrame(cr).transpose().style.set_caption("Classification report on train set"))
    y_pred = model.predict(X_test)
    if len(y_pred.shape)>1:
       y_pred = np.around(y_pred).astype(int)
       y_pred = np.argmax(y_pred, axis=1)
       y_test = y_test.idxmax(axis=1)
    cr=classification_report(y_test, y_pred, output_dict=True)
    display(pd.DataFrame(cr).transpose().style.set_caption("Classification report on test set"))
    
def K_r2(model, X_train, y_train, X_test, y_test):
    """
    Compute R^2 of a regression estimator on train and test sets.
    Parameters:
        model: estimator obtained after fitting
        X_train: train feature dataframe 
        X_test: test feature dataframe
        y_train: train target dataframe
        y_test: test target dataframe.
    Returns:
        array: scores on train and test sets.
    """         
    y_pred_train = model.predict(X_train)    
    y_pred_test = model.predict(X_test)
    dr2={'train': [r2_score(y_train, y_pred_train)],\
         'test': [r2_score(y_test, y_pred_test)]}
    display(pd.DataFrame(data=dr2).style.hide_index())
    
def K_mape(model, X_train, y_train, X_test, y_test):
    """
    Compute mean absolute percentage error of time series forcasting on train and test sets.
    Parameters:
        model: estimator obtained after fitting
        X_train: train feature dataframe 
        X_test: test feature dataframe
        y_train: train target dataframe
        y_test: test target dataframe.
    Returns:
        array: scores on train and test sets.
    """         
    y_pred_train = model.predict(X_train)    
    y_pred_test = model.predict(X_test)
    dmape={'train': [mean_absolute_percentage_error(y_train, y_pred_train)],\
           'test': [mean_absolute_percentage_error(y_test, y_pred_test)]}
    display(pd.DataFrame(data=dmape).style.hide_index())
     
# Fast API, Docker, Kubernetes functions
def fastapi_server(model, model_name, X, y, port, Docker=False, with_keras=False):
    """
    Generate the fastAPI server file, and save it in the current folder.
    Parameters:
        model: estimator obtained after fitting
        model_name : name of the saved model
        X: feature dataframe 
        y: target dataframe
        IP_address: IP address of the server 
        port: port of the server.  
    """   
    string = ""
    string = string  + "from fastapi import FastAPI\n"
    string = string  + "from joblib import load\n"
    string = string  + "import pandas as pd\n"
    string = string  + "import numpy as np\n"
    if Docker==False:
       string = string  + "import nest_asyncio\n"
       string = string  + "import uvicorn\n"
    string = string  + "import ast\n"
    string = string  + "import time\n"
    string = string  + "from sklearn.base import is_classifier\n"
    string = string  + "from pydantic import BaseModel\n"
    string = string  + "\n"
    string = string  + "# Creating FastAPI instance\n"
    string = string  + "app = FastAPI()\n"
    string = string  + "\n"
    string = string  + "# Creating class to define the request body\n"
    string = string  + "# and the type hints of each attribute\n"

    string = string  + "\n"
    string = string  + "class request_body(BaseModel):\n"
    for ind in range(X.dtypes.shape[0]):
        if str(X.dtypes[ind])[0:5]=='float':
           string = string + '      ' + X.columns[ind] + ': float\n'
        if str(X.dtypes[ind])[0:3]=='int':
           string = string + '      ' + X.columns[ind] + ': int\n'
        if str(X.dtypes[ind])[0:4]=='uint':
           string = string + '      ' + X.columns[ind] + ': int\n'
        if str(X.dtypes[ind])[0:6]=='object':
           string = string + '      ' + X.columns[ind] + ': str\n'
        if str(X.dtypes[ind])[0:4]=='bool':
           string = string + '      ' + X.columns[ind] + ': bool\n'

    string = string  + "\n"
    string = string  + "# read dataframe schema\n"
    string = string  + "schema = pd.read_csv('schema.csv')" 
    string = string  + "\n" 
    
    if with_keras:
       from modules.ezs_tech_func import keras_nn
       if is_classifier(model):
          string = string  + keras_nn('classification')
       else:
          string = string  + keras_nn('regression')
        
    string = string  + "\n"        
    string = string  + "model = load('" + model_name + "')\n"

    string = string  + "\n"
    if is_classifier(model):
       string = string  + "classes = " + str(y.unique().tolist()) + "\n"
    
    string = string  + "\n"
    string = string  + "@app.get('/ping')\n"
    string = string  + "def pong():\n"
    string = string  + "    return {'ping': 'pong!'}\n"
    
    string = string  + "\n"
    string = string  + "@app.post('/predict')\n"
    string = string  + "def predict(data : request_body):\n"
    string = string  + "\n"
    string = string  + "    elaps_start_time = time.time()\n"
    string = string  + "    cpu_start_time = time.process_time()\n"
    string = string  + "\n"
    string = string  + "    # Making the data in a form suitable for prediction\n"
    string = string  + "    test_data = [[\n"
    for ind in range(X.columns.shape[0]):
        string = string  + "              data." + X.columns[ind] + ',\n'
    string = string  + "    ]]\n"
    
    string = string  + "\n"
    
    string = string  + "    # Check input data\n"
    string = string  + "    data_err = []\n"
    string = string  + "    for ind in range(len(test_data[0])):\n"
    string = string  + "        if schema.iloc[ind][1] == 'num':\n"
    string = string  + "           interval = ast.literal_eval(schema.iloc[ind][2])\n"
    string = string  + "           if (test_data[0][ind] < interval[0]) | (test_data[0][ind] > interval[1]):\n"
    string = string  + "              data_err.append(schema.iloc[ind][0])\n"
    string = string  + "        if schema.iloc[ind][1] == 'cat':\n"
    string = string  + "           domain = ast.literal_eval(schema.iloc[ind][2])\n"
    string = string  + "           if not(np.isin(test_data[0][ind], domain)):\n"
    string = string  + "              data_err.append(schema.iloc[ind][0])\n"
    string = string  + "\n"

                
    if is_classifier(model):
       string = string  + "    # Predicting the Class\n"
       string = string  + "    result = model.predict(pd.DataFrame(test_data,\n"
       string = string  + "                                        columns=[\n"
       for ind in range(X.columns.shape[0]):
           string = string  + "                                                  '" + X.columns[ind] + "',\n"
       string = string  + "                          ]))[0].item()\n"
       string = string  + "\n"

       string = string  + "    elaps_end_time = time.time()\n"
       string = string  + "    cpu_end_time = time.process_time()\n"
       string = string  + "    elapsed_time = np.round((elaps_end_time - elaps_start_time) * 1000)\n"
       string = string  + "    elaps = str(elapsed_time) + 'ms'\n"
       string = string  + "    cpu_time = np.round((cpu_end_time - cpu_start_time) * 1000)\n"
       string = string  + "    cpu = str(cpu_time) + 'ms'\n"       
       string = string  + "\n"
       string = string  + "    # Return the Result\n"
       string = string  + "    return { 'class' : classes[result], 'error' : data_err, 'elapsed time' : elaps, 'cpu time' : cpu}\n"
    else: 
       string = string  + "    # Predicting the regression value\n"
       if with_keras: 
          string = string  + "    result = model.predict(pd.DataFrame(np.array([test_data[0],]*2),\n"
       else:
          string = string  + "    result = model.predict(pd.DataFrame(test_data,\n"        
       string = string  + "                                        columns=[\n"
       for ind in range(X.columns.shape[0]):
           string = string  + "                                                 '" + X.columns[ind] + "',\n"
       string = string  + "                          ]))[0].item()\n"
       string = string  + "\n"
       string = string  + "    elaps_end_time = time.time()\n"
       string = string  + "    cpu_end_time = time.process_time()\n"
       string = string  + "    elapsed_time = np.round((elaps_end_time - elaps_start_time) * 1000)\n"
       string = string  + "    elaps = str(elapsed_time) + 'ms'\n"
       string = string  + "    cpu_time = np.round((cpu_end_time - cpu_start_time) * 1000)\n"
       string = string  + "    cpu = str(cpu_time) + 'ms'\n"
       string = string  + "\n"
       string = string  + "    # Return the Result\n"
       string = string  + "    return { 'regression_value' : result, 'error' : data_err, 'elapsed time' : elaps, 'cpu time' : cpu}\n"
    
    string = string  + "\n"
    if Docker==False:
       string = string  + "nest_asyncio.apply()\n"
       string = string  + "uvicorn.run(app, port=" + str(port) +")\n"
       file_server = open("server.py", "w") 
    else:
       file_server = open("server_d.py", "w") 
    
    file_server.write(string)
    file_server.close()  

def dockerfile_generator(port):
    """
    Generate the Docker dockerfile, and save it in the current folder.
    Parameters:
        port: port of the server.   
    """   
    string = ""
    string = string  + "FROM python:3.10\n"
    string = string  + "\n"
    string = string  + "WORKDIR /app\n"
    string = string  + "\n"
    string = string  + "COPY ./requirements.txt /app/\n"
    string = string  + "RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt\n"
    string = string  + "\n"
    string = string  + "EXPOSE " + str(port)
    string = string  + "\n"
    string = string  + "COPY ./app /app/\n"
    string = string  + "\n"
    string = string  + 'CMD ["uvicorn", "server_d:app", "--host", "0.0.0.0", "--port", "' + str(port) + '"]\n'
    dockerfile = open("Dockerfile", "w")
    dockerfile.write(string)
    dockerfile.close()
    
def kube_yaml_generator(name, port):
    """
    Generate the Kubernetes yaml file, and save it in the current folder.
    Parameters:
        name: name of the server
        port: port of the server.   
    """   
    string = ""
    string = string  + "apiVersion: v1\n"
    string = string  + "kind: Service\n"
    string = string  + "metadata:\n"
    string = string  + "  name: " + name + "\n"
    string = string  + "spec:\n"
    string = string  + "  ports:\n"
    string = string  + "  - name: " + str(port) + "-tcp\n"
    string = string  + "    port: " + str(port) + "\n"
    string = string  + "    protocol: TCP\n"
    string = string  + "    targetPort: " + str(port) + "\n"
    string = string  + "  selector:\n"
    string = string  + "    com.docker.project: " + name + "\n"   
    string = string  + "  type: LoadBalancer\n"
    string = string  + "status:\n"
    string = string  + "  loadBalancer: {}\n"
    string = string  + "\n"
    string = string  + "---\n"
    string = string  + "apiVersion: apps/v1\n"
    string = string  + "kind: Deployment\n"
    string = string  + "metadata:\n"
    string = string  + "  labels:\n"
    string = string  + "    com.docker.project: " + name + "\n"
    string = string  + "  name: " + name + "\n"
    string = string  + "spec:\n"
    string = string  + "  replicas: 1\n"
    string = string  + "  selector:\n"
    string = string  + "    matchLabels:\n"
    string = string  + "      com.docker.project: " + name + "\n"
    string = string  + "  strategy:\n"
    string = string  + "    type: Recreate\n"
    string = string  + "  template:\n"
    string = string  + "    metadata:\n"
    string = string  + "      labels:\n"
    string = string  + "        com.docker.project: " + name + "\n"
    string = string  + "    spec:\n"
    string = string  + "      containers:\n"
    string = string  + "      - name: " + name + "\n"
    string = string  + "        image: " + name + "\n"
    string = string  + "        resources:\n"
    string = string  + "          limits:\n"
    string = string  + "            memory: 200Mi\n"
    string = string  + "          requests:\n"
    string = string  + "            cpu: 100m\n"
    string = string  + "            memory: 200Mi\n"
    string = string  + "        ports:\n"
    string = string  + "        - containerPort: " + str(port) + "\n"
    string = string  + "          protocol: TCP\n"
    string = string  + "        imagePullPolicy: IfNotPresent\n"
    string = string  + "      restartPolicy: Always\n"
    string = string  + "status: {}\n"
    kubernetes = open(name + "_deployment.yaml", "w")
    kubernetes.write(string)
    kubernetes.close()
    
def dockerize(name, model, model_name, X, y, port, with_keras):
    """
    Prepare a package for Docker delivery.
    Parameters:
        name: name of the server
        model: estimator obtained after fitting
        model_name: name of the saved model
        X: feature dataframe 
        y: target dataframe
        port: port of the server.   
    """   
    import os
    import shutil
    
    try:
        os.mkdir(name)
    except FileExistsError:
        shutil.rmtree(os.getcwd() + "/" + name)
        os.mkdir(name)
        
    shutil.copy('modules/requirements.txt', name)
    
    dockerfile_generator(port)
    shutil.move('Dockerfile', name)
    
    kube_yaml_generator(name, port)
    shutil.move(name + "_deployment.yaml", name)
    
    name = name + "/app"
    os.mkdir(name)
    shutil.copy("model.sav", name)
    shutil.copy("schema.csv", name)
    
    fastapi_server(model, model_name, X, y, port, Docker=True, with_keras=with_keras)
    shutil.move('server_d.py', name)
    
    name = name + "/modules"
    os.mkdir(name)
    shutil.copy('modules/ezs_model.py', name)    

def store_data(name, path, threshold_corr, threshold_model, threshold_feature, threshold_score, test_size, level_1_model, score_stack_0, score_stack_1, score_stack_2, 
           model_imp_0, model_imp_1, model_imp_2, 
           feature_importance_0, feature_importance_1, feature_importance_2):
    import sqlite3
    conn = sqlite3.connect(os.getcwd() + '/modules/ezs_store.db')
    cursor = conn.cursor()

    search_problem = cursor.execute("SELECT name FROM problem WHERE name = ?", (name,))
    problem_name = search_problem.fetchone()
    if problem_name == None:
       cursor.execute("INSERT INTO problem (name, path , type, target) VALUES(?, ?, ?, ?)", (name, path, problem_type, target_col))

    search_version = cursor.execute("SELECT MAX(version) FROM solution WHERE name = ?", (name,))
    row = search_version.fetchone()
    if row == (None,):
       version = 1
    else:
       version = row[0] + 1

    cursor.execute("INSERT INTO solution (name, version, correlation, nb_model, nb_feature, score, test_size) VALUES(?, ?, ?, ?, ?, ?, ?)", \
                    (name, version, threshold_corr, threshold_model, threshold_feature, threshold_score, test_size));

    schema = pd.read_csv('schema.csv')
    for ind in range(len(user_drop_cols)):
        cursor.execute("INSERT INTO eda (name, version, feature, type, range, drop_user, drop_correlation, target)  VALUES(?, ?, ?, ?, ?, ?, ?, ?)", \
                        (name, version, user_drop_cols[ind], None, None, 1, 0, 0));
    for ind in range(schema.shape[0]):
        if schema['column_name'][ind] in correlated_features:
           drop_correlation = True
        else:
           drop_correlation = False

        cursor.execute("INSERT INTO eda (name, version, feature, type, range, drop_user, drop_correlation, target)  VALUES(?, ?, ?, ?, ?, ?, ?, ?)", \
                        (name, version, schema['column_name'][ind], schema['column_type'][ind], schema['column_range'][ind], 0, drop_correlation, 0));

    cursor.execute("INSERT INTO eda (name, version, feature, type, range, drop_user, drop_correlation, target)  VALUES(?, ?, ?, ?, ?, ?, ?, ?)", \
                        (name, version, target_col, None, None, 0, 0, 1));

    for ind in range(3):
        cursor.execute("INSERT INTO model (name, version, step, L1_model) VALUES (?, ?, ?, ?)", \
                        (name, version, ind+1, level_1_model));

        score_stack = locals()["_".join(['score_stack', str(ind)])]
        for ind2 in range(score_stack.shape[0]):
            cursor.execute("INSERT INTO model_score (name, version, step, model, train_score, test_score) VALUES(?, ?, ?, ?, ?, ?)", \
                            (name, version, ind+1, score_stack[ind2,0], score_stack[ind2,1], score_stack[ind2,2]));

        model_imp = locals()["_".join(['model_imp', str(ind)])]
        for ind2 in range(model_imp.shape[0]):
            cursor.execute("INSERT INTO model_importance (name, version, step, model, importance) VALUES(?, ?, ?, ?, ?)", \
                            (name, version, ind+1, model_imp[ind2,0], model_imp[ind2,1]));

        feature_importance = locals()["_".join(['feature_importance', str(ind)])]
        for ind2 in range(feature_importance.shape[0]):
            cursor.execute("INSERT INTO feature_importance (name, version, step, feature, importance) VALUES(?, ?, ?, ?, ?)", \
                            (name, version, ind+1, feature_importance[ind2,0], feature_importance[ind2,1]));

    # cursor.execute("DELETE FROM problem WHERE name = ?", (name,))

    conn.commit()
    conn.close()

# Functions used in time series analysis
def plot_correlation(df, target_col, t=1):
    """
    Compute and plot the correlation and the hierarchical clustering of the features of the input time series dataframe
    Parameters:
        df: a dataframe,
        t: distance threshold.
    Returns:
        Plotting of the correlation and the hierarchical clustering.
    """
    if df.shape[1] > 1:
       print('Correlation matrix')
       corr = df.corr()
       display(corr.style.background_gradient(cmap='coolwarm'))
       print('Hierarchical clustering')
       selected_features_names = hierarchical_clustering(df.drop(target_col, axis=1), t=t)
       print('selected_features_names = ', selected_features_names)
    else:
       print('No correlation for univariate time series') 
    
def plot_acf_pacf(df, column):
    """
    Compute and plot the autocorrelation and partial autocorrelation functions of a selected feature of the input time series dataframe.
    For more information: 
    - https://www.statsmodels.org/devel/generated/statsmodels.graphics.tsaplots.plot_acf.html
    - https://www.statsmodels.org/devel/generated/statsmodels.graphics.tsaplots.plot_pacf.html.
    Parameters:
        df: a dataframe
        column: a column of the dataframe interactively selected.
    Returns:
        Plotting of the autocorrelation and partial autocorrelation functions.
    """
    def p_a_p(df, column):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        plot_acf(df[column], ax=ax1)
        plot_pacf(df[column], ax=ax2)
        fig.suptitle("Autocorrelation information of " + str(column))
        plt.show()
        
    interact(p_a_p, df=fixed(df), column=column)

def plot_seasonal_decompose(df, column, model, period):
    """
    Interactively plot the seasonal decomposition of a selected feature of the input time series dataframe.
    For more information: https://www.statsmodels.org/devel/generated/statsmodels.tsa.seasonal.seasonal_decompose.html.
    Parameters:
        df: a dataframe
        column: a column of the dataframe
        model: additive/multiplicative
        period: period of the series.
    Returns:
        Plotting of the seasonal decomposition.
    """    
    def p_s_d(df, column, model, period):
        result = seasonal_decompose(df[column], model=model, period=period)
        fig = result.plot()
        fig.set_size_inches((10, 10))
        fig.tight_layout()
        plt.show()
       
    interact(p_s_d, df=fixed(df), column=column, model=model, period=period, continuous_update=False)
    
def plot_seasonal_decompose_2(df, column, period1, period2):
    """
    Interactively plot the seasonal decomposition of a selected feature of the input time series dataframe taking into account 2 periods.
    For more information: https://www.statsmodels.org/devel/generated/statsmodels.tsa.seasonal.MSTL.html#statsmodels.tsa.seasonal.MSTL.
    Parameters:
        df: a dataframe
        column: a column of the dataframe
        period1: 1st period of the series
        period2: 2nd period of the series.
    Returns:
        Plotting of the seasonal decomposition.
    """
    def p_s_d_2(df, column, period1, period2):
        result = MSTL(df[column], periods=(period1, period1*period2)).fit()
        fig = result.plot()
        fig.set_size_inches((10, 10))
        fig.tight_layout()
        plt.show()
       
    interact_manual(p_s_d_2, df=fixed(df), column=column, period1=period1, period2=period2)

# Constants used in the fuction plot_unobserved_components
local_linear_trend_model = {
    'level': 'local linear trend', 'trend': True, 'damped_cycle': True, 'stochastic_cycle': True,
    'stochastic_seasonal': True, 'cycle': True
}

smooth_trend_model = {
    'level': 'smooth trend', 'cycle': True, 'damped_cycle': True, 'stochastic_cycle': True,
    'stochastic_seasonal': True, 'cycle': True, 'trend': True
}

random_trend_model = {
    'level': 'random trend', 'cycle': True, 'damped_cycle': True, 'stochastic_cycle': True,
    'stochastic_seasonal': True, 'cycle': True, 'trend': True
}

local_level_with_deterministic_trend_model = {
    'level': 'local linear deterministic trend', 'cycle': True, 'damped_cycle': True, 'stochastic_cycle': True,
    'stochastic_seasonal': True, 'cycle': True
}

random_walk_with_drift_model = {
    'level': 'random walk with drift', 'cycle': True, 'damped_cycle': True, 'stochastic_cycle': True,
    'stochastic_seasonal': True, 'cycle': True
}

model_uc = [('local linear trend', local_linear_trend_model), ('smooth trend', smooth_trend_model), 
         ('random trend', random_trend_model), ('local linear deterministic trend', local_level_with_deterministic_trend_model), 
         ('random walk with drift', random_walk_with_drift_model)
        ]

method = [('modified Powell’s method', 'powell'), ('Nelder-Mead', 'nm'), ('Broyden-Fletcher-Goldfarb-Shanno', 'bfgs'), 
          ('limited-memory BFGS with optional box constraints','lbfgs'),  ('Newton-Raphson','newton'), 
          ('conjugate gradient', 'cg'), ('Newton-conjugate gradient', 'ncg'), ('basin-hopping solver', 'basinhopping')]

def plot_unobserved_components(df, column, model, method, confidence):
    """
    Interactively plot the univariate unobserved components of a selected feature of the input time series dataframe.
    For more information: https://www.statsmodels.org/devel/generated/statsmodels.tsa.statespace.structural.UnobservedComponents.html#statsmodels.tsa.statespace.structural.UnobservedComponents.
    Parameters:
        df: a dataframe
        column: a column of the dataframe
        model: model used to compute the unobserved components
        method: method used to compute the unobserved components
        confidence: confidence intervals for the components.
    Returns:
        Plotting of the univariate unobserved components.
    """    
    def p_u_c(df, column, model, method, confidence):
        qwargs = model
        output_mod = UnobservedComponents(df[column], **qwargs)
        output_res = output_mod.fit(method=method, disp=False)
        output_res.plot_components(legend_loc='upper left', fig=plt.tight_layout(), figsize=(10, 16), alpha=1-confidence)
        plt.show();
        print(output_res.summary())
       
    interact_manual(p_u_c, df=fixed(df), column=column, model=model, method=method, confidence=confidence)

def ts_dataframe_to_supervised(df, target, n_in=1, n_out=0, dropT=True):
    """
    Transform a time series dataframe into a supervised learning dataset.
    Parameters:
        df: a dataframe.
        target: this is the target variable you intend to use in supervised learning
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropT: Boolean - whether or not to drop columns at time "t".
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    namevars = df.columns.tolist()
    # input sequence (t-n, ... t-1)
    drops = []
    for i in range(n_in, -1, -1):
        if i == 0:
            for var in namevars:
                addname = var+'_t'
                df.rename(columns={var:addname},inplace=True)
                drops.append(addname)
        else:
            for var in namevars:
                addname = var+'_t_'+str(i)
                df[addname] = df[var].shift(i)
    # forecast sequence (t, t+1, ... t+n)
    if n_out == 0:
        n_out = False
    for i in range(1, n_out):
        for var in namevars:
            addname = var+'_t_'+str(i)
            df[addname] = df[var].shift(-i)
    # drop rows with NaN values
    df.dropna(inplace=True,axis=0)
    # put it all together
    target = target+'_t'
    if dropT:
        drops.remove(target)
        df.drop(drops, axis=1, inplace=True)
    preds = [x for x in list(df) if x not in [target]] 
    return df, target, preds

def timeseries_train_test_split(X, y, test_size):
    """
    Perform train-test split with respect to time series structure.
    Parameters:
        X: feature dataframe
        y: target dataframe
        test_size: proportion reserved for the test file.
    Returns:
        X_train: train set (features)
        X_test: test set (features)
        y_train: train set (target)
        y_test: test set (target).
    """
    test_index = int(len(X) * (1 - test_size))
    X_train = X[:test_index]
    X_test = X[test_index:]
    y_train = y[:test_index]
    y_test = y[test_index:]
    return X_train, X_test, y_train, y_test

def mean_absolute_percentage_error(y_true, y_pred):
    """
    Compute the mean absolute percentage error .
    Parameters:
        y_true: correct target values
        y_pred: predicted target values. 
    Returns:
        the mean absolute percentage error.
    """
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def plot_ts_results(X_train, y_train, X_test, y_test, model, confidence, plot_intervals, plot_anomalies):
    """
    Interactively plot:
        - the modelled vs original values
        - the prediction intervals according to a given confidence interval
        - the anomalies (points that resides outside the confidence interval)
    Parameters:
        X_train: train feature dataframe
        y_train: train target dataframe 
        X_test: test feature dataframe
        y_test: test target dataframe 
        model: model used for predictions
        confidence: confidence intervals
        plot_intervals: Boolean for displaying confidence intervals
        plot_anomalies: Boolean for displaying anomalies. 
    Returns:
        Plottings.
    """    
    def p_m_s(X_train, y_train, X_test, y_test, model, confidence, plot_intervals, plot_anomalies):
        
        prediction = model.predict(X_test)

        plt.figure(figsize=(15, 7))

        x = X_test.index.date
        # x = range(prediction.size)
        plt.plot(x, prediction, label='prediction', linewidth=2.0)
        plt.plot(x, y_test, label='actual', linewidth=2.0)
        if plot_intervals:
            timeseries_cv = TimeSeriesSplit(n_splits=5)
            cv = cross_val_score(model, X_train, y_train, 
                                 cv=timeseries_cv, scoring='neg_mean_absolute_error')
            mae = -1 * cv.mean()
            deviation = cv.std()

            # confidence interval computation
            scale = stats.norm.ppf(confidence)
            margin_error = mae + scale * deviation
            lower = prediction - margin_error
            upper = prediction + margin_error

            fill_alpha = 0.2
            fill_color = '#66C2D7'
            plt.fill_between(x, lower, upper, color=fill_color, alpha=fill_alpha, label= str(confidence*100) + '% CI')      

            if plot_anomalies:
                anomalies = np.array([np.nan] * len(y_test))
                anomalies[y_test < lower] = y_test[y_test < lower]
                anomalies[y_test > upper] = y_test[y_test > upper]
                plt.plot(anomalies, 'o', markersize=10, label='Anomalies')

        error = mean_absolute_percentage_error(prediction, y_test)
        plt.title('Mean absolute percentage error: {0:.2f}%'.format(error))
        plt.legend(loc='best')
        plt.tight_layout()
        plt.grid(True)
        
    interact_manual(p_m_s, X_train=fixed(X_train), y_train=fixed(y_train), X_test=fixed(X_test), y_test=fixed(y_test), model=fixed(model), confidence=confidence, plot_intervals=plot_intervals, plot_anomalies=plot_anomalies)
    


