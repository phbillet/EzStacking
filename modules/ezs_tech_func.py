import io
import os
import random
import functools
import pandas as pd
import numpy as np
import nbformat as nbf
import ipywidgets as widgets
import shutil
from IPython.display import display
from ipyfilechooser import FileChooser
from zipfile import ZipFile

def generate(project_name, problem_type, time_dep, lag_number, date_idx, stacking, data_size, with_gauss, with_hgboost, with_keras, with_CPU,\
             with_gb, with_pipeline, yb, with_adaboost, with_bagging, with_decision_tree, with_random_forest, with_sgd, with_mlp, with_nn, with_svm,\
             seaborn, ydata_profiling, fast_eda, file, target_col, threshold_NaN, threshold_cat, threshold_Z, test_size, threshold_entropy,\
             undersampling, undersampler, level_1_model, no_decorrelator, no_optimization, random_state,\
             threshold_corr, threshold_model, threshold_score, threshold_feature, output, deployment_FastAPI_port, deployment_Docker_port):
    """
    Initialize the notebook, analyze input data from GUI, generate, write and execute the notebook.
    """
    user_drop_cols=[]
    features_of_interest = []
    nb = analyze(project_name, problem_type, time_dep, lag_number, date_idx, stacking, data_size, with_gauss, with_hgboost, with_keras, with_CPU,\
                 with_gb, with_pipeline, yb, with_adaboost, with_bagging, with_decision_tree, with_random_forest, with_sgd, with_mlp, with_nn, with_svm,\
                 seaborn, ydata_profiling, fast_eda, file, target_col, user_drop_cols, features_of_interest, threshold_NaN, threshold_cat, threshold_Z,\
                 test_size, threshold_entropy, undersampling, undersampler, level_1_model, no_decorrelator, no_optimization, random_state,\
                 threshold_corr, threshold_model, threshold_score, threshold_feature, deployment_FastAPI_port, deployment_Docker_port)
    fname = output + '.ipynb'
    with open(fname, 'w') as f:
         nbf.write(nb, f)
                

def set_config(with_gauss, with_hgboost, with_keras, with_CPU, with_gb, with_pipeline, problem_type, time_dep, lag_number, date_idx,
               stacking, yb, with_adaboost, with_bagging, with_decision_tree, with_random_forest, with_sgd, with_mlp, with_nn, with_svm,
               seaborn, ydata_profiling, fast_eda, data_size, level_1_model, no_decorrelator, no_optimization):
    """
    Set configuration: load configuration database, generate the different dataframes used to generate
    cells of the notebook according to the data from the GUI.
    """

    xls = pd.ExcelFile('./modules/ezstacking_config.ods', engine="odf")
    meta_package = pd.read_excel(xls, 'meta_package')
    package_source = pd.read_excel(xls, 'package_source')
    package = pd.read_excel(xls, 'package')
    document = pd.read_excel(xls, 'document')
    
    meta_package.loc[meta_package['meta_package_index'] == 'STACK', ['meta_package_valid']] = stacking
    meta_package.loc[meta_package['meta_package_index'] == 'KER', ['meta_package_valid']] = with_keras
    meta_package.loc[meta_package['meta_package_index'] == 'CPU', ['meta_package_valid']] = with_CPU
    meta_package.loc[meta_package['meta_package_index'] == 'HGB', ['meta_package_valid']] = with_hgboost
    meta_package.loc[meta_package['meta_package_index'] == 'GP', ['meta_package_valid']] = with_gauss
    meta_package.loc[meta_package['meta_package_index'] == 'GNB', ['meta_package_valid']] = with_gauss
    meta_package.loc[meta_package['meta_package_index'] == 'GB', ['meta_package_valid']] = with_gb
    meta_package.loc[meta_package['meta_package_index'] == 'ADA', ['meta_package_valid']] = with_adaboost
    meta_package.loc[meta_package['meta_package_index'] == 'BAG', ['meta_package_valid']] = with_bagging

    if with_bagging:
       meta_package.loc[meta_package['meta_package_index'] == 'MLP', ['meta_package_valid']] = True
       meta_package.loc[meta_package['meta_package_index'] == 'SKSV', ['meta_package_valid']] = True
    else:
       meta_package.loc[meta_package['meta_package_index'] == 'MLP', ['meta_package_valid']] = with_mlp
       meta_package.loc[meta_package['meta_package_index'] == 'SKSV', ['meta_package_valid']] = with_svm
        
    if level_1_model == 'tree':
        meta_package.loc[meta_package['meta_package_index'] == 'DT', ['meta_package_valid']] = True
    else:
        meta_package.loc[meta_package['meta_package_index'] == 'DT', ['meta_package_valid']] = with_decision_tree
    meta_package.loc[meta_package['meta_package_index'] == 'RF', ['meta_package_valid']] = with_random_forest
    meta_package.loc[meta_package['meta_package_index'] == 'SGD', ['meta_package_valid']] = with_sgd
    meta_package.loc[meta_package['meta_package_index'] == 'KN', ['meta_package_valid']] = with_nn
    meta_package.loc[meta_package['meta_package_index'] == 'PIP', ['meta_package_valid']] = with_pipeline
    meta_package.loc[meta_package['meta_package_index'] == 'YB', ['meta_package_valid']] = yb
    meta_package.loc[meta_package['meta_package_index'] == 'SNS', ['meta_package_valid']] = seaborn
    meta_package.loc[meta_package['meta_package_index'] == 'YDP', ['meta_package_valid']] = ydata_profiling
    meta_package.loc[meta_package['meta_package_index'] == 'FEDA', ['meta_package_valid']] = fast_eda
                     
    problem = problem_type
    size = data_size
    
    package_source_type = 'full'

    pd_pk_import = package_source[(package_source.package_source_type   == package_source_type)] \
                   .merge(meta_package[(meta_package.meta_package_valid == True) & \
                                         ((meta_package.meta_package_data_size == 'both') | \
                                         (meta_package.meta_package_data_size == data_size))], \
                                         left_on  = 'meta_package_index', \
                                         right_on = 'meta_package_index', \
                                         how = 'inner') \
                                 [['package_source_index', 'package_source_name', \
                                   'package_source_short_name', 'package_source_code']]

    package_source_type = 'partial'

    pd_pk_srce_from = package_source[(package_source.package_source_type == package_source_type)\
                                    ].merge(meta_package[(meta_package.meta_package_valid == True) & \
                                             ((meta_package.meta_package_data_size == 'both') | \
                                             (meta_package.meta_package_data_size == data_size))], \
                                            left_on  = 'meta_package_index', \
                                            right_on = 'meta_package_index', \
                                            how = 'inner') \
                                    [['package_source_index', 'package_source_name', \
                                      'package_source_short_name', 'package_source_code']] 
    
 
    
    pd_pk_from = package[((package.package_problem == 'None') | (package.package_problem == problem)) \
                        ].merge(pd_pk_srce_from, \
                                left_on='package_source_index', \
                                right_on='package_source_index',\
                                how='inner') \
                         .merge(meta_package[(meta_package.meta_package_valid == True) & \
                                             ((meta_package.meta_package_data_size == 'both') | \
                                             (meta_package.meta_package_data_size == data_size))], \
                                left_on  = 'meta_package_index', \
                                right_on = 'meta_package_index', \
                                how = 'inner') \
                    [['package_source_index', 'package_source_name', 'package_name']].drop_duplicates()
    

    pd_level_0 = package[(package.package_type == 'estimator') & \
                         (package.package_problem == problem) & \
                         (package.meta_package_index != 'STACK') \
                        ].merge(pd_pk_from, \
                                left_on='package_source_index', \
                                right_on='package_source_index',\
                                how='inner') \
                         .merge(meta_package[(meta_package.meta_package_valid == True) & \
                                             ((meta_package.meta_package_data_size == 'both') | \
                                             (meta_package.meta_package_data_size == data_size))], \
                                left_on  = 'meta_package_index', \
                                right_on = 'meta_package_index', \
                                how = 'inner') \
                        [[ 'package_index', 'package_code', 'meta_package_tree']].drop_duplicates()
    
    pd_tree = package[(package.package_type == 'estimator') & \
                      (package.package_problem == problem) & \
                      (package.meta_package_index != 'STACK') \
                     ].merge(pd_pk_from, \
                             left_on='package_source_index', \
                             right_on='package_source_index',\
                             how='inner') \
                      .merge(meta_package[(meta_package.meta_package_tree == True) & \
                                          (meta_package.meta_package_valid == True) & \
                                          ((meta_package.meta_package_data_size == 'both') | \
                                          (meta_package.meta_package_data_size == data_size))], \
                             left_on  = 'meta_package_index', \
                             right_on = 'meta_package_index', \
                             how = 'inner') \
                      [[ 'package_index', 'package_code']].drop_duplicates()
    
    pd_document = document[((document.document_problem == 'both') | (document.document_problem == problem)) & \
                           ((document.document_ts == 'both') | (document.document_ts == time_dep)) & \
                           ((document.document_data_size == 'both') | (document.document_data_size == data_size)) & \
                           ((document.document_level_1_model == 'both') | (document.document_level_1_model == level_1_model)) & \
                           ((document.document_no_decorrelator == 'both') | (document.document_no_decorrelator == no_decorrelator)) & \
                           ((document.document_no_optimization == 'both') | (document.document_no_optimization == no_optimization)) & \
                           ((document.document_stacking == 'both') | \
                           (document.document_stacking == \
                            meta_package[meta_package.meta_package_index == 'STACK']\
                            ['meta_package_valid'].tolist()[0]\
                           )) & \
                           ((document.document_keras == 'both') | \
                           (document.document_keras == \
                            meta_package[meta_package.meta_package_index == 'KER']\
                            ['meta_package_valid'].tolist()[0]\
                           )) & \
                           ((document.document_pipeline == 'both') | \
                           (document.document_pipeline == \
                            meta_package[meta_package.meta_package_index == 'PIP']\
                            ['meta_package_valid'].tolist()[0]\
                           )) & \
                           ((document.document_yb == 'both') | \
                           (document.document_yb == \
                            meta_package[meta_package.meta_package_index == 'YB']\
                            ['meta_package_valid'].tolist()[0]\
                           )) & \
                           ((document.document_ydp == 'both') | \
                           (document.document_ydp == \
                            meta_package[meta_package.meta_package_index == 'YDP']\
                            ['meta_package_valid'].tolist()[0]\
                           )) & \
                           ((document.document_feda == 'both') | \
                           (document.document_feda == \
                            meta_package[meta_package.meta_package_index == 'FEDA']\
                            ['meta_package_valid'].tolist()[0]\
                           )) \
                          ].merge(meta_package[meta_package.meta_package_valid == True], \
                                               left_on  = 'meta_package_index', \
                                               right_on = 'meta_package_index', \
                                               how = 'inner') \
                          [['sort_key', 'title', 'text', 'code']].sort_values('sort_key') \
                          [['title', 'text', 'code']].drop_duplicates()
    
    return pd_pk_import, pd_pk_from, pd_level_0, pd_document, pd_tree

def load_package(nb, pd_pk_import, pd_pk_from):
    """
    Generate the cells used to load packages in the first part of the notebook.
    """
    text = "## Package loading"
    nb['cells'].append(nbf.v4.new_markdown_cell(text))
    
    string = '' 
    for index, row in pd_pk_import.iterrows(): 
        if row[2] == '*':
           string = string + "from " + str(row[1]) + " import " +  str(row[2]) + "\n"
        elif row[2] != 'None': 
           string = string + "import " + str(row[1]) + " as " + str(row[2]) + "\n" 
        elif row[1] != 'None':  
           string = string + "import " + str(row[1]) + "\n"
        else:
           pass 
        if row[3] != 'None':
           string = string + str(row[3]) + "\n"
        
    for index, row in pd_pk_from.iterrows():
        string = string + "from " + str(row[1]) + " import " +  str(row[2]) + "\n"
        
    code = string
    nb['cells'].append(nbf.v4.new_code_cell(code))

    return nb

def analyze(project_name, problem_type, time_dep, lag_number, date_idx, stacking, data_size, with_gauss, with_hgboost, with_keras, with_CPU, with_gb,\
            with_pipeline, yb, with_adaboost, with_bagging, with_decision_tree, with_random_forest, with_sgd, with_mlp, with_nn, with_svm,\
            seaborn, ydata_profiling, fast_eda, file, target_col, user_drop_cols, features_of_interest,\
            threshold_NaN, threshold_cat, threshold_Z, test_size, threshold_entropy, undersampling, undersampler, level_1_model,\
            no_decorrelator, no_optimization, random_state, threshold_corr, threshold_model, threshold_score, threshold_feature, deployment_FastAPI_port, deployment_Docker_port):

    """
    Analyze input data from GUI, set configuration, generate the different cells of the notebook
    """
    pd_pk_import, pd_pk_from, pd_level_0, pd_document, pd_tree = set_config(with_gauss, with_hgboost, with_keras, with_CPU, with_gb,\
                                                                            with_pipeline,problem_type, time_dep, lag_number, date_idx, stacking, yb,\
                                                                            with_adaboost, with_bagging, with_decision_tree, with_random_forest, with_sgd, with_mlp, with_nn, with_svm,\
                                                                            seaborn, ydata_profiling, fast_eda, data_size, level_1_model, no_decorrelator, no_optimization)
    
    fileList = ['./modules/client.ipynb', './modules/server.ipynb']
    for item in fileList:
        shutil.copy(item, ".")
    
    
    nb = nbf.v4.new_notebook()
    kernelspec = dict(
            display_name='ezstacking',
            name='ezstacking',
            language='python')
    nb.metadata['kernelspec'] = kernelspec
    
    nb['cells'] = []
    
    for index, row in pd_document.iterrows():
#        print('index:', index)
#        print('row[0]:', row[0])
#        print('row[1]:', row[1])
#        print('row[2]:', row[2])
#        print('("run" in row[2]):', ('run' in row[2]))
        
        if index == 1:
           # at index 1: prepare package preamble 
           nb = load_package(nb, pd_pk_import, pd_pk_from) 
        if row[1] != 'None':
           # title management
           if row[0] == ' ': 
              text = str(row[1])            
           else:
              text = str(row[0]) + ' ' + str(row[1])
           nb['cells'].append(nbf.v4.new_markdown_cell(text))
        if row[2] != 'None':
           # code management
           code = eval(row[2])
#           print(code)
           nb['cells'].append(nbf.v4.new_code_cell(code))

    return nb

def keras_nn(problem_type):
    """
    Generate a basic skeleton of Keras neural network for classification
    """
    code_class = "from typing import Dict, Iterable, Any\n" + \
                 "from scikeras.wrappers import KerasClassifier\n" + \
                 " \n" + \
                 "class K_MLPClassifier(KerasClassifier):\n" + \
                 " \n" + \
                 "      def __init__(\n" + \
                 "          self,\n" + \
                 "          hidden_layer_sizes=100,\n" + \
                 "          activation='relu',\n" + \
                 "          batch_normalization=True,\n" + \
                 "          dropout=0.0,\n" + \
                 "          optimizer='adam',\n" + \
                 "          optimizer__learning_rate=0.001,\n" + \
                 "          epochs=200,\n" + \
                 "          verbose=0,\n" + \
                 "          **kwargs,\n" + \
                 "      ):\n" + \
                 "          super().__init__(**kwargs)\n" + \
                 "          self.hidden_layer_sizes = hidden_layer_sizes\n" + \
                 "          self.activation = activation\n" + \
                 "          self.batch_normalization = batch_normalization\n" + \
                 "          self.dropout = dropout\n" + \
                 "          self.optimizer = optimizer\n" + \
                 "          self.epochs = epochs\n" + \
                 "          self.verbose = verbose\n" + \
                 " \n" + \
                 "      def _keras_build_fn(self, compile_kwargs: Dict[str, Any]):\n" + \
                 "          model = keras.Sequential()\n" + \
                 "          inp = keras.layers.Input(shape=(self.n_features_in_))\n" + \
                 "          model.add(inp)\n" + \
                 "          for hidden_layer_size in (self.hidden_layer_sizes,):\n" + \
                 "              layer = keras.layers.Dense(hidden_layer_size, activation=self.activation)\n" + \
                 "              model.add(layer)\n" + \
                 "              if self.batch_normalization:\n" + \
                 "                 layer = keras.layers.BatchNormalization()\n" + \
                 "                 model.add(layer)\n" + \
                 "              if self.dropout > 0.0:\n" + \
                 "                 layer = keras.layers.Dropout(self.dropout)\n" + \
                 "                 model.add(layer)\n" + \
                 "          if self.target_type_ == 'binary':\n" + \
                 "              n_output_units = 1\n" + \
                 "              output_activation = 'sigmoid'\n" + \
                 "              loss = 'binary_crossentropy'\n" + \
                 "          elif self.target_type_ == 'multiclass':\n" + \
                 "              n_output_units = self.n_classes_\n" + \
                 "              output_activation = 'softmax'\n" + \
                 "              loss = 'sparse_categorical_crossentropy'\n" + \
                 "          else:\n" + \
                 "              raise NotImplementedError(f'Unsupported task type: {self.target_type_}')\n" + \
                 "          out = keras.layers.Dense(n_output_units, activation=output_activation)\n" + \
                 "          model.add(out)\n" + \
                 "          model.compile(loss=loss, optimizer=compile_kwargs['optimizer'])\n" + \
                 "          return model\n" 

    """
    Generate a basic skeleton of Keras neural network for regression
    """
    code_regre = "from typing import Dict, Iterable, Any\n" + \
                 "from scikeras.wrappers import KerasRegressor\n" + \
                 " \n" + \
                 "class K_MLPRegressor(KerasRegressor):\n" + \
                 " \n" + \
                 "      def __init__(\n" + \
                 "          self,\n" + \
                 "          hidden_layer_sizes=100,\n" + \
                 "          activation='relu',\n" + \
                 "          batch_normalization=True,\n" + \
                 "          dropout=0,\n" + \
                 "          optimizer='adam',\n" + \
                 "          optimizer__learning_rate=0.001,\n" + \
                 "          epochs=200,\n" + \
                 "          verbose=0,\n" + \
                 "          **kwargs,\n" + \
                 "      ):\n" + \
                 "          super().__init__(**kwargs)\n" + \
                 "          self.hidden_layer_sizes = hidden_layer_sizes\n" + \
                 "          self.activation = activation\n" + \
                 "          self.batch_normalization = batch_normalization\n" + \
                 "          self.dropout = dropout\n" + \
                 "          self.optimizer = optimizer\n" + \
                 "          self.epochs = epochs\n" + \
                 "          self.verbose = verbose\n" + \
                 " \n" + \
                 "      def _keras_build_fn(self, compile_kwargs: Dict[str, Any]):\n" + \
                 "          model = keras.Sequential()\n" + \
                 "          inp = keras.layers.Input(shape=(self.n_features_in_))\n" + \
                 "          model.add(inp)\n" + \
                 "          for hidden_layer_size in (self.hidden_layer_sizes,):\n" + \
                 "              layer = keras.layers.Dense(hidden_layer_size, activation=self.activation)\n" + \
                 "              model.add(layer)\n" + \
                 "              if self.batch_normalization:\n" + \
                 "                 layer = keras.layers.BatchNormalization()\n" + \
                 "                 model.add(layer)\n" + \
                 "              if self.dropout > 0:\n" + \
                 "                 layer = keras.layers.Dropout(self.dropout)\n" + \
                 "                 model.add(layer)\n" + \
                 "          out = keras.layers.Dense(1)\n" + \
                 "          model.add(out)\n" + \
                 "          model.compile(loss='mse', optimizer=compile_kwargs['optimizer'])\n" + \
                 "          return model\n" 
                 
    if problem_type == "classification":
       code = code_class 
    else:
       code = code_regre 
        
    return code
    
def level_0(pd_level_0):
    """
    Generate the 0 level model list 
    """
    string = "level_0 = [ \n"
    for index, row in pd_level_0.iterrows():
        string = string +  "          ('" + str(row[0]) + "', "  + str(row[1]) +  "), \n"  
    string = string + "          ]"
    return string


def pipe_level_0(pd_level_0):
    """
    Generate the 0 level model list with pipeline
    """
    string = "level_0 = [ \n"
    for index, row in pd_level_0.iterrows():
        if row[2] == True:
           string = string +  "          ('" + str(row[0]) + "', make_pipeline(tree_preprocessor, " \
                    + str(row[1]) +  ")), \n"  
        else:
           string = string +  "          ('" + str(row[0]) + "', make_pipeline(ntree_preprocessor, " \
                    + str(row[1]) +  ")), \n"  
    string = string + "          ]"  
    return string

def list_model(pd_level_0):
    string = ""
    for index, row in pd_level_0.iterrows():
        if (row[1] != 'K_C') & (row[1] != 'K_R'): 
           string = string + "# " + str(row[1]) + "\n"  
       
    return string    

# GUI
project_name = widgets.Text(
                value='project name',
                placeholder='Type something',
                description_tooltip='Project name',
                description='Project name:',
                disabled=False   
                )

# Tab:  File & Problem
caption_fc = widgets.Label(value='Input file:')
fc = FileChooser('./')
file = widgets.VBox([caption_fc, fc])

caption_target = widgets.Label(value='Target name:')

# interaction between FileChooser and target dropdown
target_cl = widgets.Dropdown(
                    options=['Empty',],
                    value='Empty',
                    description='Target:',
                    disabled=False,
                    )

def list_columns():
    global fc, target_cl
    if fc.selected == None:
       target_cl = widgets.Dropdown(
                    options=['Empty',],
                    value='Empty',
                    description='Target:',
                    disabled=False,
                    )
    else:
       target_cl.options=pd.read_csv(fc.selected).columns.tolist()
                 

fc.register_callback(list_columns)

target = widgets.VBox([caption_target, target_cl]) 

file_target = widgets.VBox([file, target]) 

caption_problem_option = widgets.Label(value='Problem and data characteristics:')

problem_type = widgets.RadioButtons(
                options=['classification', 'regression'],
                description='Problem type:',
                description_tooltip='If the target is categorical then choose classification, regression otherwise.',
                disabled=False
                )

time_dep = widgets.Checkbox(
                value=False,
                description='Time dependence',
                description_tooltip='The regression problem is time dependent.',
                disabled=False,
                display='none',
                indent=False
                )
time_dep.layout.display = 'none'

date_idx = widgets.IntText(
                value=0,
                description='Date index:',
                description_tooltip='Index of the date column',
                disabled=False,
                layout=widgets.Layout(width='150px', height='50px'))

date_idx.layout.display = 'none'

lag_number = widgets.IntText(
                value=3,
                description='Lag number:',
                description_tooltip='Number of lag observations as input (X)',
                disabled=False,
                layout=widgets.Layout(width='150px', height='50px'))

lag_number.layout.display = 'none'

# interaction between problem_type radio button and time dependence checkbox
def remove_td(problem_type):
    if problem_type['new'] == 'regression':
       time_dep.layout.display = 'flex'
    else:
       time_dep.layout.display, time_dep.value = 'none', False
      
problem_type.observe(remove_td, names='value')

# interaction between time dependence checkbox and date index 
def remove_date_idx(time_dep):
    if time_dep['new'] == True:
       date_idx.layout.display, date_idx.value = 'flex', 0
    else:
       date_idx.layout.display, date_idx.value = 'none', 0
      
time_dep.observe(remove_date_idx, names='value')

# interaction between time dependence checkbox and lag number 
def remove_lag(time_dep):
    if time_dep['new'] == True:
       lag_number.layout.display, lag_number.value = 'flex', 3
    else:
       lag_number.layout.display, lag_number.value = 'none', 0
      
time_dep.observe(remove_lag, names='value')

data_size = widgets.RadioButtons(
                options=['small', 'large'],
                description='Data size:',
                description_tooltip='Choose large, if the file contains more 5000 rows',
                disabled=False
                )

random_state = widgets.IntText(
                value=42,
                description='Random seed:',
                description_tooltip='Fix this number for reproductibility',
                disabled=False,
                layout=widgets.Layout(width='150px', height='50px')
                )

problem_option1 = widgets.HBox([problem_type, data_size])
problem_option1_1 = widgets.VBox([problem_option1, time_dep, date_idx, lag_number, random_state])
problem_option = widgets.VBox([caption_problem_option, problem_option1_1])

file_problem_tab = widgets.VBox([project_name, widgets.HBox([file_target, problem_option])])


# Tab:  EDA
visualizer_caption_option = widgets.Label(value='Optional visualizers:')
yb = widgets.Checkbox(
                value=False,
                description='Yellow Brick',
                description_tooltip='Visulization will use Yellow Brick',
                disabled=False,
                indent=False
                )

seaborn = widgets.Checkbox(
                value=False,
                description='Seaborn',
                tooltip='Visulization will use Seaborn',                
                disabled=False,
                indent=False
                )

ydata_profiling = widgets.Checkbox(
                value=False,
                description='Ydata profiling',
                tooltip='Visulization will use Ydata profiling',                
                disabled=False,
                indent=False
                )

fast_eda = widgets.Checkbox(
                value=False,
                description='Fast EDA',
                tooltip='Visulization will use Fast EDA',                
                disabled=False,
                indent=False
                )

EDA_1 = widgets.VBox([yb, seaborn])
EDA_2 = widgets.VBox([fast_eda, ydata_profiling])
EDA_3 = widgets.HBox([EDA_1, EDA_2])

EDA_viz_option = widgets.VBox([visualizer_caption_option, EDA_3])

caption_threshold_EDA = widgets.Label(value='EDA thresholds:')

threshold_NaN = widgets.FloatSlider(
                value=0.5,
                min=0,
                max=1,
                step=0.03,
                description='Th. NaN:',
                description_tooltip='If the proportion of NaN is greater than this number the column will be dropped',
                disabled=False,
                continuous_update=False,
                orientation='horizontal',
                readout=True,
                readout_format='.1f',
                )

threshold_cat = widgets.IntText(
                value=5,
                description='Th. Cat:',
                description_tooltip='If the number of different values in a column is less than this number,\n the column will be considered as a categorical column',
                disabled=False
                )

threshold_Z = widgets.FloatSlider(
                value=3.0,
                min=0,
                max=10,
                step=0.1,
                description='Th. Z:',
                description_tooltip='If the Z_score is greater than this number, the row will be dropped',
                disabled=False,
                continuous_update=False,
                orientation='horizontal',
                readout=True,
                readout_format='.1f',
                )

threshold_EDA = widgets.VBox([caption_threshold_EDA, threshold_cat, threshold_NaN, threshold_Z])

EDA_tab = widgets.HBox([EDA_viz_option, threshold_EDA])

# Tab:  splitting
caption_threshold_split = widgets.Label(value='Splitting thresholds:')

test_size = widgets.FloatSlider(
                 value=0.33,
                 min=0.00,
                 max=1.00,
                 step=0.01,
                 description='Test size:',
                 description_tooltip='Proportion of the dataset to include in the test split',
                 disabled=False,
                 continuous_update=False,
                 orientation='horizontal',
                 readout=True,
                 readout_format='.2f',
                 )

threshold_entropy = widgets.FloatSlider(
                value=0.75,
                min=0.00,
                max=1.00,
                step=0.01,
                description='Th. E:',
                description_tooltip='If target entropy is greater than this number, RepeatedStratifiedKFold will be used.',
                disabled=False,
                continuous_update=False,
                orientation='horizontal',
                readout=True,
                readout_format='.2f',
                ) 

undersampling = widgets.Checkbox(
                value=False,
                description='Undersampling',
                description_tooltip='Check this option, if the imbalanced target classes are badly managed without.',
                disabled=False,
                indent=False
                )

undersampler = widgets.RadioButtons(
    options=['Random', 'Centroids', 'AllKNN', 'TomekLinks'],
#     value='pineapple',
    description='Undersampler:',
    disabled=False
)
undersampler.layout.display = 'none'

# interaction between undersampling and undersampler
def remove_undersampler(undersampling):
    if undersampling['new']:
       undersampler.layout.display = 'flex'
    else:
       undersampler.layout.display = 'none'
    
undersampling.observe(remove_undersampler, names='value')

splitter = widgets.VBox([caption_threshold_split, test_size, threshold_entropy])
undersample = widgets.VBox([undersampling, undersampler])
split_tab = widgets.HBox([undersample, splitter])

# interaction between problem_type and sampling and entropy
def remove_undersample(problem_type):
    if problem_type['new'] == 'classification':
       undersample.layout.display, threshold_entropy.layout.display = 'flex', 'flex'
    else:
       undersampling.value, undersample.layout.display, threshold_entropy.layout.display = False, 'none', 'none'

problem_type.observe(remove_undersample, names='value')

# Tab:  model
level_0 = widgets.Label(value='Level 0:')
model_caption_option = widgets.Label(value='Optional models:')
stacking = widgets.Checkbox(
#                value=False,
                value=True,
                description='Stacking',
                description_tooltip='The model will use stacked generalization',
                disabled=False,
                indent=False
                )

gauss = widgets.Checkbox(
                value=False,
                description='Gaussian Methods',
                description_tooltip='The model will use Gaussian process and Gaussian naive Bayes',
                disabled=False,
                indent=False
                )

# interaction between data_size radio button and gauss checkbox
def remove_gauss(data_size):
    if data_size['new'] == 'small':
       gauss.layout.display = 'flex'
       decision_tree.layout.display = 'flex'
    else:
       gauss.layout.display, gauss.value = 'none', False
       decision_tree.layout.display, decision_tree.value = 'none', False
      

data_size.observe(remove_gauss, names='value')


hgboost = widgets.Checkbox(
                value=False,
                description='Histogram-based Gradient Boosting',
                description_tooltip='The model will use Histogram-based Gradient Boosting',
                disabled=False,
                indent=False
                )

keras = widgets.Checkbox(
                value=False,
                description='Keras',
                description_tooltip='The model will use Keras neural network',
                disabled=False,
                indent=False,
                layout=widgets.Layout(width='75px', height='50px')
                )

CPU = widgets.Checkbox(
                value=False,
                description='Train on CPU',
                description_tooltip='Keras neural network will be trained on CPU',
                disabled=False,
                indent=False,
                layout=widgets.Layout(width='100px', height='50px')
                )

# interaction between keras and CPU checkboxes
if keras.value == False:
   CPU.layout.display = 'none'

def remove_CPU(keras):
    if keras['new']:
       CPU.layout.display = 'flex'
    else:
       CPU.layout.display, CPU.value = 'none', False 

keras.observe(remove_CPU, names='value')

gboost = widgets.Checkbox(
                value=False,
                description='Gradient Boosting',
                description_tooltip='The model will use extrem gradient boosting', 
                disabled=False,
                indent=False
                )

decision_tree = widgets.Checkbox(
                value=False,
                description='Decision Tree',
                description_tooltip='The model will use decision tree',
                disabled=False,
                indent=False
                )

random_forest = widgets.Checkbox(
                value=False,
                description='Random Forest',
                description_tooltip='The model will use random forest',
                disabled=False,
                indent=False
                )

adaboost = widgets.Checkbox(
                value=False,
                description='AdaBoost',
                description_tooltip='The model will use AdaBoost',
                disabled=False,
                indent=False
                )

bagging = widgets.Checkbox(
                value=False,
                description='Bagging',
                description_tooltip='The model will use bagging',
                disabled=False,
                indent=False
                )

sgd = widgets.Checkbox(
                value=False,
                description='Stochastic Gradient Descent',
                description_tooltip='The model will use stochastic gradient descent',
                disabled=False,
                indent=False
                )

mlp = widgets.Checkbox(
                value=False,
                description='Multi-layer Perceptron',
                description_tooltip='The model will use multi-layer perceptron',
                disabled=False,
                indent=False
                )

nn = widgets.Checkbox(
                value=False,
                description='Nearest Neighbors',
                description_tooltip='The model will use nearest neighbors',
                disabled=False,
                indent=False
                )

svm = widgets.Checkbox(
                value=False,
                description='Support Vector Machines',
                description_tooltip='The model will use support vector machines',
                disabled=False,
                indent=False
                )

pipeline = widgets.Checkbox(
#                value=False,
                value=True,
                description='Pipeline',
                description_tooltip='The model will contain a preprocessing step',
                disabled=False,
                indent=False
                )

model_keras = widgets.HBox([keras, CPU])
#model_option2 = widgets.HBox([pipeline, gboost])
model_option_grp = widgets.VBox([hgboost, gboost, adaboost, bagging, decision_tree, random_forest, sgd, nn, svm, gauss, mlp, model_keras])

model_option_0 = widgets.VBox([level_0, model_caption_option, model_option_grp])

level_1 = widgets.Label(value='Level 1:')
level_1_model = widgets.RadioButtons(
                options=['regression', 'tree'],
                value='regression',
                description='Level 1 model type:',
                disabled=False
                )

decorrelator = widgets.Label(value='Decorrelation in model:')
no_decorrelator = widgets.Checkbox(
                value=False,
                description='No correlation',
                description_tooltip='The model will not include decorrelation',
                disabled=False,
                indent=False
                )

optimization = widgets.Label(value='Model optimization:')
no_optimization = widgets.Checkbox(
                value=False,
                description='No optimization',
                description_tooltip='The model will not be optimized',
                disabled=False,
                indent=False
                )

model_option_1 = widgets.VBox([level_1, level_1_model, decorrelator, no_decorrelator, optimization, no_optimization])

caption_threshold_mod = widgets.Label(value='Modelling thresholds:')

threshold_corr = widgets.FloatSlider(
                 value=1.00,
                 min=0.00,
                 max=1.00,
                 step=0.01,
                 description='Th. Corr:',
                 description_tooltip='If the correlation is greater than this number the column will be dropped',
                 disabled=False,
                 continuous_update=False,
                 orientation='horizontal',
                 readout=True,
                 readout_format='.2f',
                 )
threshold_model = widgets.IntText(
                value=5,
                description='Th. Model:',
                description_tooltip='Keep this number of most important models',
                disabled=False
                )

threshold_score = widgets.FloatSlider(
                value=0.7,
                min=0,
                max=1,
                step=0.1,
                description='Th. score:',
                description_tooltip='Keep models having test score greater than this number',
                disabled=False,
                continuous_update=False,
                orientation='horizontal',
                readout=True,
                readout_format='.2f',
                )

threshold_feature = widgets.IntText(
                value=5,
                description='Th. Feature:',
                description_tooltip='Keep this number of most important features',
                disabled=False
                )

threshold_modelling = widgets.VBox([caption_threshold_mod, threshold_corr, threshold_score, threshold_model, threshold_feature])

model_tab = widgets.HBox([model_option_0, model_option_1, threshold_modelling])

# Tab:  build
caption_output_file = widgets.Label(value='Output file name:')
output = widgets.Text(
                value='output file',
                placeholder='Type something',
                description_tooltip='Output filename',
                description='Output:',
                disabled=False   
                )


build_tab_content = output

run = widgets.Button(
                description='Build',
                tooltip='If no error, you should find the generated notebook in the current folder',
                disabled=False,
                button_style='success', # 'success', 'info', 'warning', 'danger' or ''
                icon=''
                )

deployment = widgets.Label(value='FastAPI and Docker:')

deployment_FastAPI_port = widgets.Text(
                value='8000',
                placeholder='Input a port',
                description_tooltip='Input a port',
                description='FastAPI port:',
                disabled=False   
                ) 

deployment_Docker_port = widgets.Text(
                value='80',
                placeholder='Input a port',
                description_tooltip='Input a port',
                description='Docker port:',
                disabled=False   
                ) 

deployment_fields = widgets.VBox([deployment, deployment_FastAPI_port, deployment_Docker_port])

def on_run_clicked(b, project_name, problem_type, time_dep, lag_number, date_idx, stacking, data_size, gauss, hgboost, keras, CPU, gboost, pipeline, fc, yb,\
                      adaboost, bagging, decision_tree, random_forest, sgd, mlp, nn, svm,\
                      seaborn, ydata_profiling, fast_eda, target, threshold_NaN, threshold_cat, threshold_Z, test_size, threshold_entropy,\
                      undersampling, undersampler, level_1_model, no_decorrelator, no_optimization, random_state,\
                      threshold_corr, threshold_model, threshold_score, threshold_feature, output, deployment_FastAPI_port, deployment_Docker_port):
    generate(project_name.value, problem_type.value, time_dep.value, lag_number.value, date_idx.value, stacking.value, data_size.value, gauss.value, hgboost.value,\
             keras.value, CPU.value, gboost.value, pipeline.value, yb.value,\
             adaboost.value, bagging.value, decision_tree.value, random_forest.value, sgd.value, mlp.value, nn.value, svm.value,\
             seaborn.value, ydata_profiling.value, fast_eda.value, fc.selected,\
             target_cl.value, threshold_NaN.value, threshold_cat.value, threshold_Z.value,\
             test_size.value, threshold_entropy.value, undersampling.value, undersampler.value, level_1_model.value, no_decorrelator.value, no_optimization.value,\
             random_state.value, threshold_corr.value, threshold_model.value,\
             threshold_score.value, threshold_feature.value, output.value, deployment_FastAPI_port.value, deployment_Docker_port.value)

run.on_click(functools.partial(on_run_clicked, project_name=project_name, problem_type=problem_type, time_dep=time_dep, lag_number=lag_number, date_idx=date_idx, stacking=stacking,\
                               data_size=data_size, gauss=gauss, hgboost=hgboost, keras=keras, CPU=CPU, gboost=gboost,\
                               pipeline=pipeline, yb=yb,\
                               adaboost=adaboost, bagging=bagging, decision_tree=decision_tree, random_forest=random_forest, sgd=sgd, mlp=mlp, nn=nn, svm=svm,\
                               seaborn=seaborn, ydata_profiling=ydata_profiling, fast_eda=fast_eda, fc=fc, target=target,\
                               threshold_NaN=threshold_NaN, threshold_cat=threshold_cat, threshold_Z=threshold_Z,\
                               test_size = test_size, threshold_entropy=threshold_entropy, \
                               undersampling = undersampling, undersampler = undersampler, level_1_model=level_1_model, no_decorrelator=no_decorrelator, no_optimization=no_optimization, random_state=random_state,\
                               threshold_corr = threshold_corr, threshold_model = threshold_model, threshold_score = threshold_score,\
                               threshold_feature = threshold_feature, output=output, deployment_FastAPI_port=deployment_FastAPI_port, deployment_Docker_port=deployment_Docker_port))



build_tab = widgets.VBox([caption_output_file, build_tab_content, deployment_fields, run])

# tab : test
def test_endpoint(schema, test_type, port, Docker=False):
    """
    Generate a test (i.e. all values belong to the domains define in schema.csv
    """
    newline = '\n'
    comma_newline = ',\n'
    continuation = " \\"
    string = ""
    string = string + "curl -X 'POST'"  + continuation + newline
    if Docker:
       string = string + "   'http://0.0.0.0:" + str(port) + "/predict'"  + continuation + newline        
    else:
       string = string + "   'http://127.0.0.1:" + str(port) + "/predict'"  + continuation + newline
    string = string + "   -H 'accept: application/json'"  + continuation + newline
    string = string + "   -H 'Content-Type: application/json'"  + continuation + newline
    string = string + "   -d '{\n"
    if test_type == 'passing':
       for ind in range(schema.shape[0]):
           string = string + "   " + "\"" + \
                    schema['column_name'][ind] + "\"" + ": "
           if schema['column_type'][ind] == 'num':
              string = string + str(random.uniform(eval(schema['column_range'][ind])[0], eval(schema['column_range'][ind])[1])) 
           if schema['column_type'][ind] == 'cat':
              string = string + '"' + str(random.choice(eval(schema['column_range'][ind]))) + '"' 
           if ind == (schema.shape[0] - 1):
              string = string + newline
           else:
              string = string + comma_newline
    else:
       rand_num = random.randint(0, 1) 
       for ind in range(schema.shape[0]):
           string = string + "   " + "\"" + \
                    schema['column_name'][ind] + "\"" + ": "
           if schema['column_type'][ind] == 'num':
              if rand_num == 1: 
                 string = string + str(eval(schema['column_range'][ind])[1] + random.randint(1,10)) 
              else:
                 string = string + str(random.uniform(eval(schema['column_range'][ind])[0], eval(schema['column_range'][ind])[1])) 
           if schema['column_type'][ind] == 'cat':
              if rand_num == 1: 
                 word_list = ["tata", "tete", "titi", "toto", "tutu", "tyty"] 
                 string = string + '"' + str(random.choice(word_list)) + '"'  
              else:
                 string = string + '"' + str(random.choice(eval(schema['column_range'][ind]))) + '"'
           if ind == (schema.shape[0] - 1):
              string = string + newline
           else:
              string = string + comma_newline
           rand_num_new = random.randint(0, 1) 
           if rand_num != rand_num_new: 
              rand_num = rand_num_new
           else:
              rand_num = random.randint(0, 1) 
    string = string + "}'" + newline 
    return string

def test_generator(nb_ok, nb_ko, port, Docker=False):
    """
    Generate file test.sh depending on number of passing and non-passing tests.
    """
    string = ""
    if not os.path.isfile('schema.csv'):
       print('It seems that the model is not built.')
    else:
       schema = pd.read_csv("schema.csv")
       for ind in range(nb_ok):
           string = string + str('echo -e "Test OK: ' + str(ind)) + '"\n'
           string = string + test_endpoint(schema, 'passing', port, Docker)
           string = string + str('echo -e ""\n')
       for ind in range(nb_ko):
           string = string + str('echo -e "Test KO: ' + str(ind)) + '"\n'
           string = string + test_endpoint(schema, 'notpassing', port, Docker)
           string = string + str('echo -e ""\n')
       if Docker:     
          file_server = open("test_d.sh", "w")
       else:     
          file_server = open("test.sh", "w")
       file_server.write(string)
       file_server.close()  

caption_test = widgets.Label(value='Test generator:')

nb_ok = widgets.IntText(
                value=3,
                description='# passing:',
                description_tooltip='Number of passing tests',
                disabled=False
                )    
    
nb_ko = widgets.IntText(
                value=3,
                description='# non-passing:',
                description_tooltip='Number of non-passing tests',
                disabled=False
                )  
test_nb = widgets.HBox([nb_ok, nb_ko])
    
test = widgets.Button(
                description='Generate tests',
                tooltip='Generate tests according to number of passing and non-passing tests',
                disabled=False,
                button_style='success', # 'success', 'info', 'warning', 'danger' or ''
                icon=''
                )

def test_gen(nb_ok, nb_ko, project_name, deployment_FastAPI_port, deployment_Docker_port):
    test_generator(nb_ok.value, nb_ko.value, deployment_FastAPI_port.value, Docker=False)
    test_generator(nb_ok.value, nb_ko.value, deployment_Docker_port.value, Docker=True)

    source = os.getcwd()
    destination = source + "/" + str(project_name.value)
    
    try:
        os.remove(destination + "/" + "test_d.sh")
    except FileNotFoundError:
        pass
    
    try:
        shutil.move(source + "/" + "test_d.sh", destination)
    except FileNotFoundError:
        pass
    

def on_test_clicked(b, nb_ok, nb_ko, project_name,  deployment_FastAPI_port, deployment_Docker_port):
    test_gen(nb_ok, nb_ko, project_name, deployment_FastAPI_port, deployment_Docker_port)

test.on_click(functools.partial(on_test_clicked, nb_ok=nb_ok, nb_ko=nb_ko, project_name=project_name, \
              deployment_FastAPI_port=deployment_FastAPI_port, deployment_Docker_port=deployment_Docker_port))

test_tab = widgets.VBox([caption_test, test_nb, test])


# tab : zip and clean
def zip_files(fc, output):
    # create a ZipFile object
    zipname = output + '.zip'
    zipObj = ZipFile(zipname, 'w')
    # Add multiple files to the zip
    nbname = output + '.ipynb'
    # add dataset (without full path)
    try:
       zipObj.write(fc, fc.split(os.sep)[-1])
    except AttributeError:
       pass
    file_list = [nbname, 'server.ipynb', 'client.ipynb', 'modules/ezs_func.py', 'model.sav', 'schema.csv', 'test.sh', 'server.py']
    for file in file_list:
        try:
           zipObj.write(file)
        except FileNotFoundError:
           pass
    # close the Zip File
    zipObj.close()
    
def delete_files(output, project_name):
    # delete work files
    nbname = output + '.ipynb'
    file_list = [nbname, 'model.sav', 'schema.csv', 'server.py', 'test.sh', 'client.ipynb', 'server.ipynb']  
    for file in file_list:
        try:
           os.remove(file)
        except FileNotFoundError:
           pass
    try:
       shutil.rmtree(os.getcwd() + "/" + project_name)
    except FileNotFoundError:
       pass
        
def zip_and_clean(fc, output, project_name):
    zip_files(fc, output)
    delete_files(output, project_name)
    
zip = widgets.Button(
                description='Zip & Clean',
                tooltip='Zip your files and clean the folder',
                disabled=False,
                button_style='success', # 'success', 'info', 'warning', 'danger' or ''
                icon=''
                ) 

def on_zip_clicked(b, fc, output, project_name):
    zip_and_clean(fc.selected, output.value, project_name.value)

zip.on_click(functools.partial(on_zip_clicked, fc=fc, output=output, project_name=project_name))

dev_names = ['EDA', 'Split', 'Model', 'Build']
dev_tabs = [EDA_tab, split_tab, model_tab, build_tab]
dev_tab_layout = widgets.Layout(display='flex')
#                    flex_flow='column', 
#                    align_items='stretch', 
#                    border='solid',
#                    height='250px')
dev_gui = widgets.Tab(dev_tabs, layout=dev_tab_layout)
[dev_gui.set_title(i, title) for i, title in enumerate(dev_names)]

gui_names = ['File & Problem', 'Development', 'Test', 'Zip & Clean']
gui_tabs = [file_problem_tab, dev_gui, test_tab, zip] 
gui_tab_layout = widgets.Layout(display='flex')
#                    flex_flow='column', 
#                    align_items='stretch', 
#                    border='solid',
#                    height='300px')
ezs_gui = widgets.Tab(gui_tabs, layout=gui_tab_layout)
[ezs_gui.set_title(i, title) for i, title in enumerate(gui_names)]
