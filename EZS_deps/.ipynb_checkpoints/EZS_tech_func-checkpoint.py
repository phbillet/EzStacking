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

def generate(problem_type, stacking, data_size, cv, with_gauss, with_hgboost, with_keras, with_CPU,\
             with_xgb, with_pipeline, yb, seaborn, file, target_col, threshold_NaN,\
             threshold_cat, threshold_Z, test_size, threshold_entropy,\
             undersampling, undersampler, level_1_model,\
             threshold_corr, threshold_model, threshold_score, threshold_feature, output):
    """
    Initialize the notebook, analyze input data from GUI, generate, write and execute the notebook.
    """
    user_drop_cols=[]
    features_of_interest = []
    nb = analyze(problem_type, stacking, data_size, cv, with_gauss, with_hgboost, with_keras, with_CPU,\
                 with_xgb, with_pipeline, yb, seaborn, file, target_col, user_drop_cols,\
                 features_of_interest, threshold_NaN, threshold_cat, threshold_Z,\
                 test_size, threshold_entropy, undersampling, undersampler, level_1_model,\
                 threshold_corr, threshold_model,\
                 threshold_score, threshold_feature)
    fname = output + '.ipynb'
    with open(fname, 'w') as f:
         nbf.write(nb, f)
    

def set_config(with_gauss, with_hgboost, with_keras, with_CPU, with_xgb, with_pipeline, problem_type, stacking, yb, seaborn, data_size, cv, level_1_model):
    """
    Set configuration: load configuration database, generate the different dataframes used to generate
    cells of the notebook according to the data from the GUI.
    """

    xls = pd.ExcelFile('./EZS_deps/EZStacking_config.ods', engine="odf")
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
    meta_package.loc[meta_package['meta_package_index'] == 'XGB', ['meta_package_valid']] = with_xgb
    meta_package.loc[meta_package['meta_package_index'] == 'PIP', ['meta_package_valid']] = with_pipeline
    meta_package.loc[meta_package['meta_package_index'] == 'YB', ['meta_package_valid']] = yb
    meta_package.loc[meta_package['meta_package_index'] == 'SNS', ['meta_package_valid']] = seaborn
                     
    problem = problem_type
    size = data_size
    
    package_source_type = 'full'

    pd_pk_import = package_source[(package_source.package_source_type   == package_source_type) \
                                 ].merge(meta_package[(meta_package.meta_package_valid == True) & \
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
                           ((document.document_data_size == 'both') | (document.document_data_size == data_size)) & \
                           ((document.document_cv == 'both') | (document.document_cv == cv)) & \
                           ((document.document_level_1_model == 'both') | (document.document_level_1_model == level_1_model)) & \
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
                           ((document.document_xgb == 'both') | \
                           (document.document_xgb == \
                            meta_package[meta_package.meta_package_index == 'XGB']\
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

def analyze(problem_type, stacking, data_size, cv, with_gauss, with_hgboost, with_keras, with_CPU, with_xgb,\
            with_pipeline, yb, seaborn, file, target_col, user_drop_cols, features_of_interest,\
            threshold_NaN, threshold_cat, threshold_Z, test_size,\
            threshold_entropy, undersampling, undersampler, level_1_model,\
            threshold_corr, threshold_model, threshold_score, threshold_feature):

    """
    Analyze input data from GUI, set configuration, generate the different cells of the notebook
    """
    pd_pk_import, pd_pk_from, pd_level_0, pd_document, pd_tree = set_config(with_gauss, with_hgboost, with_keras, with_CPU, with_xgb,\
                                                                            with_pipeline,problem_type, stacking, yb,\
                                                                            seaborn, data_size, cv, level_1_model)
    
    fileList = ['./EZS_deps/client.ipynb', './EZS_deps/server.ipynb']
    for item in fileList:
        shutil.copy(item, ".")
    
    
    nb = nbf.v4.new_notebook()
    kernelspec = dict(
            display_name='EZStacking',
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
    code_class = "def K_Class(): \n" + \
                 "    keras.backend.clear_session() \n" + \
                 "#   neural network architecture: start \n" + \
                 "    model = Sequential() \n" + \
                 "    model.add(BatchNormalization()) \n" + \
                 "    model.add(Dense(layer_size, activation='selu')) \n" + \
                 "#    model.add(LayerNormalization()) \n" + \
                 "    model.add(BatchNormalization()) \n" + \
                 "    model.add(Dropout(0.2)) \n" + \
                 "    model.add(Dense(nb_targets, activation='softmax')) \n" + \
                 "#   neural network architecture: end   \n" + \
                 "    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])\n" + \
                 "    return model\n"
    """
    Generate a basic skeleton of Keras neural network for regression
    """
    code_regre = "def K_Regre(): \n" + \
                 "    keras.backend.clear_session()\n" + \
                 "#   neural network architecture: start  \n" + \
                 "    model = Sequential() \n" + \
                 "    model.add(BatchNormalization()) \n" + \
                 "    model.add(Dense(layer_size, activation='relu')) \n" + \
                 "    model.add(BatchNormalization()) \n" + \
                 "#    model.add(LayerNormalization()) \n" + \
                 "    model.add(Dropout(0.2)) \n" + \
                 "    model.add(Dense(1)) \n" + \
                 "#   neural network architecture: end   \n" + \
                 "    model.compile(loss='mean_squared_error', optimizer='adam') \n" + \
                 "    return model\n"
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
# Tab:  File & Problem
caption_fc = widgets.Label(value='Input file:')
fc = FileChooser('./')
file = widgets.VBox([caption_fc, fc])

caption_target = widgets.Label(value='Target name:')
# interaction between data_size radio button and gauss checkbox
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

data_size = widgets.RadioButtons(
                options=['small', 'large'],
                description='Data size:',
                description_tooltip='Choose large, if the file contains more 5000 rows',
                disabled=False
                )

problem_option1 = widgets.HBox([problem_type, data_size])
problem_option = widgets.VBox([caption_problem_option, problem_option1])

file_problem_tab = widgets.HBox([file_target, problem_option])


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
EDA_viz_option = widgets.VBox([visualizer_caption_option, yb, seaborn])

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

splitter = widgets.VBox([caption_threshold_split, test_size, threshold_entropy])
undersample = widgets.VBox([undersampling, undersampler])
split_tab = widgets.HBox([splitter, undersample])

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
                description='Gauss',
                description_tooltip='The model will use Gaussian process and Gaussian naive Bayes',
                disabled=False,
                indent=False
                )

# interaction between data_size radio button and gauss checkbox
def remove_gauss(data_size):
    if data_size['new'] == 'small':
       gauss.layout.display = 'flex'
    else:
       gauss.layout.display = 'none'

data_size.observe(remove_gauss, names='value')


hgboost = widgets.Checkbox(
                value=False,
                description='HGBoost',
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
       CPU.layout.display = 'none'

keras.observe(remove_CPU, names='value')

xgboost = widgets.Checkbox(
                value=False,
                description='XGBoost',
                description_tooltip='The model will use gradient boosting', 
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
#model_option2 = widgets.HBox([pipeline, xgboost])
model_option1 = widgets.VBox([xgboost, hgboost])
model_option2 = widgets.VBox([gauss, model_keras])
model_option_grp = widgets.VBox([model_option1, model_option2])

model_option_0 = widgets.VBox([level_0, model_caption_option, model_option_grp])

level_1 = widgets.Label(value='Level 1:')
level_1_model = widgets.RadioButtons(
                options=['regression', 'tree'],
                value='regression',
                description='Level 1 model type:',
                disabled=False
                )

cv = widgets.Checkbox(
                value=False,
                description='Level 1 with cross-validation',
                description_tooltip='The model will use cross-validation during training of level 1 model',
                disabled=False,
                indent=False
                )

# interaction between level_1_model radio button and cv checkbox
def remove_cv(level_1_model):
    if level_1_model['new'] == 'regression':
       cv.layout.display = 'flex'
    else:
       cv.layout.display = 'none'

level_1_model.observe(remove_cv, names='value')

model_option_1 = widgets.VBox([level_1, level_1_model, cv])

caption_threshold_mod = widgets.Label(value='Modelling thresholds:')

threshold_corr = widgets.FloatSlider(
                 value=0.95,
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

run = widgets.Button(
                description='Build',
                tooltip='If no error, you should find the generated notebook in the current folder',
                disabled=False,
                button_style='success', # 'success', 'info', 'warning', 'danger' or ''
                icon=''
                )

def on_run_clicked(b, problem_type, stacking, data_size, cv, gauss, hgboost, keras, CPU, xgboost, pipeline, fc, yb,\
                      seaborn, target, threshold_NaN, threshold_cat, threshold_Z, test_size, threshold_entropy,\
                      undersampling, undersampler, level_1_model,\
                      threshold_corr, threshold_model, threshold_score, threshold_feature, output):

    generate(problem_type.value, stacking.value, data_size.value, cv.value, gauss.value, hgboost.value,\
             keras.value, CPU.value, xgboost.value, pipeline.value, yb.value, seaborn.value, fc.selected,\
             target_cl.value, threshold_NaN.value, threshold_cat.value, threshold_Z.value,\
             test_size.value, threshold_entropy.value, undersampling.value, undersampler.value, level_1_model.value,\
             threshold_corr.value, threshold_model.value,\
             threshold_score.value, threshold_feature.value, output.value)

run.on_click(functools.partial(on_run_clicked, problem_type=problem_type, stacking=stacking,\
                               data_size=data_size, cv=cv, gauss=gauss, hgboost=hgboost, keras=keras, CPU=CPU, xgboost=xgboost,\
                               pipeline=pipeline, yb=yb, seaborn=seaborn, fc=fc, target=target,\
                               threshold_NaN=threshold_NaN, threshold_cat=threshold_cat, threshold_Z=threshold_Z,\
                               test_size = test_size, threshold_entropy=threshold_entropy, \
                               undersampling = undersampling, undersampler = undersampler, level_1_model=level_1_model, threshold_corr = threshold_corr,\
                               threshold_model = threshold_model, threshold_score = threshold_score,\
                               threshold_feature = threshold_feature, output=output))

build_tab = widgets.VBox([caption_output_file, output, run])

# tab : test
def test_endpoint(schema, test_type):
    """
    Generate a passing test (i.e. all values belong to the domains define in schema.csv
    """
    newline = '\n'
    comma_newline = ',\n'
    continuation = " \\"
    string = ""
    string = string + "curl -X 'POST'"  + continuation + newline
    string = string + "   'http://127.0.0.1:8000/predict'"  + continuation + newline
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
                 string = string + str(eval(schema['column_range'][ind])[1] + random.randint(0,10)) 
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

def test_generator(nb_ok, nb_ko):
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
           string = string + test_endpoint(schema, 'passing')
           string = string + str('echo -e ""\n')
       for ind in range(nb_ko):
           string = string + str('echo -e "Test KO: ' + str(ind)) + '"\n'
           string = string + test_endpoint(schema, 'notpassing')
           string = string + str('echo -e ""\n')
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

def on_test_clicked(b, nb_ok, nb_ko):

    test_generator(nb_ok.value, nb_ko.value)

test.on_click(functools.partial(on_test_clicked, nb_ok=nb_ok, nb_ko=nb_ko))

test_tab = widgets.VBox([caption_test, test_nb, test])


# tab : zip and clean
def zip_files(fc, output):
    # create a ZipFile object
    zipname = output + '.zip'
    zipObj = ZipFile(zipname, 'w')
    # Add multiple files to the zip
    nbname = output + '.ipynb'
    # add dataset (without full path)
    zipObj.write(fc, fc.split(os.sep)[-1])
    file_list = [nbname, 'server.ipynb', 'client.ipynb', 'EZS_deps/EZS_func.py', 'model.sav', 'schema.csv', 'test.sh', 'server.py']
    for file in file_list:
        zipObj.write(file)    
    # close the Zip File
    zipObj.close()
    
def delete_files(output):
    # delete work files
    nbname = output + '.ipynb'
    file_list = [nbname, 'model.sav', 'schema.csv', 'server.py', 'test.sh', 'client.ipynb', 'server.ipynb']  
    for file in file_list:
        os.remove(file)

def zip_and_clean(fc, output):
    nbname = output + '.ipynb'
    # Check if notebook exists
    if not os.path.isfile(nbname):
       print('It seems that the notebook is not generated.')
    # Check if model exists
    elif (not os.path.isfile('model.sav') or
         not os.path.isfile('schema.csv') or
         not os.path.isfile('server.py')):
         print('It seems that the model is not built.')
    else:
    # if everything is OK, zip and clean
         zip_files(fc, output)
         delete_files(output)
    
zip = widgets.Button(
                description='Zip & Clean',
                tooltip='Zip your files and clean the folder',
                disabled=False,
                button_style='success', # 'success', 'info', 'warning', 'danger' or ''
                icon=''
                ) 

def on_zip_clicked(b, fc, output):

    zip_and_clean(fc.selected, output.value)

zip.on_click(functools.partial(on_zip_clicked, fc=fc, output=output))

dev_names = ['EDA', 'Split', 'Model', 'Build']
dev_tabs = [EDA_tab, split_tab, model_tab, build_tab]
dev_gui = widgets.Tab(dev_tabs)
[dev_gui.set_title(i, title) for i, title in enumerate(dev_names)]

gui_names = ['File & Problem', 'Development', 'Test', 'Zip & Clean']
gui_tabs = [file_problem_tab, dev_gui, test_tab, zip] 
EZS_gui = widgets.Tab(gui_tabs)
[EZS_gui.set_title(i, title) for i, title in enumerate(gui_names)]