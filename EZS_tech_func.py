import io
import papermill as pm
import functools
import pandas as pd
import numpy as np
import nbformat as nbf
import ipywidgets as widgets
from IPython.display import display
from ipyfilechooser import FileChooser

def generate(problem_type, stacking, data_size, with_keras, with_xgb, with_pipeline, yb, seaborn, file, target_col,\
             threshold_NaN, threshold_cat, threshold_Z, threshold_corr, threshold_model, threshold_score, \
             threshold_feature, auto_exec, output):
    """
    Initialize the notebook, analyze input data from GUI, generate, write and execute the notebook.
    """
    user_drop_cols=[]
    features_of_interest = []
    nb = analyze(problem_type, stacking, data_size, with_keras, with_xgb, with_pipeline, yb, seaborn, file, target_col,\
                 user_drop_cols, features_of_interest, threshold_NaN, threshold_cat, threshold_Z,\
                 threshold_corr, threshold_model, threshold_score, threshold_feature)
    fname = output + '.ipynb'
    with open(fname, 'w') as f:
         nbf.write(nb, f)
    
    if auto_exec:
       pm.execute_notebook(fname, fname)

def set_config(with_keras, with_xgb, with_pipeline, problem_type, stacking, yb, seaborn, data_size):
    """
    Set configuration: load configuration database, generate the different dataframes used to generate
    cells of the notebook according to the data from the GUI.
    """

    xls = pd.ExcelFile('EZStacking_config.ods', engine="odf")
    meta_package = pd.read_excel(xls, 'meta_package')
    package_source = pd.read_excel(xls, 'package_source')
    package = pd.read_excel(xls, 'package')
    document = pd.read_excel(xls, 'document')
    
    meta_package.loc[meta_package['meta_package_index'] == 'STACK', ['meta_package_valid']] = stacking
    meta_package.loc[meta_package['meta_package_index'] == 'KER', ['meta_package_valid']] = with_keras
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
        else:
           string = string + "import " + str(row[1]) + "\n"
        if row[3] != 'None':
           string = string + str(row[3]) + "\n"
        
    for index, row in pd_pk_from.iterrows():
        string = string + "from " + str(row[1]) + " import " +  str(row[2]) + "\n"
        
    code = string
    nb['cells'].append(nbf.v4.new_code_cell(code))
    
    return nb

def analyze(problem_type, stacking, data_size, with_keras, with_xgb, with_pipeline, yb, seaborn, file,\
            target_col, user_drop_cols, features_of_interest, threshold_NaN, threshold_cat, threshold_Z,\
            threshold_corr, threshold_model, threshold_score, threshold_feature):

    """
    Analyze input data from GUI, set configuration, generate the different cells of the notebook
    """
    pd_pk_import, pd_pk_from, pd_level_0, pd_document, pd_tree = set_config(with_keras, with_xgb, with_pipeline,\
                                                                            problem_type, stacking, yb, seaborn, data_size)
    
    nb = nbf.v4.new_notebook()
    kernelspec = dict(
            display_name='EZStacking',
            name='ezstacking',
            language='python')
    nb.metadata['kernelspec'] = kernelspec
    
    nb['cells'] = []
    
    for index, row in pd_document.iterrows():
        if index == 1:
           nb = load_package(nb, pd_pk_import, pd_pk_from) 
        if row[1] != 'None':
           if row[0] == ' ': 
              text = str(row[1])            
           else:
              text = str(row[0]) + ' ' + str(row[1])
           nb['cells'].append(nbf.v4.new_markdown_cell(text))
        if row[2] != 'None':
           code = eval(row[2])
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
                 "    model.add(Dense(10 * layer_size, activation='relu')) \n" + \
                 "    model.add(BatchNormalization()) \n" + \
                 "    model.add(Dropout(0.5)) \n" + \
                 "#    model.add(LayerNormalization()) \n" + \
                 "    model.add(Dense(layer_size, activation='relu')) \n" + \
                 "    model.add(BatchNormalization()) \n" + \
                 "    model.add(Dropout(0.5)) \n" + \
                 "#    model.add(LayerNormalization()) \n" + \
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
                 "    model.add(Dense(10 * layer_size, activation='relu')) \n" + \
                 "    model.add(BatchNormalization()) \n" + \
                 "    model.add(Dropout(0.5)) \n" + \
                 "#    model.add(LayerNormalization()) \n" + \
                 "    model.add(Dense(layer_size, activation='relu')) \n" + \
                 "    model.add(BatchNormalization()) \n" + \
                 "    model.add(Dropout(0.5)) \n" + \
                 "#    model.add(LayerNormalization()) \n" + \
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
caption_fc = widgets.Label(value='Select your input file:')
fc = FileChooser('./')
file = widgets.VBox([caption_fc, fc])

caption_target = widgets.Label(value='Enter target name:')
target_cl = widgets.Text(
                value='column name',
                placeholder='Type something',
                description='Target:',
                disabled=False   
                )
target = widgets.VBox([caption_target, target_cl]) 

caption_problem_option = widgets.Label(value='Problem and data characteristics:')
caption_problem_type = widgets.Label(value='Select your problem type:') 
problem_type = widgets.RadioButtons(
                options=['classification', 'regression'],
                description='Problem type:',
                description_tooltip='If the target is categorical then choose classification, regression otherwise.',
                disabled=False
                )
problem = widgets.VBox([caption_problem_type, problem_type])
caption_data_size = widgets.Label(value='Select your data size:')
data_size = widgets.RadioButtons(
                options=['small', 'large'],
                description='Data size:',
                description_tooltip='Choose large, if the file contains more 5000 rows',
                disabled=False
                )
data = widgets.VBox([caption_data_size, data_size])

problem_option1 = widgets.HBox([problem, data])
problem_option = widgets.VBox([caption_problem_option, problem_option1])

caption_option = widgets.Label(value='Select your options:')
model_caption_option = widgets.Label(value='Optional Models:')
stacking = widgets.Checkbox(
#                value=False,
                value=True,
                description='Stacking',
                description_tooltip='The model will use stacked generalization',
                disabled=False,
                indent=False
                )

keras = widgets.Checkbox(
                value=False,
                description='Keras',
                description_tooltip='The model will use Keras neural network',
                disabled=False,
                indent=False
                )

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
#model_option1 = widgets.HBox([stacking, keras])
#model_option2 = widgets.HBox([pipeline, xgboost])
model_option1 = widgets.VBox([xgboost, keras])

model_option = widgets.VBox([model_caption_option, model_option1])

visualizer_caption_option = widgets.Label(value='Visualizers:')
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
visualizer_option = widgets.VBox([visualizer_caption_option, yb, seaborn])

option = widgets.VBox([caption_option, widgets.HBox([model_option, visualizer_option])])

caption_threshold_EDA = widgets.Label(value='Fix your thresholds for EDA:')

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

caption_threshold_mod = widgets.Label(value='Fix your thresholds for modelling:')

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
                description_tooltip='Keep this number of best models',
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

threshold1 = widgets.VBox([caption_threshold_EDA, threshold_cat, threshold_NaN, threshold_Z])
threshold2 = widgets.VBox([caption_threshold_mod, threshold_corr, threshold_score, threshold_model, threshold_feature])
threshold = widgets.HBox([threshold1, threshold2])

caption_output_file = widgets.Label(value='Enter output file name:')
output = widgets.Text(
                value='output file',
                placeholder='Type something',
                description_tooltip='Output filename',
                description='Output:',
                disabled=False   
                )

run_button = widgets.Button(
                description='Generate',
                tooltip='If no error, you should find the generated notebook in the current folder',
                disabled=False,
                button_style='success', # 'success', 'info', 'warning', 'danger' or ''
                icon=''
                )
                
auto_exec = widgets.Checkbox(
                value=False,
                description='Auto execute notebook',
                description_tooltip='The generated notebook will be automatically executed',
                disabled=False,
                indent=False
                )

run = widgets.HBox([run_button, auto_exec])

generator = widgets.VBox([caption_output_file, output, run])

def on_button_clicked(b, problem_type, stacking, data_size, keras, xgboost, pipeline, fc, yb,\
                      seaborn, target, threshold_NaN, threshold_cat, threshold_Z, threshold_corr,\
                      threshold_model, threshold_score, threshold_feature, auto_exec, output):

    generate(problem_type.value, stacking.value, data_size.value, keras.value, xgboost.value, pipeline.value, yb.value,\
             seaborn.value, fc.selected, target_cl.value, threshold_NaN.value, threshold_cat.value,\
             threshold_Z.value, threshold_corr.value, threshold_model.value, threshold_score.value,\
             threshold_feature.value, auto_exec.value, output.value)

run_button.on_click(functools.partial(on_button_clicked, problem_type=problem_type, stacking=stacking,\
                               data_size=data_size,keras=keras, xgboost=xgboost, pipeline=pipeline, yb=yb,\
                               seaborn=seaborn,  fc=fc, target=target, threshold_NaN=threshold_NaN,\
                               threshold_cat=threshold_cat, threshold_Z=threshold_Z, threshold_corr = threshold_corr,\
                               threshold_model = threshold_model, threshold_score = threshold_score,\
                               threshold_feature = threshold_feature, auto_exec = auto_exec, output=output))

EZS_gui = widgets.VBox([file, target, problem_option, option, threshold, generator])
