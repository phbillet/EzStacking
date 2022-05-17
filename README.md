# EZStacking
EZStacking is **[Jupyter notebook](https://jupyter.org/) generator** for machine learning (classification or regression problems) using [Scikit-Learn](https://scikit-learn.org/stable/) pipelines.

It can also be viewed as a **development tool**, because a notebook generated with EZStacking contains: 
* an **exploratory data analysis** (EDA) used to analyze quality of data
* a **modelling** producing a reduced-size stacked estimator  
* a **server** returning a prediction, a measure of the quality of input data and the execution time.

_**Notes:**_ 
* _EZStacking **must** be used with *.csv dataset using separator ','_  
* _the column names **must not** contain spaces (otherwise it will produce error during server generation)._

# EZStacking - How to install it
First you have to:
* install [Anaconda](https://anaconda.org/) 
* create the virtual environment EZStacking using the following command: `conda env create -f EZStacking.yaml`
* activate the virtual environment using the following command: `conda activate EZStacking`.

# EZStacking - How to use it

## Input file and problem characteristics

In Jupyter, first open the notebook named EZStacking.ipynb:
![First launch](/screenshots/EZStacking_first_launch.png)

Then run all:

![EZStacking GUI](/screenshots/EZStacking_gui.png)

First select your file, fill the target name, the problem type and the data size:

![EZStacking GUI](/screenshots/EZStacking_file_selection.png)

_Notes:_ 
* _the data size is "small", if the row number is less than 3000._
* _models depending on data size:_

|Model	|Data size | |Model |Data size |
|------|----------|-|------|----------|
|[XGBoost](https://arxiv.org/abs/1603.02754)	|both | |[SGD](https://scikit-learn.org/stable/modules/linear_model.html#stochastic-gradient-descent-sgd)	|large |
|[Support vector](https://scikit-learn.org/stable/modules/svm.html)	|large | |[Logistic Regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)	|small |
|[Keras](https://keras.io/guides/)	|both | |[Linear Regression](https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares)	|small |
|[Gaussian Process](https://scikit-learn.org/stable/modules/gaussian_process.html)	|small | |[ElasticNet](https://scikit-learn.org/stable/modules/linear_model.html#elastic-net) |both |
|[Decision Tree](https://scikit-learn.org/stable/modules/tree.html)	|small | |[Multilayer Perceptron](https://scikit-learn.org/stable/modules/neural_networks_supervised.html)	|small |
|[Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forests-of-randomized-trees) |both | |[KNeighbors](https://scikit-learn.org/stable/modules/neighbors.html) |small |
|[AdaBoost](https://scikit-learn.org/stable/modules/ensemble.html#adaboost)	|both | |[Gaussian Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html)	|small     |

## Options
Now, let's choose the options:

![EZStacking GUI](/screenshots/EZStacking_options.png)

### Processing options
|Option   | Notes                                                   |
|---------|---------------------------------------------------------|
|XGBoost  | the model will be built using gradient boosting         |
|Keras    | the model will be a neural network based on Keras       |

_Note: estimators based on Keras benefit from early stopping, those based on XGBoost not_

### Visualization options
|Option        | Notes                                                              |
|--------------|--------------------------------------------------------------------|
|Yellow bricks | The graphics will be constructed with Matplotlib and Yellow bricks |
|Seaborn       | The graphics will be constructed with Matplotlib and Seaborn       |

_Note: the visualisation option Seaborn can produce time consuming graphics._

## Thresholds
### Thresholds in EDA
![EZStacking Thresholds EDA](/screenshots/EZStacking_thresholds_eda.png)

_Notes:_
* _threshold_cat: threshold for categorical data, if the number of different values in a column is less than this number, the column will be considered as a categorical column_
* _threshold_NaN: threshold for NaN, if the proportion of NaN is greater than this number the column will be dropped_
* _threshold_Z: threshold for outliers, if the Z_score is greater than this number, the row will be dropped._

### Thresholds in EDA
![EZStacking Thresholds Modelling](/screenshots/EZStacking_thresholds_mod.png)

_Notes:_
* _threshold_corr: if the correlation is greater than this number the column will be dropped'_
* _threshold_model: keep this number of best models_
* _threshold_score: keep models having test score greater than this number._

## Output file name
Simply enter a file name:

![EZStacking Output](/screenshots/EZStacking_output.png)

## Notebook generation and execution
Just click on the button ![EZStacking Generate](/screenshots/EZStacking_generate.png), you should find your notebook in the current folder (otherwise a Python error and maybe an issue in Github).
If the option ![auto execution](/screenshots/auto_exec.png) is checked, the notebook will be processed. 
If it ends correctly, the result looks like: ![exec_OK](/screenshots/exec_OK.png), otherwise: ![exec_KO](/screenshots/exec_KO.png).

# EZStacking - As development tool
## Development process
Once the first workbook has been generated, the development process must be launched.

It follows the following workflow:
![EZStacking_development_process](/screenshots/EZStacking_development_process.png)

## Data quality & EDA
EDA can be seen as a toolbox to evaluate data quality like: 
* dataframe statistics
* compression
* cleaning
* ranking / correlation if [Yellow Brick](https://www.scikit-yb.org) option is checked
This process returns:
* a data schema _i.e._ a description of the input data with data type and associated domain: 
  * minimum and maximum for continous features, 
  * a list for categorical features
* a list of columns that should be suppressed at the departure of the EDA  

## Modelling
The first generated model is structured as follow:

![EZStacking_initial_model](EZStacking_initial_model.png)

This initial model is big, the modelling process reduces its size in terms of models and features as follw:
1. the whole estimator is trained 
2. the set of estimators is reduced according to the scores and the importance of the models
3. the reduced estimator is trained 
4. the feature importance graphic gives which columns could also be suppressed.

## Serving the model:
EZStacking also generates a server based on [FastAPI](https://fastapi.tiangolo.com/), it returns:
* a prediction 
* a list of columns in error (_i.e._ the value does not belong to the domain given in the schema)
* the elapsed and CPU times.

Example: ![EZStacking_server_response](EZStacking_server_response.png)

# Some results
Some results are given [here](https://github.com/phbillet/EZStacking/tree/main/examples).

# Resources used for this project:
* [Python courses](https://youtu.be/82KLS2C_gNQ) (in French), Guillaume Saint-Cirgue
* [Machine learning](https://www.coursera.org/learn/machine-learning), Andrew Ng & Stanford University
* [Deep Learning Specialization](https://www.deeplearning.ai/program/deep-learning-specialization/), Andrew Ng & DeepLearning.ai
* [Advanced Machine Learning Specialization](https://www.coursera.org/specializations/aml), HSE University
* [Machine Learning Mastery](https://machinelearningmastery.com/), Jason Brownlee
* ...










