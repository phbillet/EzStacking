# EZStacking
EZStacking is **[Jupyter notebook](https://jupyter.org/) generator** for [**supervised learning**](https://en.wikipedia.org/wiki/Supervised_learning) problems using [**Scikit-Learn pipelines**](https://scikit-learn.org/stable/modules/compose.html#combining-estimators) and [**stacked generalization**](https://scikit-learn.org/stable/modules/ensemble.html#stacking). 

EZStacking handles **classification** and **regression** problems for **structured data** (_cf. Notes hereafter_). 

It can also be viewed as a [**development tool**](#ezstacking---as-development-tool), because a notebook generated with EZStacking contains: 
* an [**exploratory data analysis (EDA)**](#data-quality--eda) used to assess data quality
* a [**modelling**](#modelling) producing a reduced-size stacked estimator  
* a [**server**](#serving-the-model) returning a prediction, a measure of the quality of input data and the execution time.

Those three activities and the intrinsic **interactivity** of **notebooks** offer the possibility to **recursively develop** a custom estimator while keeping its **construction process** (simply saving the different relevant versions of the notebook).

_Notes:_ 
* _EZStacking **must** be used with *.csv dataset using separator ','_  
* _the column names **must not** contain spaces (otherwise it will produce error during server generation)._

# EZStacking - How to install it
First you have to:
* install [**Anaconda**](https://anaconda.org/) 
* create the **virtual environment** EZStacking using the following command: `conda env create -f EZStacking.yaml`
* **activate** the virtual environment using the following command: `conda activate EZStacking`
* **install kernel** in ipython using the following command: `ipython kernel install --user --name=ezstacking`
* launch the **Jupyter server** using the following command: `jupyter notebook --no-browser`

# EZStacking - How to use it

## Input file and problem characteristics

In Jupyter, first open the notebook named `EZStacking.ipynb`:
![First launch](/screenshots/EZStacking_first_launch.png)

Then `run all`:

![EZStacking GUI](/screenshots/EZStacking_gui.png)

First select your **file**, fill the **target** name (_i.e._ the variable on which we want to make predictions), the **problem type** (_i.e._ **classification** if the target is discrete, **regression** otherwise) and the **data size**:

![EZStacking GUI](/screenshots/EZStacking_file_selection.png)

_Notes:_ 
* _the data size is **small**, if the number of row is smaller than **3000**._
* _depending on the data size, EZStacking uses those estimators for the level 0:_

|Model	|Data size | |Model |Data size |
|------|----------|-|------|----------|
|[XGBoost](https://arxiv.org/abs/1603.02754)	|both | |[SGD](https://scikit-learn.org/stable/modules/linear_model.html#stochastic-gradient-descent-sgd)	|large |
|[Support vector](https://scikit-learn.org/stable/modules/svm.html)	|large | |[Logistic Regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)	|small |
|[Keras](https://keras.io/guides/)	|both | |[Linear Regression](https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares)	|small |
|[Gaussian Process](https://scikit-learn.org/stable/modules/gaussian_process.html)	|small | |[ElasticNet](https://scikit-learn.org/stable/modules/linear_model.html#elastic-net) |both |
|[Decision Tree](https://scikit-learn.org/stable/modules/tree.html)	|small | |[Multilayer Perceptron](https://scikit-learn.org/stable/modules/neural_networks_supervised.html)	|small |
|[Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forests-of-randomized-trees) |both | |[KNeighbors](https://scikit-learn.org/stable/modules/neighbors.html) |small |
|[AdaBoost](https://scikit-learn.org/stable/modules/ensemble.html#adaboost)	|both | |[Gaussian Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html)	|small     |
|[Histogram-based Gradient Boosting](https://scikit-learn.org/stable/modules/ensemble.html#histogram-based-gradient-boosting)|both |

## Options
Now, let's choose the options:

![EZStacking GUI](/screenshots/EZStacking_option.png)

### Processing options
|Option   | Notes                                                   |
|---------|---------------------------------------------------------|
|XGBoost  | the model will be built using gradient boosting         |
|Keras    | the model will be a neural network based on Keras       |

_Note: estimators based on **Keras** or on **Histogram-Based Gradient Boosting** benefit from [**early stopping**](https://en.wikipedia.org/wiki/Early_stopping), those based on XGBoost not._

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
* _threshold_cat: if the **number of different values** in a column is less than this number, the column will be considered as a **categorical column**_
* _threshold_NaN: if the proportion of **NaN** is greater than this number the column will be **dropped**_
* _threshold_Z: if the **Z_score**  (indicating **outliers**) is greater than this number, the row will be **dropped**._

### Thresholds in modelling
![EZStacking Thresholds Modelling](/screenshots/EZStacking_thresholds_mod.png)

_Notes:_
* _threshold_corr: if the **correlation** is greater than this number the column will be **dropped**_
* _threshold_score:  **keep** models having **test score** greater than this number._
* _threshold_model: **keep** this number of **best models** (in the sens of **model importance**)_
* _threshold_feature: **keep** this number of **most important features**_

## Output file name
Simply enter a file name:

![EZStacking Output](/screenshots/EZStacking_output.png)

## Notebook generation and execution
Just click on the button ![EZStacking Generate](/screenshots/EZStacking_generate.png), you should find **your notebook** in the **current folder** (otherwise a Python error and maybe an issue in Github).
If the option ![auto execution](/screenshots/auto_exec.png) is checked, the notebook will be processed. 
If it ends correctly, the result looks like: ![exec_OK](/screenshots/exec_OK.png), otherwise: ![exec_KO](/screenshots/exec_KO.png).

# EZStacking - As development tool
## Development process
Once the first notebook has been generated, the development process can be launched.

You simply have to follows the following workflow:

<img src="/screenshots/EZStacking_development_process.png" data-canonical-src="/screenshots/EZStacking_development_process.png" width="50%" height="50%" />

## Data quality & EDA
EDA can be seen as a **toolbox** to evaluate **data quality** like: 
* dataframe **statistics**
* **cleaning** _i.e._ **NaN** and **outlier** dropping
* ranking / **correlation** 

_Notes: the EDA step **doest not** modify data, it just indicates which actions should be done_

This process returns:
* a **data schema** _i.e._ a description of the input data with data type and associated domain: 
  * minimum and maximum for **continous features**, 
  * a list for **categorical features**
* a list of columns `dropped_cols` that should be **suppressed** (simply adding at the departure of the EDA this list to the variable `user_drop_cols`, then it is necessary to **re-launch** from the EDA). 

_Notes:_
* _[Yellow Brick](https://www.scikit-yb.org) offers different graphs associated to ranking and correlation and many more informations._
* _The main steps of data **pre-processing**:_
  1. _not all estimators support **NaN** : they must be corrected using **imputation**_
  2. _data **normalization** and **encoding** are also key points for [**successful learning**](https://machinelearningmastery.com/how-to-improve-neural-network-stability-and-modeling-performance-with-data-scaling/)_ 
  3. _only **correlations** with the target are interesting, the others must be removed (for [linear algebra reasons](https://datascience.stackexchange.com/questions/24452/in-supervised-learning-why-is-it-bad-to-have-correlated-features))_
* _Those steps are implemented in the **modelling pipeline**._

## Modelling
The **first step** of modelling is structured as follow:

<img src="/screenshots/EZStacking_initial_model.png" data-canonical-src="/screenshots/EZStacking_initial_model.png" width="50%" height="50%" />

This initial model is big, the modelling process **reduces** its size in terms of **models** and **features** as follow:
1. the set of **estimators** is reduced according to the **test scores** and the **importance** of each level 0 models
2. the reduced estimator is trained 
3. the **feature importance** graphic indicates which columns could also be **dropped**
4. those columns are added to variable `dropped_cols` depending on the value of `threshold_feature`
5. `dropped_cols` can be added to `user_drop_cols` at the departure of the EDA (then it is necessary to **re-launch** from the EDA).

_Notes:_ 
* _the calculation of the **model importance** is based on the coefficients of the regularized linear regression used as level 1 estimator_
* _the **feature importance** is computed using [permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html)_
* _it is important not to be too stingy, it is not necessary to remove too many estimators and features, as this can lead to a decrease in performance.._

## Serving the model
EZStacking also generates an API based on [**FastAPI**](https://fastapi.tiangolo.com/).

The complete **development process** produces three objects:
* a **schema**
* a **model**
* a **server** source.

They can be used as basic **prediction service** returning:
* a **prediction** 
* a list of columns in **error** (_i.e._ the value does not belong to the domain given in the schema)
* the elapsed and CPU **times**.

Example: 

![EZStacking_server_response](/screenshots/EZStacking_server_response.png)

# Some results
Some results are given [here](https://github.com/phbillet/EZStacking/tree/main/examples).

# Resources used for this project:
* [Python courses](https://youtu.be/82KLS2C_gNQ) (in French), Guillaume Saint-Cirgue
* [Machine learning](https://www.coursera.org/learn/machine-learning), Andrew Ng & Stanford University
* [Deep Learning Specialization](https://www.deeplearning.ai/program/deep-learning-specialization/), Andrew Ng & DeepLearning.ai
* [Advanced Machine Learning Specialization](https://www.coursera.org/specializations/aml), HSE University
* [Machine Learning Engineering for Production](https://www.coursera.org/specializations/machine-learning-engineering-for-production-mlops), Andrew Ng & DeepLearning.ai
* [Machine Learning Mastery](https://machinelearningmastery.com/), Jason Brownlee
* ...
