# EZStacking
EZStacking is [Jupyter notebook](https://jupyter.org/) generator for machine learning (classification or regression problems).

A notebook generated with EZStacking contains: 
* an exploratory data analysis EDA:
  * dataframe statistics
  * compression
  * cleaning
  * ranking / correlation
* a modelling:
  * with or without stacked generalization, Keras, XGBoost or Pipeline
  * model evaluation.

_Note: **EZStacking must be used with *.csv dataset**._

# EZStacking - What will you need?
|Package                                                    | Version |
|-----------------------------------------------------------|---------|
|[pandas](https://pandas.pydata.org/)                       | 1.3.3   | 
|[scikit-learn](https://scikit-learn.org/)                  | 0.24.1  |
|[keras](https://keras.io/)                                 | 2.4.3   |
|[xgboost](https://xgboost.readthedocs.io/en/latest/)       | 1.3.3   |
|[polylearn](https://contrib.scikit-learn.org/polylearn/)   | 0.1     |
|[scipy](https://www.scipy.org)                             | 1.6.0   |
|[yellowbrick](https://www.scikit-yb.org)                   | dev     |
|[matplotlib](https://matplotlib.org/)                      | 3.4.2   |
|[seaborn](https://seaborn.pydata.org/)                     | 0.11.2  |
|[graphviz](https://graphviz.org/)                          | 2.40.1  |
|[python-graphviz](https://graphviz.org/)                   | 0.16    |
|[nbformat](https://nbformat.readthedocs.io/en/latest/)     | 5.1.2   |
|[ipywidgets](https://ipywidgets.readthedocs.io/en/latest/) | 7.6.3   |
|[ipyfilechooser](https://github.com/crahan/ipyfilechooser) | 0.6.0   |

_Note: Yellow Brick must be installed from source._

# EZStacking - How to use it

In Jupyter, first open the notebook named EZStacking.ipynb:
![First launch](/screenshots/EZStacking_first_launch.png)
Then run all:
![EZStacking GUI](/screenshots/EZStacking_gui.png)
First select you file, fill the target name, the problem type and the data size:
![EZStacking GUI](/screenshots/EZStacking_file_selection.png)

_Note: the data size is "small", if the row number is less than 3000._

Now let's choose the options:

![EZStacking GUI](/screenshots/EZStacking_options.png)

## Options
### Processing options
|Option   | Notes                                                   |
|---------|---------------------------------------------------------|
|Stacking | the model will be built using stacked generalization    |
|XGBoost  | the model will be built using gradient boosting         |
|Keras    | the model will be a neural network based on Keras       |
|Pipeline | the model will be built integrating a preprocessing step|

_Notes:_ 
* _the processing options XGBoost and Keras must not be both checked if Stacking option is not checked_
* _if the processing option Pipeline is checked, you may have to modify preprocessors_

## Visualization options
|Option        | Notes                                                              |
|--------------|--------------------------------------------------------------------|
|Yellow bricks | The graphics will be constructed with Matplotlib and Yellow bricks |
|Seaborn       | The graphics will be constructed with Matplotlib and Seaborn       |

_Note: the visualisation option Seaborn can produce time consuming graphics

## Thresholds
![EZStacking Thresholds](/screenshots/EZStacking_thresholds.png)

_Notes:_
* threshold_cat: threshold for categorical data, if the number of different values in a column is less than this number, the column will be considered as a categorical column.
* threshold_NaN: hreshold for NaN, if the proportion of NaN is greater than this number the column will be dropped
* threshold_Z: threshold for outliers, if the Z_score is greater than this number, the row will be dropped
