# Student Performance and Internet Access Project

## Overview

This project studies whether internet access is associated with student final exam performance. It also compares three regression models for predicting final_exam_score from student academic, behavioral, and demographic features.

The project uses a leakage prevention step before model training. The columns overall_score and grade are removed because they are likely calculated from student score information and may indirectly reveal the target variable.

## Research Questions

1. Are students with internet access associated with higher final exam scores?
2. Can final exam scores be predicted from the available student features?
3. Does increasing model complexity improve prediction performance?

## Dataset

Dataset source:

https://www.kaggle.com/datasets/upcoderr/student-performance-analytics-dataset

Expected dataset file name:

student_performance_data.csv

Place the dataset inside the data folder:

data/student_performance_data.csv

The dataset contains 10,000 student records and 14 columns.

## Project Structure

student_performance_project/
  data/
    student_performance_data.csv
  figures/
  report/
    references.bib
  results/
  src/
    config.py
    data_utils.py
    train_models.py
    make_figures.py
    make_all.py
  README.md
  requirements.txt

## Requirements

This project uses Python 3 and the following packages:

pandas
numpy
matplotlib
scikit-learn

Install all required packages with:

py -m pip install -r requirements.txt

On some systems, use:

python -m pip install -r requirements.txt

## How to Run

From the main project folder, run:

py src/make_all.py

On some systems, use:

python src/make_all.py

This command runs the full project pipeline:

1. Loads and checks the dataset
2. Removes leakage and target columns from the feature matrix
3. Trains the three regression models
4. Evaluates models with RMSE, MAE, and R2
5. Saves model results and predictions
6. Creates figures for the report

## Running Files Separately

You can also run each script separately.

Run model training and evaluation:

py src/train_models.py

Create report figures:

py src/make_figures.py

## Source Code Files

### src/config.py

Stores project settings, including file paths, target variable, feature lists, leakage columns, test size, and random seed.

### src/data_utils.py

Loads the dataset, checks required columns, prints basic dataset information, and creates the feature matrix X and target variable y.

### src/train_models.py

Trains and evaluates the prediction models.

Models included:

1. Linear Regression
2. Random Forest Regression
3. Neural Network Regression

The script saves:

results/model_results.csv
results/test_predictions.csv

### src/make_figures.py

Creates figures used in the final report.

Figures created:

figures/final_exam_score_distribution.png
figures/mean_score_by_internet_access.png
figures/model_comparison_rmse.png
figures/model_comparison_mae.png
figures/linear_regression_actual_vs_predicted.png
figures/random_forest_regression_actual_vs_predicted.png
figures/neural_network_regression_actual_vs_predicted.png

### src/make_all.py

Runs the full pipeline by executing train_models.py and make_figures.py.

## Output Folders

### figures/

Stores generated plots used in the final report.

Expected outputs include:

final_exam_score_distribution.png
mean_score_by_internet_access.png
model_comparison_rmse.png
model_comparison_mae.png
linear_regression_actual_vs_predicted.png
random_forest_regression_actual_vs_predicted.png
neural_network_regression_actual_vs_predicted.png

### results/

Stores generated CSV files from model training and evaluation.

Expected outputs include:

model_results.csv
test_predictions.csv

## Data Leakage Prevention

The following columns are removed before model training:

student_id
final_exam_score
overall_score
grade

Reasons:

student_id is only an identifier.
final_exam_score is the target variable.
overall_score is likely calculated from score variables.
grade is likely calculated from score variables.

Removing these columns makes the prediction task more realistic.

## Models

This project compares three levels of model complexity.

### Linear Regression

This is the basic model. It is simple and interpretable.

### Random Forest Regression

This is the medium complexity model. It can capture nonlinear patterns and feature interactions.

### Neural Network Regression

This is the complex model. It uses a multilayer perceptron with two hidden layers.

## Evaluation Metrics

The models are evaluated with:

RMSE
MAE
R2

Lower RMSE and MAE values are better.

R2 shows how much variation in final_exam_score is explained by the model.

## Notes

Before running the project, make sure:

1. The dataset is downloaded from Kaggle.
2. The dataset is named student_performance_data.csv.
3. The dataset is placed inside the data folder.
4. All required packages are installed.
5. You run commands from the main project folder.

## Main Result Summary

The project finds a small descriptive difference in mean final exam score between students with and without internet access. However, the three prediction models perform weakly after removing outcome related variables. This suggests that the available input features do not contain strong predictive signal for individual final exam scores.