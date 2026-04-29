# Student Performance and Internet Access Project

## Overview

This project analyzes a student performance dataset to study the relationship between internet access and final exam performance. It also builds machine learning models to predict `final_exam_score` using student academic, behavioral, and demographic features.

The project is designed to avoid data leakage. Columns such as `overall_score` and `grade` are removed before model training because they are likely calculated from student score variables.

## Dataset

Dataset source:

https://www.kaggle.com/datasets/upcoderr/student-performance-analytics-dataset

Expected dataset file name:

student_performance_data.csv

Place the dataset inside the `data` folder:

data/student_performance_data.csv

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
    eda.py
    modeling.py
    stats_analysis.py
    extra_figures.py
    make_all.py
  README.md
  requirements.txt

## Requirements

This project uses Python 3 and the following packages:

pandas
numpy
matplotlib
scikit-learn
scipy
statsmodels

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

Exploratory data analysis
Model training
Model evaluation
Statistical analysis
Extra figure generation

## Running Files Separately

You can also run each script separately.

Run exploratory data analysis:

py src/eda.py

Run model training and evaluation:

py src/modeling.py

Run statistical analysis:

py src/stats_analysis.py

Create extra report figures:

py src/extra_figures.py

## Source Code Files

### `src/config.py`

Stores project settings, including file paths, target variable, feature lists, leakage columns, and train test split settings.

### `src/data_utils.py`

Loads the dataset, checks required columns, prints basic dataset information, and creates the feature matrix and target variable.

### `src/eda.py`

Creates exploratory data analysis tables and figures.

### `src/modeling.py`

Trains and evaluates the prediction models.

Models included:

Mean baseline
Linear regression
Random forest regression

### `src/stats_analysis.py`

Runs statistical analysis for internet access and final exam score.

Methods included:

Group summary
Welch's t test
Cohen's d
Controlled regression

### `src/extra_figures.py`

Creates additional figures used in the final report.

### `src/make_all.py`

Runs all project scripts in order.

## Output Folders

### `figures/`

Stores generated plots used for analysis and the report.

### `results/`

Stores generated CSV and text files from the analysis.

Expected outputs include:

model_results.csv
internet_access_ttest_results.csv
internet_access_score_summary.csv
controlled_regression_summary.txt
controlled_regression_coefficients.csv
random_forest_feature_importance.csv
numeric_summary.csv
missing_values.csv

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

## Notes

Before running the project, make sure:

The dataset is downloaded from Kaggle.
The dataset is named student_performance_data.csv.
The dataset is placed inside the data folder.
All required packages are installed.