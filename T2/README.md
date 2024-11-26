# Bias Mitigation Analysis
These scripts analyzes the bias and mitigation techniques that can be applied to clasification models.

## Requirements

See _requirements.txt_

## Usage

Ensure you have the dataset file (diabetes_binary_5050split_health_indicators_BRFSS2015.csv) in the same directory as the scripts.

Run the scripts:

- For analyzing which model works best on the dataset (optional, the rest of the scripts only use LogisticRegression):
    `python gridsearch.py`

- For analyzing bias mitigation techniques one by one:
    `python each_method_processing.py`

- For analyzing bias mitigation techniques one after another:
    `reweigh_and_eq_processing.py`
    `python all_methods_processing.py`

Every script will print its results to the console.