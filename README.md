# A Dynamic Ensemble Model for short-term Forecasting in Pandemic Situations - Botz et al 

This repository is part of the publication entitled "a dynamic ensemble model for short-term forecasting in pandemic situations". In the following we will describe how this repository is organized and how to run the scripts.
For questions please contact Jonas Botz (jonas.botz@scai.fraunhofer.de).

## Organization:


    .
    ├── environment.yml
    ├── README.md
    └── src                                 -> includes necessary scripts for loading data and running models  
        ├── data                            -> scripts for data loading and processing
        │   ├── datasets.py
        │   ├── __init__.py
        │   └── load_data.py
        ├── __init__.py
        ├── models                          -> scripts for modeling, tuning and evaluation
        │   ├── arima.py                    -> ARIMA base-model
        │   ├── __init__.py
        │   ├── main_ens_model.py           -> meta-model without metadata
        │   ├── main_meta.py                -> meta-model with metadata
        │   ├── main_optuna_ens.py          -> optuna tuning for meta-model without metadata
        │   ├── main_optuna_LSTM.py         -> optuna tuning for LSTM base-model
        │   ├── main_optuna_meta.py         -> optuna tuning for meta-model with metadata
        │   ├── main_optuna_trees.py        -> optuna tuning for XGBoost and Random Forest 
        │   ├── main.py                     -> script for running tuned basemodel and baseline ensemble methods
        │   ├── networks.py                 -> LSTM base-model and meta-model
        │   ├── parser.py                   -> configurations
        │   ├── regression.py               -> Linear Regression 
        │   ├── solver_ens_model.py         -> for running meta-model without metadata
        │   ├── solver_meta_model.py        -> for running meta-model with metadata
        │   ├── solver_optuna.py            -> for running optuna tuning for LSTM base-model
        │   ├── solver.py                   -> for running LSTM base-model
        │   ├── test.py                     -> for testing LSTM and Regression base-models
        │   ├── train_ens_model.py          -> for training meta-model without metadata
        │   ├── train_meta_model.py         -> for training meta-model with metadata
        │   ├── train_optuna.py             -> for training LSTM base-model in optuna tuning
        │   ├── train.py                    -> for training LSTM base-model
        │   ├── tree_models.py              -> XGBoost and Random Forest
        │   ├── utils.py                    -> Definition of Metrics
        │   └── visualizations.py           -> Barplots, Violinplots and Boxplots
        └── README.md


## Data:

We used German COVID-19, Influenza and SARI Surveillance Data provided by the RKI, available here: https://github.com/robert-koch-institut
We used French COVID-19 Surveillance Data provided by SPF, available here: https://www.data.gouv.fr/fr/organizations/sante-publique-france

The metadata was accessed via the Google Trends API. For access you have to apply here: https://docs.google.com/document/d/1Ybu3gHUHtcSXXzgDJ-m7PPto9tw0QG8A5oOBsFP2jao/edit?pli=1#heading=h.qye80d9e325z
Then follow Danqi et al.: https://www.nature.com/articles/s41598-023-48096-3

For smoothing we applied a centered moving average over seven days (for the COVID-19 and metadata).



## Configuration Parameters:

For running the scripts following configuration parameters need to be set (we also mention the parameters that we used):
1. iterations - number of test windows (140 for COVID, 30 for Influenza, 80 for SARI)
2. period - training period (70 for daily data, 52 for weekly data)
3. size_SW - fitting window size (7 for daily data, 5 for weekly data)
4. size_PW - prediction window size (14 for daily data, 2 for weekly data)
5. exp_name - name of the experiment to be correctly stored
6. num_trials - number of optuna trials
7. num_epochs - number of epochs the LSTM or meta-model are trained 

There are more, which are specific to the models, for example the prediction type (stacking or selection) for the meta-model. How to set these should become clear when looking at the corresponding scripts. 

## Model Running:

1. Hyperparameter Tuning Base-Models:
    - main_optuna_LSTM.py
    - main_optuna_trees.py

2. Base-Model and Baseline Ensemble Evaluation:
    - main.py (check that the correct optuna databases are selected)

3. Hyperparameter Tuning for Meta-Model:
    - main_optuna_ens.py
    - main_optuna_meta.py*

4. Meta-Model Evaluation:
    - main_ens_model.py
    - main_meta.py*

*Only if metadata is available

The results for step 2 should be stored and are used for steps 3 and 4. Further the results for step 4 are stored. 

       
