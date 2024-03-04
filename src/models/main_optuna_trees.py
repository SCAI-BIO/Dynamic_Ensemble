#!/usr/bin/ipython
import os
import warnings
import numpy as np
import torch
import pandas as pd
from parser import base_parser
from utils import define_logs
from train import Train
from test import Test
from sklearn.model_selection import KFold
import optuna
import sys
from visualizations import *
sys.path.append('../')
from data.load_data import get_iters, load_dataset, load_data_AIOLOS, load_dataset_trees
warnings.filterwarnings('ignore')
torch.autograd.set_detect_anomaly(True)
from regression import *
from tree_models import *
from arima import *

def main(trial, config,iter,region=None):
    i=iter
    region=region
    
    # here need to define the hyperparameters to be tuned in the form of 
    if config.tuning=="rf":
        config.rf_min_samples_split = trial.suggest_int('rf_min_samples_split', 0,20)
        config.rf_min_samples_leaf = trial.suggest_int('rf_min_samples_leaf',0,10 )
        
    if config.tuning=="xg":
        config.xg_learning_rate = trial.suggest_float('xg_learning_rate', 0.01,0.5)
        config.xg_max_depth = trial.suggest_int('xg_max_depth',2,12)
        config.xg_reg_lambda = trial.suggest_float('xg_reg_lambda',0,10) # L2 reg
        config.xg_reg_alpha = trial.suggest_float('xg_reg_alpha', 0,10) # L1 reg
    
   
  
    os.makedirs(config.save_path, exist_ok=True)


    train_x, train_y, = load_dataset_trees(
        config=config, region=region, iteration=i, test=False, diff=False)
    train_x_diff, train_y_diff = load_dataset_trees(
        config=config, region=region, iteration=i, test=False, diff=True)
    
    kf=KFold(n_splits=10)
    kf.get_n_splits()


    scores=[]
    for i,(train_index, test_index) in enumerate(kf.split(train_x)):
        if config.tuning=="rf":
            _,score = random_forest_regression(config, train_x_diff.iloc[train_index], 
                                               train_y_diff.iloc[train_index],
                                                train_x_diff.iloc[test_index], 
                                                train_y_diff.iloc[test_index], 
                                                train_x.iloc[test_index]) 
        
        elif config.tuning=="xg":                                  
            _,score = xgboost_regression(config, train_x_diff.iloc[train_index], 
                                         train_y_diff.iloc[train_index], 
                                         train_x_diff.iloc[test_index], 
                                         train_y_diff.iloc[test_index], 
                                         train_x.iloc[test_index])
        
        scores.append(score)   

    return(np.mean(scores))
   

if __name__ == '__main__':

    config = base_parser()
    if config.GPU != '-1':
        config.GPU_print = [int(config.GPU.split(',')[0])]
        os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU
        config.GPU = [int(i) for i in range(len(config.GPU.split(',')))]
    else:
        config.GPU = False
   
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
    
    
    save_path = os.path.join(config.save_path, config.exp_name, 
                             config.data_name.partition("_")[0],config.tuning)
    config.train_dir = os.path.join(config.train_dir, config.dataset)
    os.makedirs(save_path, exist_ok=True)
    
    regions = load_data_AIOLOS(config)[1]
    for i in range(config.iterations):
        config.from_best=False
        for region in regions:
            trial_path = save_path + '/optuna_trials_'+config.data_name.partition("_")[0]+"_"+config.tuning +"_"+ str(config.size_PW) +"/" + region.replace(".","_") 
            os.makedirs(trial_path, exist_ok=True)
            
            storage_name = 'sqlite:///%s/database%s.db'%(trial_path,str((i+1)))
          
            study_name = "optuna_"+config.data_name.partition("_")[0]+"_"+config.tuning+"_"+region.replace(".","_")+str(i+1)
            if config.tuning=="rf":
                search_space={"rf_min_samples_split":[2,4,8,16], "rf_min_samples_leaf":[1,2,4]}
                sampler=optuna.samplers.GridSampler(search_space)
            if config.tuning=="xg":
                sampler=optuna.samplers.TPESampler()
            
            study = optuna.create_study(study_name=study_name,
                                        storage=storage_name,
                                        load_if_exists=True,
                                        sampler=sampler,
                                        direction='minimize')
            
            objective_with_config = lambda trial: main(trial, config,iter=i,region=region)
            study.optimize(objective_with_config, n_trials=config.num_trials)  
            best_params = study.best_params
            best_score = study.best_value

    print(f"Best Hyperparameters: {best_params}")
    print(f"Best Score: {best_score}")

