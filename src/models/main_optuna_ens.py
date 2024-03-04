#!/usr/bin/ipython
import os
import warnings
import numpy as np
import torch
import pandas as pd
import optuna
from parser import base_parser
from utils import define_logs
from train_ens_model import Train
from test import Test
from sklearn.model_selection import KFold
import sys
from visualizations import *
sys.path.append('../')
from data.load_data import get_iters, load_dataset, load_data_AIOLOS, load_dataset_trees, get_loader, load_dataset_ensemble
warnings.filterwarnings('ignore')
torch.autograd.set_detect_anomaly(True)


def main(trial, config):

    config.lr =  trial.suggest_float('lr', 0.001,0.1)
    config.batch_size = trial.suggest_int('bs', 16, 128)
    config.num_hidden_layers = trial.suggest_int('hl', 0,2)
    config.hs_ens_dim = trial.suggest_int('hs',8,32)
    
    main_path = os.path.join(config.model_path, "results",
                             config.data_name.partition("_")[0])
    prediction_csv = main_path + "/predictions/preds.csv"
    scores_csv = main_path + "/scores/scores.csv"    

    config, dataloader, dataloader_val = load_dataset_ensemble(config, 
                                                               prediction_csv, 
                                                               scores_csv)
    kf=KFold(n_splits=5)
    kf.get_n_splits()

    scores=[]
    for i,(train_index, val_index) in enumerate(kf.split(dataloader.dataset[:][0])):
        config.train_idx=train_index
        config.val_idx=val_index
        config, dataloader, dataloader_val = load_dataset_ensemble(config, prediction_csv, scores_csv,optuna=True)
        config.epoch_init=1
        
        model = Train(config, dataloader, dataloader_val) 
        score = model.test(optuna=True)
        scores.append(score)
        config.from_best=False

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
    
    config.model_path = config.save_path
    config.train_dir = os.path.join(config.train_dir, config.dataset)
    config.save_path = os.path.join(config.save_path, config.exp_name, config.data_name.partition("_")[0],config.type_pred)
    config.save_path_samples = os.path.join(config.save_path, 'samples')
    config.save_path_models = os.path.join(config.save_path, 'models')
    config.save_path_losses = os.path.join(config.save_path,'losses')

    save_path=config.save_path
    save_path_samples= config.save_path_samples
    save_path_models = config.save_path_models
    save_path_losses = config.save_path_losses

    os.makedirs(config.save_path, exist_ok=True)
    os.makedirs(config.save_path_samples, exist_ok=True)
    os.makedirs(config.save_path_models, exist_ok=True)
    os.makedirs(config.save_path_losses, exist_ok=True)

    trial_path = save_path + '/optuna_trials_'+config.data_name.partition("_")[0]+"_ens_"+config.type_pred

    os.makedirs(trial_path, exist_ok=True)
    storage_name = 'sqlite:///%s/database.db'%(trial_path)
    
    study_name = "optuna_"+config.data_name.partition("_")[0]+"_ens_"+config.type_pred
    search_space={"lr": [0.001,0.005,0.01,0.05,0.1], 
                  "bs":[16,32,64,128],
                  'hl': [0,1,2],
                  'hs': [8,16,32]}
    
 
    
  
    study = optuna.create_study(study_name=study_name,
                                storage=storage_name,
                                load_if_exists=True,
                                sampler=optuna.samplers.GridSampler(search_space),
                                direction='minimize')

    objective_with_config = lambda trial: main(trial, config)
    study.optimize(objective_with_config, n_trials=config.num_trials)  

    best_params = study.best_params
    best_score = study.best_value


