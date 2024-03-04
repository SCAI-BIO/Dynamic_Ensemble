#!/usr/bin/ipython
import os
import warnings
import numpy as np
import torch
import pandas as pd
import optuna
from parser import base_parser
from utils import define_logs
from train_optuna import Train
from test import Test
import sys
from visualizations import *
sys.path.append('../')
from data.load_data import get_iters, load_dataset, load_data_AIOLOS, load_dataset_trees
warnings.filterwarnings('ignore')
torch.autograd.set_detect_anomaly(True)
from regression import *
from tree_models import *
from arima import *

def main(trial, config,iter):
    i=iter

    config.hs_dim =  trial.suggest_categorical('hs_dim', [16,32,64,128])
    config.lr =  trial.suggest_categorical('lr', [0.001,0.005,0.01,0.05,0.1]) 
    config.num_layers = trial.suggest_categorical('num_layers',[1,2,3]) 
    config.drop_rate = trial.suggest_categorical('drop_rate',[0.0,0.2,0.5])
    config.batch_size = trial.suggest_categorical('bs', [16,32,64,128])
        

    if config.iterations == 0:
        config = get_iters(config)
    
    model = Train(config, dataloader, dataloader_val) 
    score = model.val()
    config.from_best=False
    config.epoch_init=1
    return(score)

     

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
    
    
    config.train_dir = os.path.join(config.train_dir, config.dataset)
    config.save_path = os.path.join(config.save_path, config.exp_name, 
                                    config.data_name.partition("_")[0])
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
    for i in range(config.iteration_end-10, config.iteration_end):
        # -10 to run in arrays of size 10
        config.from_best=False
        config.save_path_samples = os.path.join(save_path_samples, 'Iter_%d'%(i+1))
        config.save_path_models = os.path.join(save_path_models, 'Iter_%d'%(i+1))
        config.save_path_losses = os.path.join(save_path_losses, 'Iter_%d'%(i+1))
        os.makedirs(config.save_path, exist_ok=True)
        os.makedirs(config.save_path_samples, exist_ok=True)
        os.makedirs(config.save_path_models, exist_ok=True)
        os.makedirs(config.save_path_losses, exist_ok=True)
        
        dataloader, dataloader_val = load_dataset(config, i,optuna=True)
        

        trial_path = save_path + '/optuna_trials_'+config.data_name.partition("_")[0]+"_LSTM"

        os.makedirs(trial_path, exist_ok=True)
        storage_name = 'sqlite:///%s/database%s.db'%(trial_path,str((i+1)))
        
        study_name = "optuna_"+config.data_name.partition("_")[0]+"_LSTM"+str(i+1)

        study = optuna.create_study(study_name=study_name,
                                    storage=storage_name,
                                    load_if_exists=True,
                                    sampler=optuna.samplers.TPESampler(),
                                    direction='minimize')
        
        objective_with_config = lambda trial: main(trial, config,iter=i)
        study.optimize(objective_with_config, n_trials=config.num_trials)  

        best_params = study.best_params
        best_score = study.best_value
