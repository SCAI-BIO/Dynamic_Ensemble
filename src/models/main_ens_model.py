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
import sys
from visualizations import *
sys.path.append('../')
from data.load_data import get_iters, load_dataset, load_data_AIOLOS, load_dataset_trees, load_dataset_ensemble
warnings.filterwarnings('ignore')
torch.autograd.set_detect_anomaly(True)
from regression import *
from tree_models import *
from arima import *

def main(config):

    main_path = os.path.join(config.model_path, "results",
                             config.data_name.partition("_")[0])
    prediction_csv = main_path + "/predictions/preds.csv"
    scores_csv = main_path + "/scores/scores.csv"
    config, dataloader, dataloader_test = load_dataset_ensemble(config, 
                                                                prediction_csv, scores_csv)
    config.type_pred="selection"
    if config.type_pred == "selection":
        trial_path_selection = os.path.join(config.model_path, "optuna_ens", 
                                            config.data_name.partition("_")[0], 
                                            config.type_pred, "optuna_trials_"+
                                            config.data_name.partition("_")[0]+
                                            "_ens_"+config.type_pred)
        storage_name = 'sqlite:///%s/database.db'%(trial_path_selection)
        study_name = "optuna_"+config.data_name.partition("_")[0]+"_ens_"+config.type_pred
        study = optuna.load_study(study_name=study_name, storage=storage_name)
        best_params=study.best_params
       
        config.lr = best_params['lr']
        config.batch_size = best_params['bs']
        config.num_hidden_layers = best_params['hl']
        config.hs_ens_dim = best_params['hs']

        model = Train(config, dataloader, dataloader_test) 
        scores_selection = model.test()
    config.from_best=False    
    config.epoch_init=1
    config.type_pred="stacking"
    if config.type_pred == "stacking":
        trial_path_stacking = os.path.join(config.model_path, "optuna_ens", 
                                           config.data_name.partition("_")[0], 
                                           config.type_pred, "optuna_trials_"+
                                           config.data_name.partition("_")[0]+
                                           "_ens_"+config.type_pred)
        storage_name = 'sqlite:///%s/database.db'%(trial_path_stacking)
        study_name = "optuna_"+config.data_name.partition("_")[0]+"_ens_"+config.type_pred
        study = optuna.load_study(study_name=study_name, storage=storage_name)
        best_params=study.best_params

        config.lr = best_params['lr']
        config.batch_size = best_params['bs']
        config.num_hidden_layers = best_params['hl']
        config.hs_ens_dim = best_params['hs']

        model = Train(config, dataloader, dataloader_test) 
        scores_stacking = model.test()

    scores_test_models = load_dataset_ensemble(config,prediction_csv,
                                               scores_csv,meta=False,optuna=False,
                                               evaluation=True)
    scores_test_models.index = scores_test_models.iloc[:,0]
    
    meta_df = pd.DataFrame({"selection":scores_selection, "stacking":scores_stacking})
    meta_df.index = scores_test_models.index
    
    scores_df = pd.concat([scores_test_models, meta_df],axis=1)
    meta_mean_df = scores_df.groupby(scores_df.index).mean().round(2)
    meta_median_df = scores_df.groupby(meta_df.index).median().round(2)
    meta_std_df = scores_df.groupby(scores_df.index).std().round(2)
    column_pairs = [("reg_x","reg_y"),("LSTM_x","LSTM_y"),
                    ("XG_x","XG_y"),("RF_x","RF_y"),("ARIMA_x","ARIMA_y"),
                    ("Mean_x","Mean_y"),("Median_x","Median_y"),
                    ("Ens_x","Ens_y"),("selection_x","selection_y"),
                    ("stacking_x","stacking_y")]

    meta_mean_df.reset_index(inplace=True)
    meta_std_df.reset_index(inplace=True)
    meta_mean_df["geography"]=meta_median_df.index
    meta_std_df["geography"]=meta_median_df.index
   
    merged_data = pd.merge(meta_mean_df, meta_std_df,how='inner', 
                           left_on="geography", right_on="geography")
    combined_df=pd.DataFrame()
    for column1_csv1, column1_csv2 in column_pairs:
        combined_df[f'combined_{column1_csv1}'] = merged_data[column1_csv1].astype(str) + ' (' + merged_data[column1_csv2].astype(str) + ')'
    combined_df.insert(0,"geography",meta_mean_df.geography)
    combined_df.columns = meta_mean_df.columns
    
    save_path_output = os.path.join(config.save_path, "scores")
    os.makedirs(save_path_output, exist_ok=True)
    scores_df.to_csv(save_path_output+"/scores.csv",index=False)
    meta_mean_df.to_csv(save_path_output+"/scores_ens_mean.csv",index=False)
    meta_median_df.to_csv(save_path_output+"/scores_ens_median.csv")
    meta_std_df.to_csv(save_path_output+"/scores_ens_std.csv", index=False)
    combined_df.to_csv(save_path_output+"/combined_scores.csv",index=False)
     

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
    config.model_path = config.save_path
    config.save_path = os.path.join(config.save_path, config.exp_name,config.data_name.partition("_")[0])
    config.save_path_samples = os.path.join(config.save_path, 'samples')
    config.save_path_models = os.path.join(config.save_path, 'models')
    config.save_path_losses = os.path.join(config.save_path, 'losses')

    os.makedirs(config.save_path, exist_ok=True)
    os.makedirs(config.save_path_samples, exist_ok=True)
    os.makedirs(config.save_path_models, exist_ok=True)
    os.makedirs(config.save_path_losses, exist_ok=True)

    config.save_path_losses = os.path.join(config.save_path_losses, 'losses.txt')
    main(config)
