#!/usr/bin/ipython
import os
import warnings
import numpy as np
import torch
import pandas as pd
import optuna
from parser import base_parser
from utils import define_logs
from train import Train
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

def main(config):

    if config.iterations == 0:
        config = get_iters(config)
    
    model_path = os.path.dirname(config.save_path)
    config.save_path = os.path.join(config.save_path, config.data_name.partition("_")[0])
    config.save_path_samples = os.path.join(config.save_path, 'samples')
    config.save_path_models = os.path.join(config.save_path, 'models')
    config.save_path_losses = os.path.join(config.save_path,'losses')
    save_path_samples = config.save_path_samples
    save_path_models = config.save_path_models
    save_path_losses = config.save_path_losses
    
    config.save_path_predictions = config.save_path + "/predictions/"
    config.save_path_scores = config.save_path + "/scores/"
    os.makedirs(config.save_path_predictions, exist_ok=True)
    os.makedirs(config.save_path_scores, exist_ok=True)
    regions = load_data_AIOLOS(config)[1]
    config.regions = regions
    #import ipdb; ipdb.set_trace()
    total_preds_lstm, total_mape_lstm = pd.DataFrame(), pd.DataFrame()
    total_preds_log, total_mape_log = pd.DataFrame(), pd.DataFrame()
    total_preds_rf, total_mape_rf = pd.DataFrame(), pd.DataFrame()
    total_preds_xg, total_mape_xg = pd.DataFrame(), pd.DataFrame()
    total_preds_ar, total_mape_ar = pd.DataFrame(), pd.DataFrame()
    total_mape_ens = pd.DataFrame()
    total_pred_df, total_score_df = pd.DataFrame(), pd.DataFrame()
    
    print('Mode:', config.mode)
    #import ipdb; ipdb.set_trace()
    for i in range(config.iterations):
        selection_df = pd.DataFrame()
        config.from_best=False
        config.save_path_samples = os.path.join(save_path_samples, 'Iter_%d'%(i+1))
        config.save_path_models = os.path.join(save_path_models, 'Iter_%d'%(i+1))
        config.save_path_losses = os.path.join(save_path_losses, 'Iter_%d'%(i+1))
        

        os.makedirs(config.save_path, exist_ok=True)
        os.makedirs(config.save_path_samples, exist_ok=True)
        os.makedirs(config.save_path_models, exist_ok=True)
        os.makedirs(config.save_path_losses, exist_ok=True)
        config.save_path_losses = os.path.join(config.save_path_losses, 'losses.txt')
        define_logs(config)
        dataloader, dataloader_val = load_dataset(config, i, optuna=False)

        
        #import ipdb; ipdb.set_trace()
        trial_path_lstm = model_path  +'/optuna_lstm/' + config.data_name.partition("_")[0]+'/optuna_trials_'+config.data_name.partition("_")[0]+"_LSTM"
        storage_name_lstm = 'sqlite:///%s/database%s.db'%(trial_path_lstm,str((i+1)))
        study_name_lstm = "optuna_"+config.data_name.partition("_")[0]+"_LSTM"+str(i+1)
        study_lstm = optuna.load_study(study_name=study_name_lstm, storage=storage_name_lstm)
        best_params_lstm=study_lstm.best_params
        config.lr = best_params_lstm["lr"]
        config.drop_rate = best_params_lstm["drop_rate"]
        config.hs_dim = best_params_lstm["hs_dim"]
        config.num_layers = best_params_lstm["num_layers"]
        



        Train(config, dataloader, dataloader_val)
        config.from_best=True
        config.scaled=False
        dataloader_test = load_dataset(config, i, test=True)
        test_process = Test(config, dataloader_test)
        pred_lstm, mape_lstm,_,_, real_counts = test_process.run(config)
        config.scaled=False
        dataloader_test = load_dataset(config, i, test=True)
        test_process = Test(config, dataloader_test)
        _, _,pred_log,mape_log, real_counts = test_process.run(config)
        #import ipdb; ipdb.set_trace()

        

        predictions_rf, predictions_xg, predictions_ar= [],[],[]
        scores_rf, scores_xg, scores_ar = [],[],[]
        for region in regions:
            train_x, train_y, = load_dataset_trees(
                config=config, region=region, iteration=i, test=False, diff=False)
            test_x, test_y = load_dataset_trees(
                config=config, region=region, iteration=i, test=True, diff=False)
            train_x_diff, train_y_diff = load_dataset_trees(
                config=config, region=region, iteration=i, test=False, diff=True)
            test_x_diff, test_y_diff = load_dataset_trees(
                config=config, region=region, iteration=i, test=True, diff=True)


            trial_path_xg = model_path  + '/optuna_trees/'+config.data_name.partition("_")[0]+"/xg"+ '/optuna_trials_'+config.data_name.partition("_")[0]+"_xg_" +str(config.size_PW)+ "/" + region.replace(".","_") 
            storage_name_xg = 'sqlite:///%s/database%s.db'%(trial_path_xg,str((i+1)))
            study_name_xg = "optuna_"+config.data_name.partition("_")[0]+"_xg_"+region.replace(".","_")+str(i+1)
            study_xg = optuna.load_study(study_name=study_name_xg, storage=storage_name_xg)
            best_params_xg=study_xg.best_params
            config.xg_learning_rate = best_params_xg["xg_learning_rate"]
            config.xg_max_depth = best_params_xg["xg_max_depth"]
            config.xg_reg_lambda = best_params_xg["xg_reg_lambda"]
            config.xg_reg_alpha = best_params_xg["xg_reg_alpha"]
            trial_path_rf = model_path  + '/optuna_trees/'+config.data_name.partition("_")[0]+"/rf" '/optuna_trials_'+config.data_name.partition("_")[0]+"_rf_"+str(config.size_PW)+ "/"  + region.replace(".","_") 
            storage_name_rf = 'sqlite:///%s/database%s.db'%(trial_path_rf,str((i+1)))
            study_name_rf = "optuna_"+config.data_name.partition("_")[0]+"_rf_"+region.replace(".","_")+str(i+1)
            study_rf = optuna.load_study(study_name=study_name_rf, storage=storage_name_rf)
            best_params_rf=study_rf.best_params
            config.rf_min_samples_split = best_params_rf["rf_min_samples_split"]
            config.rf_min_samples_leaf = best_params_rf["rf_min_samples_leaf"]
            
            pred_rf, score_rf = random_forest_regression(config,
                                                         train_x_diff,train_y_diff,
                                                         test_x_diff,test_y_diff,
                                                         train_y, diff=True)
            pred_xg, score_xg = xgboost_regression(config,
                                                   train_x_diff,train_y_diff,
                                                   test_x_diff,test_y_diff,
                                                   train_y, diff=True)
            train_data_ar = pd.concat([train_x, test_x],axis=0, ignore_index=True)
            pred_ar, score_ar = arima_prediction(config,train_data_ar, test_y)
           

            predictions_rf.append(pred_rf)
            scores_rf.append(score_rf)
            predictions_xg.append(pred_xg)
            scores_xg.append(score_xg)
            predictions_ar.append(pred_ar)
            scores_ar.append(score_ar)


                                                            

        
        
        total_preds_xg = pd.DataFrame(np.array(predictions_xg).flatten())
        total_mape_xg = pd.DataFrame(np.array(scores_xg))
        total_preds_rf = pd.DataFrame(np.array(predictions_rf).flatten())
        total_mape_rf = pd.DataFrame(np.array(scores_rf))
        total_preds_ar = pd.DataFrame(np.array(predictions_ar).flatten())
        total_mape_ar = pd.DataFrame(np.array(scores_ar))
        total_preds_lstm =pd.DataFrame(pred_lstm.flatten())
        total_mape_lstm = pd.DataFrame(mape_lstm)
        total_preds_log = pd.DataFrame(pred_log.flatten())
        total_mape_log = pd.DataFrame(mape_log)
        total_real_counts = np.exp(pd.DataFrame(real_counts.flatten()))

        ensemble_stack_mape = pd.DataFrame([mape_log.flatten(),
                                            mape_lstm.flatten(), 
                                            np.array(scores_xg).flatten(), 
                                            np.array(scores_rf).flatten(),
                                            np.array(scores_ar).flatten()]).T
        ensemble_stack_pred = [pred_log, pred_lstm.cpu().numpy(), 
                               np.array(predictions_xg), 
                               np.array(predictions_rf), 
                               np.array(predictions_ar)]
        ensemble_stack_pred_arr = np.array(ensemble_stack_pred)
  
        ensemble_stack_mean = np.mean(ensemble_stack_pred_arr, axis=0, keepdims=True)
        ensemble_stack_median = np.median(ensemble_stack_pred_arr, axis=0, keepdims=True)
        real_counts_arr = total_real_counts.values.reshape(1,len(regions),config.size_PW)
        scores_ensemble_mean = np.mean(np.abs((ensemble_stack_mean-real_counts_arr)/real_counts_arr),2)*100
        scores_ensemble_median = np.mean(np.abs((ensemble_stack_median-real_counts_arr)/real_counts_arr),2)*100
        total_preds_mean = pd.DataFrame(ensemble_stack_mean.flatten())
        total_mape_mean = pd.DataFrame(scores_ensemble_mean.flatten())
        total_preds_median = pd.DataFrame(ensemble_stack_median.flatten())
        total_mape_median = pd.DataFrame(scores_ensemble_median.flatten())
        config.epoch_init=1
        
        if i == 0:
            preds_ens = pred_log
            mape_ens = mape_log
            total_preds_ens = pd.DataFrame(preds_ens.flatten())
            total_mape_ens = pd.DataFrame(mape_ens)
        else: 
            preds_ens_idx = total_score_df.tail(len(regions)).iloc[:,:-3].idxmin(axis=1)
            total_preds_ens = pd.DataFrame()
            for k,j in zip(preds_ens_idx.values, np.arange(0,len(regions))):
                total_preds_ens = pd.concat([total_preds_ens, pd.DataFrame(ensemble_stack_pred[k][j].flatten())],axis=0)
            selection_df  = pd.concat([selection_df, preds_ens_idx],axis=1)
            mape_ens = np.diag(ensemble_stack_mape[preds_ens_idx])
            
            total_mape_ens = pd.DataFrame(mape_ens)
        total_real_counts.index = np.repeat(regions,config.size_PW)
        total_preds_log.index, total_preds_lstm.index, total_preds_ens.index = np.repeat(regions,config.size_PW),np.repeat(regions,config.size_PW),np.repeat(regions,config.size_PW)
        total_preds_rf.index, total_preds_xg.index, total_preds_ar.index = np.repeat(regions,config.size_PW),np.repeat(regions,config.size_PW),np.repeat(regions,config.size_PW) 
        total_mape_log.index, total_mape_lstm.index, total_mape_ens.index = regions, regions, regions
        total_mape_rf.index, total_mape_xg.index, total_mape_ar.index = regions,regions, regions
        total_preds_mean.index, total_preds_median.index = np.repeat(regions,config.size_PW), np.repeat(regions,config.size_PW)
        total_mape_mean.index, total_mape_median.index = regions, regions

        pred_df = pd.concat([total_preds_log, total_preds_lstm, total_preds_xg, total_preds_rf,total_preds_ar,total_preds_mean, total_preds_median,total_preds_ens,total_real_counts], axis=1)
        score_df = pd.concat([total_mape_log, total_mape_lstm, total_mape_xg, total_mape_rf, total_mape_ar, total_mape_mean, total_mape_median, total_mape_ens], axis=1)
        score_df.columns = [0,1,2,3,4,5,6,7]

        
        total_pred_df = pd.concat([total_pred_df, pred_df], axis=0)
        total_score_df = pd.concat([total_score_df, score_df], axis=0) 
    
        filename_mean_score = config.save_path_scores + 'regional_mean_scores.csv'
        regional_mean = total_score_df.groupby(total_score_df.index).mean().round(2)
        regional_mean.to_csv(filename_mean_score)   
        filename_median_score = config.save_path_scores +'regional_median_scores.csv'
        regional_median = total_score_df.groupby(total_score_df.index).median().round(2)
        regional_median.to_csv(filename_median_score)
        filename_std_score = config.save_path_scores +'regional_std_scores.csv'
        regional_std = total_score_df.groupby(total_score_df.index).std().round(2)
        regional_std.to_csv(filename_std_score)
        #import ipdb; ipdb.set_trace()
        filename_scores = config.save_path_scores  + 'scores.csv'
        filename_preds = config.save_path_predictions + 'preds.csv'
        filename_selection = config.save_path_scores + 'selection.csv'
        
        
        if i <1:
            selections = pd.DataFrame(0, index=regions, columns=[0,1,2,3,4]) 
        if i >= 1:
            selection = selection_df.apply(pd.Series.value_counts, axis=1).fillna(0)
            selections = selections.add(selection, fill_value=0)    
        selections.to_csv(filename_selection)
        total_pred_df.to_csv(filename_preds)
        total_score_df.to_csv(filename_scores)
    
    plotting_barplots(selections,config.save_path_scores)
    regional_mean.columns = ["LR","LSTM","XG","RF","ARIMA","Mean","Median","Ens"]
    regional_std.columns = ["LR","LSTM","XG","RF","ARIMA","Mean","Median","Ens"]
    column_pairs = [("LR_x","LR_y"),("LSTM_x","LSTM_y"),
                    ("XG_x","XG_y"),("RF_x","RF_y"),("ARIMA_x","ARIMA_y"),
                    ("Mean_x","Mean_y"),("Median_x","Median_y"),("Ens_x","Ens_y")]

    regional_mean["geography"]=regional_mean.index
    regional_std["geography"]=regional_mean.index

    merged_data = pd.merge(regional_mean, regional_std,how='inner', left_on="geography", right_on="geography")
    combined_df=pd.DataFrame()
    
    for column1_csv1, column1_csv2 in column_pairs:
        combined_df[f'combined_{column1_csv1}'] = merged_data[column1_csv1].astype(str) + ' (' + merged_data[column1_csv2].astype(str) + ')'
    
    combined_df.index= regional_mean.index
    combined_df.columns = regional_mean.columns[:-1]
    filename_combined = config.save_path_scores  + 'combined_scores.csv'
    combined_df.to_csv(filename_combined)
    print(filename_combined)
   

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

    config.save_path = os.path.join(config.save_path, config.exp_name)
    config.save_path_samples = os.path.join(config.save_path, 'samples')
    config.save_path_models = os.path.join(config.save_path, 'models')
    config.save_path_losses = os.path.join(config.save_path, 'losses')

    os.makedirs(config.save_path, exist_ok=True)
    os.makedirs(config.save_path_samples, exist_ok=True)
    os.makedirs(config.save_path_models, exist_ok=True)
    os.makedirs(config.save_path_losses, exist_ok=True)
    main(config)
