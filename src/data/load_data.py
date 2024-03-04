import os
import pandas as pd
import torch
from .datasets import *
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_data_AIOLOS(config):
    """
    loading the datasets, defining the regions and preprocessing the columns
    """
    path = os.path.join(config.train_dir, config.data_name)
    data_complete = pd.read_csv(path)
    regions = data_complete.geography.unique()
    data_complete = data_complete.rename(columns={"total":"counts"})
    data_complete.counts = data_complete.counts + 1 # avoiding log(0)
    return data_complete, regions

def get_iters(config):
    """
    Get the number of max iterations / test windows for a dataset
    """
    dataset, _ = load_data_AIOLOS(config)
    dataset = AIOLOS_Dataset(config, dataset)
    n_wdw = config.period - config.size_SW + 1
    last_idx = len(dataset.counts_inp[1])-config.size_SW-n_wdw+1
    config.iterations = len(np.arange(0,last_idx,config.size_PW))
    return config

def standard_scaling(train_x, val_x, test_x):
    """
    Function to standard scale the data.
    """
    scaler = StandardScaler()
    num_samples, num_timesteps, num_features = train_x.shape
    train_x_reshaped = train_x.reshape((num_samples*num_timesteps,num_features))
    val_x_reshaped = val_x.reshape((val_x.shape[0]*val_x.shape[1], val_x.shape[2]))
 
    scaler.fit(train_x_reshaped)
    train_x_scaled = scaler.transform(train_x_reshaped)
    train_x_scaled = train_x_scaled.reshape((num_samples,num_timesteps,num_features))
    val_x_scaled = scaler.transform(val_x_reshaped)
    val_x_scaled = val_x_reshaped.reshape(val_x.shape[0],val_x.shape[1], val_x.shape[2])
    test_x_scaled = scaler.transform(test_x)

    return(train_x_scaled, val_x_scaled, test_x_scaled)

def time_series_split(dataset, config, iteration,test=False, torch_tensor=False,optuna=False):
    """
    Function to getting the training, validation and test data 
    """
    n_wdw = config.period - config.size_SW + 1
    last_idx = len(dataset.counts_inp[0])-config.size_SW-n_wdw+1
    iteration = iteration
    iter_temp = 0
    
    for i in np.arange(0,last_idx,config.size_SW):
        
        if iter_temp == iteration+1:
            break
        iter_temp +=1
            
        x , y = dataset.counts_inp[:,i:i+n_wdw,:] , dataset.counts_out[:,i:i+n_wdw,:]
        
        if optuna:
            train_split = int(len(x[0])*0.8)
            
            train_x , train_y = x[:,0:train_split,:] , y[:,0:train_split,:]
            val_x , val_y = x[:,train_split:,:] , y[:,train_split:,:]
            test_idx = i + n_wdw + config.size_PW - 1
            test_x , test_y = dataset.counts_inp[:,test_idx,:], dataset.counts_out[:,test_idx,:]
        else:
            train_x , train_y = x , y
            val_x , val_y = x,y # just need something here to not raise erros.
            test_idx = i + n_wdw + config.size_PW - 1
            test_x , test_y = dataset.counts_inp[:,test_idx,:], dataset.counts_out[:,test_idx,:]

    if torch_tensor: # this is only the case for the LSTM, so we can use the standardscaling here
        if config.scaled==True:
            train_x_scaled, val_x_scaled, test_x_scaled = standard_scaling(train_x, val_x, test_x)
        if config.scaled==False:
            train_x_scaled, val_x_scaled, test_x_scaled = train_x, val_x, test_x
        train_x, train_y = torch.from_numpy(train_x_scaled), torch.from_numpy(train_y)
        val_x, val_y =  torch.from_numpy(val_x_scaled) , torch.from_numpy(val_y)
        test_x, test_y = torch.from_numpy(test_x_scaled) , torch.from_numpy(test_y)
    if test:
        return(test_x, test_y) 
    else:
        return(train_x, train_y, val_x, val_y)
    
def time_series_split_trees(config, dataset, iteration, test=False, kfold=False):        
    """
    Function to getting the train and test data for the decision tree models as well as ARIMA.
    """
    iteration = iteration
    x = dataset.counts.iloc[iteration*config.size_SW:(config.period+iteration*config.size_SW),] 
    y = dataset.counts.iloc[iteration*config.size_SW+config.size_PW:
                            (config.period+iteration*config.size_SW+config.size_PW),]    
    train_x , train_y = x , y

    test_idx = iteration*config.size_SW + config.period
    test_x = dataset.counts.iloc[test_idx:test_idx+config.size_PW,]
    test_y = dataset.counts[test_idx+config.size_PW:(test_idx+config.size_PW+config.size_PW),]
    if test:
        return(test_x, test_y) 
    else:
        return(train_x, train_y)
        
    
def load_dataset_trees(config, region, iteration, test=False, diff=False):
    """
    Loading the datasets and preprocessing for the tree based models as well as ARIMA.
    If diff=True the data gets differenced and later backtransformed.
    """
    data, _ = load_data_AIOLOS(config)
    dataset = np.log(data.loc[data.geography==region,].counts+0.001)
    dataset = pd.DataFrame(dataset)
    dataset.columns = ["counts"]

    padding = pd.DataFrame([0])
    padding.columns = ["counts"]
    dataset_diff = pd.Series(np.diff(np.log(data.loc[data.geography==region,].counts+0.001)))
    dataset_diff = pd.DataFrame(dataset_diff)
    dataset_diff.columns = ["counts"]
    dataset_diff = pd.concat([padding, dataset_diff],axis=0)
    
    

    if test:
        if diff:
            test_x_diff, test_y_diff = time_series_split_trees(
                config=config, dataset=dataset_diff, iteration=iteration, test=True)
            return(test_x_diff, test_y_diff)
        else:
            test_x, test_y = time_series_split_trees(
                config=config, dataset=dataset, iteration=iteration, test=True)
            return(test_x, test_y)
    else:
        if diff:
           
            train_x_diff, train_y_diff = time_series_split_trees(
                config=config, dataset=dataset_diff, iteration=iteration)
            return(train_x_diff, train_y_diff)
        else:
        
            train_x, train_y = time_series_split_trees(
                config=config, dataset=dataset, iteration=iteration)
            return(train_x, train_y)

def load_dataset(config, iteration, test=False,optuna=False): 
    """
    Ensembling everything together to return dataloader 
    """
    data, _ = load_data_AIOLOS(config)
    dataset = AIOLOS_Dataset(config, data)    

    if test:
        x_test, y_test = time_series_split(
            dataset=dataset, config=config, iteration=iteration,
            test=True, torch_tensor=config.torch_tensor)
        
        test_data = AIOLOS_TORCH(x_test, y_test)
        dataloader_test = get_loader(config, test_data, bs=x_test.shape[0])
        return dataloader_test

    else:
        if optuna:
            x_train, y_train, x_val, y_val = time_series_split(
                dataset=dataset, config=config, iteration=iteration,
                test=config.test, torch_tensor=config.torch_tensor,optuna=True)
            
            train_data = AIOLOS_TORCH(x_train, y_train)
            val_data = AIOLOS_TORCH(x_val, y_val)
            dataloader = get_loader(config, train_data)
            dataloader_val = get_loader(config, val_data, bs=x_val.shape[0])
            return dataloader, dataloader_val
        else:            
            x_train, y_train, x_val, y_val = time_series_split(
            dataset=dataset, config=config, iteration=iteration,
            test=config.test, torch_tensor=config.torch_tensor,optuna=False)
            train_data = AIOLOS_TORCH(x_train, y_train)
            val_data = AIOLOS_TORCH(x_val, y_val)
            dataloader = get_loader(config, train_data)
            dataloader_val = get_loader(config, val_data, bs=x_val.shape[0]) 
            return dataloader, dataloader_val


def split_ensemble(config, inp, real, meta=None):
    """
    Splitting the predictions and mapes to 80/20 train test split.
    """

    nw = inp.size(0)/config.num_regions

    nw_train = int(nw * 0.8) * config.num_regions

    inp_train = inp[:nw_train]
    real_train = real[:nw_train]

    inp_test = inp[nw_train:]
    real_test = real[nw_train:]
    config.test_length = len(real_test)

    if meta is not None:
        meta_train = meta[:nw_train,...]
        meta_test = meta[nw_train:,...]
    else:
        meta_train, meta_test = None, None

    
    return (inp_train, real_train, inp_test,
            real_test, meta_train, meta_test)

def reshape_column(column, window_size, window_stride):
    """
    Function for reshaping the windows.
    """
    windows = [column[i:i+window_size] for i in range(0, len(column)-window_size+1, window_stride)]
    return windows

def load_metadata(config, meta_data_csv):
    """
    Loading meta data to be aligned with the test windows
    """
    meta_data_df = pd.read_csv(meta_data_csv)
    meta_data_df = meta_data_df.iloc[config.period+config.size_SW-config.size_SW*config.meta_lookback:,]
    reshaped_columns = [reshape_column(meta_data_df[col],config.size_SW*config.meta_lookback, config.size_SW) for col in meta_data_df.columns[2:]]
    reshaped_array = np.array(reshaped_columns).transpose(1, 2, 0)
    reshaped_array_cut = reshaped_array[:config.iterations,...]
    reshaped_array_cut_log = np.log(reshaped_array_cut+0.001) # log transformation
    meta_tensor = torch.tensor(reshaped_array_cut_log)
    meta_tensor_rep = meta_tensor.repeat(config.num_regions,1,1) 
    config.feat_meta = meta_tensor_rep.size(-1)
    meta_tensor = meta_tensor[1:] # need to skip the first window as we don't use this ensemble data not having the previous mape
    return config, meta_tensor_rep

def load_dataset_ensemble(config, predictions, scores, meta_data=None, 
                          meta=False, optuna=False, evaluation=False):
    """
    loading the whole dataset for the meta-model with and without meta dat
    """
    prediction_df = pd.read_csv(predictions)
    scores_df = pd.read_csv(scores)
    prediction_df.columns = ["geography","reg","LSTM","XG","RF","ARIMA","Mean","Median","Ens","Real"]
    scores_df.columns = ["geography","reg","LSTM","XG","RF","ARIMA","Mean","Median","Ens"]
    config.regions = prediction_df.geography.unique()
    num_regions = len(prediction_df.geography.unique())
    config.num_regions = num_regions
    prediction_df_cut = prediction_df.iloc[num_regions*config.size_PW:,:-4]
    scores_df_cut = scores_df.iloc[:-num_regions,:-3]
    real_df_cut = prediction_df.iloc[num_regions*config.size_PW:,-1]
    
    chunks = [prediction_df_cut.iloc[i:i+config.size_PW] for i in range(0, len(prediction_df_cut),config.size_PW)]
    ensemble_data = pd.DataFrame()
    for i, chunk in enumerate(chunks):
        ensemble_data = pd.concat([ensemble_data,chunk,scores_df_cut[i:i+1]])
    real_data = real_df_cut.iloc[:,]
    real_chunks =  [real_data.iloc[i:i+(config.size_PW),] for i in range(0, real_data.shape[0], (config.size_PW))]
    real_chunks_trans = [chunk.transpose() for chunk in real_chunks]
    real_transposed = pd.DataFrame()
    for chunk in real_chunks_trans:
        chunk.columns= [i for i in range(config.size_PW)]
        real_transposed = pd.concat([real_transposed, chunk.reset_index(drop=True)], ignore_index=True)
     
    ensemble_data = ensemble_data.iloc[:,1:]
    ensemble_chunks = [ensemble_data.iloc[i:i+(config.size_PW+1), :] for i in range(0, ensemble_data.shape[0], (config.size_PW+1))]
    ens_chunks_trans = [chunk.transpose() for chunk in ensemble_chunks]
    ensemble_transposed = pd.DataFrame()
    for chunk in ens_chunks_trans:
        chunk.columns = [i for i in range(config.size_PW+1)]
        ensemble_transposed = pd.concat([ensemble_transposed, chunk.reset_index(drop=True)],ignore_index=True)
    real_tensor = torch.tensor(real_transposed.values.astype('float32')).reshape(len(real_transposed)//config.size_PW, config.size_PW)
    ensemble_tensor = torch.tensor(ensemble_transposed.values.astype('float32')).view(-1, 5, config.size_PW+1)
    
    if meta:
        config, meta_data = load_metadata(config, meta_data)
        
    else:
        meta_data = None

    # shuffling the data before splitting to train and test making sure to preserve the regional order
    
    ensemble_tensor = ensemble_tensor.view(-1, num_regions, ensemble_tensor.size(1), ensemble_tensor.size(2))
    real_tensor = real_tensor.view(-1, num_regions, real_tensor.size(1))
   
    idxs_rand = torch.randperm(ensemble_tensor.size(0))
    ensemble_tensor, real_tensor = ensemble_tensor[idxs_rand], real_tensor[idxs_rand]
  
    ensemble_tensor = ensemble_tensor.reshape(-1, ensemble_tensor.size(2), ensemble_tensor.size(3))
    real_tensor = real_tensor.reshape(-1, real_tensor.size(2))
    
    if meta:
        meta_data = meta_data.view(-1, num_regions, meta_data.size(1), meta_data.size(2))
        meta_data = meta_data[idxs_rand]
        meta_data = meta_data.reshape(-1, meta_data.size(2), meta_data.size(3))

    data = split_ensemble(config, ensemble_tensor, real_tensor, meta_data)
    inp_train, real_train, inp_test, real_test, meta_train, meta_test = data

    if evaluation:
        return(scores_df.iloc[-config.test_length:,])
    else:
        if optuna:
            inp_test=inp_train[config.val_idx]
            real_test=real_train[config.val_idx]
            inp_train=inp_train[config.train_idx]
            real_train=real_train[config.train_idx]

            if meta:
                meta_test = meta_train[config.val_idx]
                meta_train = meta_train[config.train_idx]

        dataset = ENSEMBLE(inp_train, real_train, meta_train)
        dataset_test = ENSEMBLE(inp_test, real_test, meta_test)
    
        dataloader = get_loader(config, dataset)
        dataloader_test = get_loader(config, dataset_test, bs=inp_test.size(0))

        return(config, dataloader, dataloader_test)


def get_loader(config, dataset, bs=None): 
    """
    Dataloader inlcuding batching and shuffling
    """
    
    pm = True if torch.cuda.is_available() else False

    if bs is None:
        bs = config.batch_size
        shuffle = True
    else:
        shuffle = False

    dataloader = torch.utils.data.DataLoader(
        dataset = dataset,
        batch_size = bs,
        shuffle = shuffle,
        num_workers=1,
        pin_memory=pm,
        drop_last = False)
    return dataloader