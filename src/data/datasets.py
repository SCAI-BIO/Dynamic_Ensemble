import pandas as pd
import numpy as np
import torch

"""
constructed the class such that it takes the dataframe creates indices according to the window size and 
step size and returns a 3D array (basically a tensor) with dim BL X Window X counts
"""

class AIOLOS_Dataset():
    """
    class to create the sliding windows for the LSTM model and linear regression.
    The window can be called according to its index. Windows are created per region.
    """

    def __init__(self, config, data):
        
        max_levels = int(len(data.geography.unique()))
        max_tp = int(len(data)/max_levels)
        
        sw_counts = np.zeros((max_levels,max_tp-config.size_SW-config.size_PW+1,config.size_SW))
        sw_timepoints = np.zeros((max_levels,max_tp-config.size_SW-config.size_PW+1,config.size_SW))
        pw_counts = np.zeros((max_levels,max_tp-config.size_PW-config.size_SW+1,config.size_PW))
        pw_timepoints = np.zeros((max_levels,max_tp-config.size_PW-config.size_SW+1,config.size_PW))
        
        for geo,i in zip(data.geography.unique(),np.arange(0,max_levels+1)):
            data_reg = data.loc[data.geography==geo,]
            counts = data_reg.counts.values
        
            sw_c_ = []
            pw_c_ = []
            sw_t_ = []     # not necessarily needed but for the purpose of debugging 
            pw_t_ = []    

            for idx_i in range(1, max_tp, config.step_SW):
                if idx_i + config.size_SW > max_tp + 1:
                    break
                sw_timepoints_ = np.arange(idx_i, idx_i + config.size_SW, config.step_SW) - 1
                sw_c = counts[sw_timepoints_] 
                sw_t_.append(sw_timepoints_)
                sw_c_.append(sw_c)

            for idx_i in range(1, max_tp, config.step_SW):
                if idx_i + config.size_PW > max_tp + 1:
                    break    
                pw_timepoints_ = np.arange(idx_i, idx_i + config.size_PW, config.step_SW) - 1
                pw_c = counts[pw_timepoints_] + 1
                pw_t_.append(pw_timepoints_)
                pw_c_.append(pw_c)

            X_ = sw_c_[:-config.size_PW]
            Y_ = pw_c_[config.size_SW:]
            TX = sw_t_[:-config.size_PW]
            TY = pw_t_[config.size_SW:]
            
            sw_counts[i,:,:] = X_
            pw_counts[i,:,:] = Y_
            sw_timepoints[i,:,:] = TX
            pw_timepoints[i,:,:] = TY

        self.counts_inp = sw_counts
        self.counts_out = pw_counts
        self.time_inp = sw_timepoints
        self.time_out = pw_timepoints
        
            
            
class AIOLOS_TORCH():
    """
    For the LSTM model the arrays created by AIOLOS_Dataset are transformed
    to torch tensors.
    """
    def __init__(self, x, y):
        ld1 = x.shape[-1]
        ld2 = y.shape[-1]
        self.x = torch.log(torch.reshape(x, (-1, ld1)).unsqueeze(-1) + 0.001)
        self.y = torch.log(torch.reshape(y, (-1, ld2)) + 0.001)
       

    def __getitem__(self, idx):

        x = self.x[idx, ...].float()
        y = self.y[idx, ...].float()
        return x, y

    def __len__(self):
        return len(self.x)


class ENSEMBLE():
    """
    Class for transforming the arrays for meta-model to torch tensors.
    """
    def __init__(self, x, y, metadata=None):
        self.x = torch.log(x + 0.001)
        self.y = torch.log(y + 0.001)

        if metadata is not None:
            self.meta = True
            self.metadata = metadata
        else:
            self.meta = False

    def __getitem__(self, idx):

        x = self.x[idx, ...].float()
        y = self.y[idx, ...].float()
        if self.meta:
            return x, y, self.metadata[idx, ...].float()
        else:
            return x, y

    def __len__(self):
        return len(self.x)
