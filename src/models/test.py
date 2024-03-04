import warnings
import os
from tqdm import tqdm
import time
import torch
import torch.nn as nn
from collections import OrderedDict
from termcolor import colored
from networks import *
from utils import *
from visualizations import *
from solver import Solver
from regression import log_linear_regression
warnings.filterwarnings('ignore')


# ==================================================================#
# ==================================================================#
class Test(Solver):
    def __init__(self, config, dataloader_test, dataloader=None, dataloader_val=None):
        super(Test, self).__init__(config, dataloader, dataloader_val, dataloader_test)

    @torch.no_grad()
    def run(self,config):
        #self.config.from_best = True ?
        #self.load_models() ? 
        self.model.eval()
        desc_bar = '[TEST]'

        progress_bar = tqdm(enumerate(self.dataloader_test),
                                unit_scale=True,
                                total=len(self.dataloader_test),
                                desc=desc_bar)

        # Testing LSTM along dataset    
        for iter, data in progress_bar:            

            inp_data = data[0].to(self.device)
            real_counts = data[1].to(self.device)
            config.droprate = 0.0
            pred_counts = self.model(inp_data)

            
            
            mape_lstm =  MAPE(torch.exp(real_counts).cpu().numpy(), torch.exp(pred_counts).cpu().numpy())


        # Log linear regression
        
        inp_data = inp_data[..., 0].cpu().numpy()
        pred_log, mape_log = log_linear_regression(inp_data, real_counts, self.config)
        
        return torch.exp(pred_counts), mape_lstm, pred_log, mape_log, real_counts


