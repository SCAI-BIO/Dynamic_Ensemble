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
warnings.filterwarnings('ignore')


# ==================================================================#
# ==================================================================#
class Train(Solver):
    def __init__(self, config, dataloader, dataloader_val, dataloader_test=None):
        super(Train, self).__init__(config, dataloader, dataloader_val, dataloader_test)

        self.run()
        print('Training has finished')

    def run(self):

        global_steps = 0
        no_improvement = 0

        if self.config.epoch_init > 1:
            self.config.epoch_init += 1

        for epoch in range(self.config.epoch_init, self.config.num_epochs + 1):

            avg_loss = 0
            desc_bar = '[Iter: %d] Epoch: %d/%d' % (
                global_steps, epoch, self.config.num_epochs)

            progress_bar = tqdm(enumerate(self.dataloader),
                                unit_scale=True,
                                total=len(self.dataloader),
                                desc=desc_bar)

            for iter, data in progress_bar:
                global_steps += 1

                inp_data = data[0].to(self.device)
                real_counts = data[1].to(self.device)

                if global_steps == 1:
                    self.data_fix = [inp_data.clone(), real_counts.clone()]
                
                pred_counts = self.model(inp_data)
                
                loss = ((pred_counts - real_counts)**2).sum(-1).mean() 
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                avg_loss += loss.item()

                if (iter + 1) % self.config.print_freq == 0 or (iter + 1) == len(self.dataloader):
                    losses = OrderedDict()

                    losses['Total'] = loss.item()
                    losses['Average'] = avg_loss / (iter + 1)
                    progress_bar.set_postfix(**losses)

            avg_loss = avg_loss / (iter + 1)

            if (epoch) % self.config.save_freq == 0:
                self.save(epoch, avg_loss)


            if avg_loss < self.best_loss:
                self.save(epoch, avg_loss, best=True)
                self.best_loss = avg_loss
                no_improvement = 0
            else:
                no_improvement += 1

            if no_improvement > self.config.patience:
                print('Early Stoped in epoch ', epoch)
                break

        