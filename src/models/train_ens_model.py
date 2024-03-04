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
from solver_ens_model import Solver
warnings.filterwarnings('ignore')


# ==================================================================#
# ==================================================================#
class Train(Solver):
    def __init__(self, config, dataloader, dataloader_test):
        super(Train, self).__init__(config, dataloader, dataloader_test)

        if config.mode == 'test':
            self.test()
        else:
            self.run()
        print('Training has finished')

    def run(self):

        global_steps = 0
        total_time = time.time()
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

            epoch_time_init = time.time()
            for iter, data in progress_bar:
                global_steps += 1

                inp_data = data[0].to(self.device)
                real_data = data[1].to(self.device)

                if global_steps == 1:
                    self.data_fix = [inp_data.clone(), real_data.clone()]
                
                soft_output = self.model(inp_data)
                

                if self.config.type_pred == 'selection':
                    _, idxs = soft_output.max(1)
                    inp_data = inp_data.requires_grad_(True)
                    inp_ = []
                    for i, idx in enumerate(idxs):
                        inp_.append(inp_data[i, idx, :-1])
                    preds = torch.stack(inp_)
                else:
                    inp_data = inp_data[..., :-1]
                    preds = (soft_output.unsqueeze(-1) * inp_data).sum(1)
                
                loss = (((preds - real_data)**2)/real_data).sum(-1).mean()
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

            if avg_loss < self.best_loss:
                self.save(epoch, avg_loss, best=True)
                self.best_loss = avg_loss
                no_improvement = 0
            else:
                no_improvement += 1

            if no_improvement > self.config.patience:
                print('Early Stoped in epoch ', epoch)
                break

        self.save(epoch, avg_loss)

    @torch.no_grad()
    def test(self,optuna=False):

        self.config.from_best = True
        self.load_models()
        self.set_nets_eval()

        desc_bar = '[TEST:]'

        progress_bar = tqdm(enumerate(self.dataloader_test),
                            unit_scale=True,
                            total=len(self.dataloader_test),
                            desc=desc_bar)

        for iter, data in progress_bar:

            inp_data = data[0].to(self.device)
            real_data = data[1].to(self.device)

            soft_output = self.model(inp_data)
            inp_data = inp_data[..., :-1]

            if self.config.type_pred == 'selection':
                _, idxs = soft_output.max(1)
                inp_ = []
                for i, idx in enumerate(idxs):
                    inp_.append(inp_data[i, idx, :])
                preds = torch.stack(inp_)
            else:
                preds = (soft_output.unsqueeze(-1) * inp_data).sum(1)
            
            mape = MAPE(torch.exp(real_data).cpu().numpy(), torch.exp(preds).cpu().numpy())
            
            print(mape.mean())

        if optuna:
            return(mape.mean())
        else:
            return(mape)