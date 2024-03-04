import warnings
import os
import torch
from termcolor import colored
from networks import *
from utils import *
from visualizations import *
warnings.filterwarnings('ignore')


# ==================================================================#
# ==================================================================#
class Solver(object):
    def __init__(self, config, dataloader, dataloader_val, dataloader_test):

        self.config = config
        self.device = torch.device('cuda:{}'.format(config.GPU[0])) if config.GPU else torch.device('cpu')
        self.dataloader = dataloader
        self.dataloader_val = dataloader_val
        self.dataloader_test = dataloader_test

        self.build_model()

    def build_model(self):

        self.model = LSTM_Base_Model(self.config).to(self.device)
        print('Models are build and have set to device')

        params = list(self.model.parameters()) 
        self.optimizer = torch.optim.Adam(params, self.config.lr)

        if self.config.epoch_init != 1 or self.config.from_best:
            self.load_models()
        else:
            self.best_loss = 10e15

    def load_models(self):

        if self.config.from_best:
            weights = torch.load(os.path.join(
                self.config.save_path_models, 'Best.pth'),
                map_location=self.device)
            self.config.epoch_init = weights['Epoch']
            epoch = self.config.epoch_init
        else:
            epoch = self.config.epoch_init
            weights = torch.load(os.path.join(
                self.config.save_path_models, 'Ckpt_%d.pth'%(epoch)),
                map_location=self.device)

        self.best_loss = weights['Loss']
        self.model.load_state_dict(weights['Model'])

        if 'train' in self.config.mode:
            self.optimizer.load_state_dict(weights['Opt'])

        print('Models have loaded from epoch:', epoch)

    def save(self, epoch, loss, best=False):

        weights = {}
        weights['Model'] = self.model.state_dict()
        weights['Opt'] = self.optimizer.state_dict()

        weights['Loss'] = loss
        if best:
           weights['Epoch'] = epoch
           torch.save(weights, 
               os.path.join(self.config.save_path_models, 'Best.pth'))
        else:
           torch.save(weights, 
               os.path.join(self.config.save_path_models, 'Ckpt_%d.pth'%(epoch)))

        print('Models have been saved')
