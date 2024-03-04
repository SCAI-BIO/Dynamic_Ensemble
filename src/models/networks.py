import torch
import torch.nn as nn
from torch.nn import init

class LSTM_Base_Model(nn.Module):
    def __init__(self, config):
        super (LSTM_Base_Model, self).__init__()

        # input_dim here would be the number of time series (1)
        # output_dim the number of days you wanna forecast
        input_dim = config.inp_dim
        output_dim = config.size_PW
        hs_dim = config.hs_dim
        num_layers = config.num_layers
        self.lstm = nn.LSTM(input_dim, hs_dim,num_layers, batch_first=True, dropout=config.drop_rate)
        self.dense = nn.Sequential(
            nn.Linear(hs_dim, output_dim),
            nn.LeakyReLU()) 
        

    def forward(self, inp_seq):
        out, _ = self.lstm(inp_seq)
        out = out[:, -1] # last hidden state
        out = self.dense(out)
    
        return out


class MLP_ens(nn.Module):
    """
    This class refers to the meta-model without meta data. 
    It is just a shallow MLP with softmax heads.
    """
    def __init__(self, config, meta=False):
        super (MLP_ens, self).__init__()

        inp_ = config.size_PW+1+config.hs_meta if meta else config.size_PW+1
        if config.num_hidden_layers==0:   
            self.dense = nn.Linear(inp_, 1)
        elif config.num_hidden_layers==1:
            self.dense = nn.Sequential(*[
                nn.Linear(inp_,  config.hs_ens_dim),
                nn.ReLU(),
                nn.Linear(config.hs_ens_dim,1)])
        elif config.num_hidden_layers==2:
            self.dense = nn.Sequential(*[
                nn.Linear(inp_, config.hs_ens_dim),
                nn.ReLU(),
                nn.Linear(config.hs_ens_dim, config.hs_ens_dim),
                nn.ReLU(),                
                nn.Linear(config.hs_ens_dim,1)])
        self.act = nn.Softmax()

    def forward(self,inp):
        out = self.act(self.dense(inp)[..., 0])
        return out

class Meta_Model(nn.Module):
    """
    This class incoorporates the meta-model from above and includes meta data
    by encoding it using an LSTM.
    """
    def __init__(self, config):
        super (Meta_Model, self).__init__()

        self.lstm = nn.LSTM(config.feat_meta, config.hs_meta,
                            config.l_lstm_meta, batch_first=True)
        self.mlp = MLP_ens(config, meta=True)

    def forward(self, meta_data, ens_data):
        out, hs = self.lstm(meta_data)
        hs = hs[0]
        n_m = ens_data.size(1)
        hs = hs[-1].unsqueeze(1).repeat(1, n_m, 1)
        inp_ = torch.cat((ens_data, hs), -1)

        out = self.mlp(inp_)
        return out