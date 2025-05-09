"""
This function is adapted from [OmniAnomaly] by [TsingHuasuya et al.]
Original source: [https://github.com/NetManAIOps/OmniAnomaly]
"""

from __future__ import division
from __future__ import print_function

import numpy as np
import math
import torch
import torch.nn.functional as F
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from torch import nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
import tqdm

from .base import BaseDetector
from ..utils.dataset import ReconstructDataset
from ..utils.torch_utility import EarlyStoppingTorch, get_gpu
from scipy.signal import savgol_filter

def create_sliding_window(X, window, step=1, shuffle=True):
    total_length = X.shape[0]
    W = window
    step = step
    num_samples = (total_length - W ) // step + 1      

    # 2 coordinates for the sliding window per window
    X_train_indices = []
    for i in range(total_length):
        X_train_indices.append( np.arange(i, i + W) )
    
    X_train_indices = np.array(X_train_indices)
    
    if shuffle:
        indices = np.random.permutation(num_samples)
        X_train_indices = X_train_indices[indices]
    return X_train_indices

class OmniAnomalyModel(nn.Module):
    def __init__(self, feats, device):
        super(OmniAnomalyModel, self).__init__()
        self.name = 'OmniAnomaly'
        self.device = device
        self.lr = 0.002
        self.beta = 0.01
        self.n_feats = feats
        self.n_hidden = 32
        self.n_latent = 8
        self.lstm = nn.GRU(feats, self.n_hidden, 2)
        self.encoder = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
            nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
            # nn.Flatten(),
            nn.Linear(self.n_hidden, 2*self.n_latent)
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.n_latent, self.n_hidden), nn.PReLU(),
            nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
            nn.Linear(self.n_hidden, self.n_feats), nn.Sigmoid(),
        )

    def forward(self, x, hidden = None):
        bs = x.shape[0]
        win = x.shape[1]

        # hidden = torch.rand(2, bs, self.n_hidden, dtype=torch.float64) if hidden is not None else hidden
        hidden = torch.rand(2, bs, self.n_hidden).to(self.device) if hidden is not None else hidden

        out, hidden = self.lstm(x.view(-1, bs, self.n_feats), hidden)

        # print('out: ', out.shape)       # (L, bs, n_hidden)
        # print('hidden: ', hidden.shape) # (2, bs, n_hidden)

        ## Encode
        x = self.encoder(out)
        mu, logvar = torch.split(x, [self.n_latent, self.n_latent], dim=-1)
        ## Reparameterization trick
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        x = mu + eps*std
        ## Decoder
        x = self.decoder(x)             # (L, bs, n_feats)
        return x.reshape(bs, win*self.n_feats), mu.reshape(bs, win*self.n_latent), logvar.reshape(bs, win*self.n_latent), hidden


class OmniAnomaly(BaseDetector):
    def __init__(self,
                 win_size = 5,
                 feats = 1,
                 batch_size = 128,
                 epochs = 50,
                 patience = 3,
                 lr = 0.002,
                 validation_size=0.2
                 ):
        super().__init__()

        self.__anomaly_score = None

        self.cuda = True
        self.device = get_gpu(self.cuda)

        self.win_size = win_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.feats = feats
        self.validation_size = validation_size
        self.lr = lr

        self.model = OmniAnomalyModel(feats=self.feats, device=self.device).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=1e-5
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 5, 0.9)
        self.criterion = nn.MSELoss(reduction = 'none')

        self.early_stopping = EarlyStoppingTorch(None, patience=patience)

    def fit(self, data):
        tsTrain = data

        tsTrain = torch.tensor(tsTrain, dtype=torch.float32).to(self.device)
        ts_changes = -tsTrain.clone().to(self.device)  

        X_train_indices = create_sliding_window(tsTrain, self.win_size, step=1)
        
        
        ts_changes.requires_grad = True

        self.optimizer = torch.optim.AdamW(
            [
                {'params': self.model.parameters(), 'lr': self.lr, 'weight_decay': 1e-5},
                {'params': [ts_changes], 'lr': 0.001, 'weight_decay': 0}
                
            ]

        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 5, 0.9)

        
        for epoch in range(1, self.epochs + 1):
            self.model.train(mode=True)
            n = epoch + 1
            avg_loss = 0
            
            for i in range(0, X_train_indices.shape[0], self.batch_size):
                x_batch_indices = X_train_indices[i:i + self.batch_size] 
                d = tsTrain[x_batch_indices, :]
                x_changes = ts_changes[x_batch_indices, :]
                d = d + x_changes
                y_pred, mu, logvar, hidden = self.model(d, hidden if epoch>1 else None)
                d = d.view(-1, self.feats*self.win_size) 
                MSE = torch.mean(self.criterion(y_pred, d), axis=-1)
                KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
                loss = torch.mean(MSE + self.model.beta * KLD) 

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                avg_loss += loss.cpu().item()
            print(f"Training Epoch [{epoch}/{self.epochs}] Loss {avg_loss / epoch}")


            self.scheduler.step()

            self.optimizer.zero_grad()    
            loss = torch.norm(ts_changes, p=1) 
            loss.backward()
            self.optimizer.step()

        scores = np.abs(ts_changes.detach().cpu().permute(1,0).numpy()[:,:])

        scores_sav = savgol_filter(scores, self.win_size, 2)

        scores += scores_sav

        scores = np.abs(scores - np.mean(scores, axis=1, keepdims=True)) / (np.std(scores, axis=1, keepdims=True) + 1e-8)

        scores = scores.mean(axis=0)

        return scores

    def decision_function(self, X):
        return super().decision_function(X)

    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score

    def param_statistic(self, save_file):
        pass
