from typing import Dict
import torchinfo
import tqdm, math
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from TSB_AD.evaluation.metrics import get_metrics
from scipy.signal import savgol_filter


from ..utils.utility import get_activation_by_name
from ..utils.torch_utility import EarlyStoppingTorch, get_gpu
from ..utils.dataset import ForecastDataset

class AdaptiveConcatPool1d(nn.Module):
    def __init__(self):
        super().__init__()
        self.ap = torch.nn.AdaptiveAvgPool1d(1)
        self.mp = torch.nn.AdaptiveAvgPool1d(1)
    
    def forward(self, x):
        return torch.cat([self.ap(x), self.mp(x)], 1)

class CNNModel(nn.Module):
    def __init__(self,
                 n_features,
                 num_channel=[32, 32, 40],
                 kernel_size=3,
                 stride=1,
                 predict_time_steps=1,
                 dropout_rate=0.25,
                 hidden_activation='relu',
                 device='cpu'):

        # initialize the super class
        super(CNNModel, self).__init__()

        # save the default values
        self.n_features = n_features
        self.dropout_rate = dropout_rate
        self.hidden_activation = hidden_activation
        self.kernel_size = kernel_size
        self.stride = stride
        self.predict_time_steps = predict_time_steps
        self.num_channel = num_channel
        self.device = device

        # get the object for the activations functions
        self.activation = get_activation_by_name(hidden_activation)

        # initialize encoder and decoder as a sequential
        self.conv_layers = nn.Sequential()
        prev_channels = self.n_features

        for idx, out_channels in enumerate(self.num_channel[:-1]):
            self.conv_layers.add_module(
                "conv" + str(idx),
                torch.nn.Conv1d(prev_channels, self.num_channel[idx + 1], 
                self.kernel_size, self.stride))
            self.conv_layers.add_module(self.hidden_activation + str(idx),
                                    self.activation)
            self.conv_layers.add_module("pool" + str(idx), nn.MaxPool1d(kernel_size=2))
            prev_channels = out_channels

        self.fc = nn.Sequential(
            AdaptiveConcatPool1d(),
            torch.nn.Flatten(),
            torch.nn.Linear(2*self.num_channel[-1], self.num_channel[-1]),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(self.num_channel[-1], self.n_features)
        )

    def forward(self, x):
        b, c, l = x.shape
        #x = x.view(b, c, l)
        x = self.conv_layers(x)     # [128, feature, 23]

        outputs = torch.zeros(self.predict_time_steps, b, self.n_features).to(self.device)
        for t in range(self.predict_time_steps):
            decoder_input = self.fc(x)
            outputs[t] = torch.squeeze(decoder_input, dim=-2)

        return outputs
    
class CNN_uns():
    def __init__(self,
                 window_size=100,
                 pred_len=1,
                 batch_size=256,
                 epochs=30,
                 lr=0.0008,
                 feats=1,
                 num_channel=[32, 32, 40],
                 validation_size=0.2):
        super().__init__()
        self.__anomaly_score = None
        
        cuda = True
        self.y_hats = None
        
        self.cuda = cuda
        self.device = get_gpu(self.cuda)
        
        self.window_size = window_size
        self.pred_len = pred_len
        self.batch_size = batch_size
        self.epochs = 5
        
        self.feats = feats
        self.num_channel = num_channel
        self.lr = lr
        
        self.model = CNNModel(n_features=feats, num_channel=num_channel, predict_time_steps=self.pred_len, device=self.device).to(self.device)
        
        self.optimizer = None
        self.loss = nn.MSELoss()
        self.save_path = None
        
        self.mu = None
        self.sigma = None
        self.eps = 1e-10
    
    def _normalize(self, ts):
        mean, std = np.mean(ts, axis=0), np.std(ts, axis=0)
        std = np.where(std == 0, 1e-8, std)  # Avoid division by zero
        return (ts - mean) / std
    
    def create_sliding_window(self, X, window, pred_w,  shuffle=True):
        total_length = X.shape[2]
        W = window
        P = pred_w
        num_samples = (total_length - W - P)  + 1      

        # 2 coordinates for the sliding window per window
        X_train_indices = []
        Y_train_indices = []
        for i in range(total_length):
            X_train_indices.append( np.arange(i, i + W) )
            Y_train_indices.append( np.arange(i + W, i + W + P) )
        
        X_train_indices = np.array(X_train_indices)
        Y_train_indices = np.array(Y_train_indices)
        

        if shuffle:
            indices = np.random.permutation(num_samples)
            X_train_indices = X_train_indices[indices]
            Y_train_indices = Y_train_indices[indices]
        return X_train_indices, Y_train_indices

    def fit(self, data, train_idx= 1000):
        print("Training CNN_RW model...")
        ts = self._normalize(data)

        ts = torch.from_numpy(ts).float().permute(1, 0).unsqueeze(0).to(self.model.device) # 1, feat, LEN

        ts_changes = -ts.clone().to(self.device)  


        print("Sliding window...")
        X_train_indices, Y_train_indices = self.create_sliding_window(ts, self.window_size, self.pred_len, shuffle=True)

        ts_changes.requires_grad = True

        self.optimizer = optim.Adam([
        {
        'params': self.model.parameters(),
        'lr': self.lr},
        {
        'params': [ts_changes],
        'lr': 0.01 # higher learning rate for go down to 0 faster
        }])
        
        for epoch in range(1, self.epochs + 1):
            self.model.train(mode=True)
            avg_loss = 0
            
            for i in range(0, X_train_indices.shape[0], self.batch_size):
                x_batch_indices = X_train_indices[i:i + self.batch_size]
                y_batch_indices = Y_train_indices[i:i + self.batch_size]


                x = ts[0, :, x_batch_indices].permute(1, 0, 2) # B, feat, W
                target = ts[0, :, y_batch_indices].permute(1, 0, 2)

                x_changes = ts_changes[0, :, x_batch_indices].permute(1, 0, 2) # B, feat, W
                target_changes = ts_changes[0, :, y_batch_indices].permute(1, 0, 2)

                self.optimizer.zero_grad()
                
                output = self.model(x + x_changes)

                output = output.view(-1, self.feats*self.pred_len)
                target = target.view(-1, self.feats*self.pred_len)
                target_changes = target_changes.view(-1, self.feats*self.pred_len)

                loss = self.loss(output, target + target_changes)
                loss.backward()

                self.optimizer.step()
                
                avg_loss += loss.cpu().item()
            
            self.optimizer.zero_grad()    
            loss = torch.norm(ts_changes, p=1) 
            loss.backward()
            self.optimizer.step()
            
            avg_loss /= max(X_train_indices.shape[0] // self.batch_size, 1)
            print(f"Epoch [{epoch}/{self.epochs}] | Loss: {avg_loss:.4f} | Changes: {loss.item():.4f}")


        scores = np.abs(ts_changes.detach().cpu().numpy()[0,:,:])

        scores_sav = savgol_filter(scores, self.window_size, 2)

        scores += scores_sav

        scores = np.abs(scores - np.mean(scores, axis=1, keepdims=True)) / (np.std(scores, axis=1, keepdims=True) + 1e-8)

        scores = scores.mean(axis=0)

        return scores


    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score
    
    def get_y_hat(self) -> np.ndarray:
        return self.y_hats
    
    def param_statistic(self, save_file):
        model_stats = torchinfo.summary(self.model, (self.batch_size, self.window_size), verbose=0)
        with open(save_file, 'w') as f:
            f.write(str(model_stats))
