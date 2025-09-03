from utils.scaler import MinMaxScaler, ZScoreScaler
from utils.time_feature import time_feature
from typing import List
import numpy as np
import json
from logger import logger
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

class UTS_Dataset(Dataset):
    def __init__(self, logger, data_path, train_val_test_ratio, in_len=12, out_len=12, mode='train', norm_type='ZScore', if_col_norm=False, timeenc=0, freq='h', stride=1):
        self.logger = logger
        self.data_path = data_path
        self.train_val_test_ratio = train_val_test_ratio
        self.mode = mode
        self.norm_type = norm_type
        self.timeenc = timeenc
        self.in_len = in_len
        self.out_len = out_len
        self.if_col_norm = if_col_norm
        self.stride = stride
        self.data, self.date = self._load_data() # the data shape is LxC

    def _load_data(self):
        try:
            df_raw = pd.read_csv(self.data_path)
            logger.info(f"load original data, the data shape is: {df_raw.shape}")
        except (FileNotFoundError, ValueError) as e:
            raise ValueError(f'Error loading data file: {self.data_path}') from e
        
        cols = df_raw.columns[1:]
        data = df_raw[cols].values
        # here choose data source
        # data (LxC) C_0 cpu_util; C_1:8 gpu_util; C_9 mem; C_10 power; C_11 in_temp; C_12 out_temp1; C_13 out_temp2
        data = data[:, :1] # cpu only
        df_stamp = df_raw[['timestamp']]
        df_stamp['timestamp'] = pd.to_datetime(df_stamp.timestamp)

        # split data into train, val and test parts
        total_len = len(data)
        valid_len = int(total_len * self.train_val_test_ratio[1])
        test_len = int(total_len * self.train_val_test_ratio[2])
        train_len = total_len - valid_len - test_len

        # Normalize
        if self.norm_type == "MinMax":
            self.normalizer = MinMaxScaler(self.if_col_norm)
        elif self.norm_type == "ZScore":
            self.normalizer = ZScoreScaler(self.if_col_norm)
        else:
            raise KeyError("Specific a normalization type")
        self.normalizer.fit(data[:train_len]) # LxN data
        data = self.normalizer.transform(data)

        if self.timeenc == 0:
            '''
            discrete/categorical feature
            '''
            df_stamp['day'] = df_stamp.timestamp.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.timestamp.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.timestamp.apply(lambda row: row.hour, 1)
            df_stamp['min'] = df_stamp.timestamp.apply(lambda row: row.minute, 1)
            data_stamp = df_stamp.drop(['timestamp'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_feature(pd.to_datetime(df_stamp['timestamp'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        else:
            raise NotImplementedError("Specify a right time encoder type")
        
        if self.mode == 'train':
            return data[:train_len].copy(), data_stamp[:train_len].copy()
        elif self.mode == 'valid':
            return data[train_len : train_len + valid_len].copy(), data_stamp[train_len : train_len + valid_len].copy() 
        else:  # self.mode == 'test'
            return data[train_len + valid_len:].copy(), data_stamp[train_len + valid_len:].copy()
    
    def __getitem__(self, idx):
        '''
        return single sample
        '''
        start = idx * self.stride
        end = start + self.in_len
        x = self.data[start:end] # (in_len, features)
        y = self.data[end : end+self.out_len]

        if self.timeenc > -1:
            x_time = self.date[start:end]
            y_time = self.date[end: end+self.out_len]
            return x, x_time, y, y_time
        else:
            return x, y

    def __len__(self):
        return int(np.floor((self.data.shape[0] - self.in_len - self.out_len) // self.stride)) + 1
    
def get_dataloaders(logger, cfg):
    train_dataset = UTS_Dataset(logger, cfg.data_path, cfg.train_val_test_ratio, in_len=cfg.in_len, out_len=cfg.out_len, mode='train', norm_type=cfg.norm_type, if_col_norm=cfg.if_col_norm, timeenc=cfg.timeenc, freq=cfg.freq, stride=cfg.stride)

    valid_dataset = UTS_Dataset(logger, cfg.data_path, cfg.train_val_test_ratio, in_len=cfg.in_len, out_len=cfg.out_len, mode='valid', norm_type=cfg.norm_type, if_col_norm=cfg.if_col_norm, timeenc=cfg.timeenc, freq=cfg.freq, stride=cfg.stride)

    test_dataset = UTS_Dataset(logger, cfg.data_path, cfg.train_val_test_ratio, in_len=cfg.in_len, out_len=cfg.out_len, mode='test', norm_type=cfg.norm_type, if_col_norm=cfg.if_col_norm, timeenc=cfg.timeenc, freq=cfg.freq, stride=cfg.stride)

    scaler = train_dataset.normalizer
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0
    )

    return train_loader, valid_loader, test_loader, scaler

if __name__ =='__main__':
  data_path = './data/processed_data/data.dat'
  train_val_test_ratio = [0.7, 0.1, 0.2]
  logger = None
  mode = "train"
  in_len, out_len = 12, 12
