from utils.scaler import MinMaxScaler, ZScoreScaler
from utils.time_feature import time_feature
from typing import List
import numpy as np
import json
from logger import logger
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pickle as pkl
import warnings
import pyarrow.parquet as pq


warnings.filterwarnings('ignore')

class TS_Dataset(Dataset):
    def __init__(self, logger, data_path, train_val_test_ratio, in_len=12, out_len=12, mode='train', norm_type='ZScore', if_col_norm=False, timeenc=0, freq='h', stride=1, in_var=None, out_var=None):
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
        self.in_var = in_var
        self.out_var = out_var
        
        logger.info(f"the num of input variables:{len(self.in_var)}, the num of output variables: {len(self.out_var)}.")
        
        self.data, self.date, self.jobtiming = self._load_data() # the data shape is LxC
        
        self.filter_data_flag = False
        if self.filter_data_flag and (self.mode == "train"):
            self.training_indices = self._filter_training_data(0.3)
        
        # self.__load__pre_repr() # load representation

    def _filter_training_data(self, keep_ratio):
        training_indices = []
        low_indices = []
        for i in range(len(self.data) - self.in_len - self.out_len+1):
            if self.is_good_sample(i):
                training_indices.append(i)
            else:
                low_indices.append(i)
        counts = int(keep_ratio * len(low_indices))
        training_indices += low_indices[:counts]
        logger.info(f"delete {len(low_indices)-counts} samples with {keep_ratio}")
        return training_indices

    def is_good_sample(self, i):
        sample = self.data[i+self.in_len: i+self.in_len+self.out_len]
        if sample.std() < 3:
            return False
        else:
            return True

    def _load_data(self):
        try:
            if 'csv' in self.data_path:
                df_raw = pd.read_csv(self.data_path)
            elif 'parquet' in self.data_path:
                df_raw = pq.read_table(self.data_path)
                df_raw = df_raw.to_pandas(ignore_metadata=True)
            else:
                print("Give the right dataset")

            logger.info(f"load original data, the data shape is: {df_raw.shape}")
        except (FileNotFoundError, ValueError) as e:
            raise ValueError(f'Error loading data file: {self.data_path}') from e
        # import pdb; pdb.set_trace()


        # 做数据采样，预测12步，相当于预测120步，性能提升从13.01提升到10.38
        # df_raw['timestamp'] = pd.to_datetime(df_raw.timestamp)
        # df_raw.set_index('timestamp', inplace=True)
        # df_raw = df_raw.resample('5T').mean()
        # df_raw = df_raw.reset_index()
    
        # load more data
        # more_df = pq.read_table("data/processed_data/clean_data_r30n1.parquet")
        # more_df = more_df.to_pandas(ignore_metadata=True)
        # df_raw = pd.concat([more_df, df_raw], axis=0, ignore_index=True)
        cols = df_raw.columns[1:]
        data = df_raw[cols].values
        # here choose data source
        # data (LxC) C_0 cpu_util; C_1:8 gpu_util; C_9 mem; C_10 power; C_11 in_temp; C_12 out_temp1; C_13 out_temp2

        df_stamp = df_raw[['timestamp']]
        df_stamp['timestamp'] = pd.to_datetime(df_stamp.timestamp)

        # split data into train, val and test parts
        total_len = len(data)
        valid_len = int(total_len * self.train_val_test_ratio[1]) #15234
        test_len = int(total_len * self.train_val_test_ratio[2]) #30469
        # valid_len = 15234
        # test_len = 30469
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
            # df_stamp['day'] = df_stamp.timestamp.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.timestamp.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.timestamp.apply(lambda row: row.hour, 1)
            # df_stamp['min'] = df_stamp.timestamp.apply(lambda row: row.minute, 1)
            data_stamp = df_stamp.drop(['timestamp'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_feature(pd.to_datetime(df_stamp['timestamp'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        else:
            raise NotImplementedError("Specify a right time encoder type")
        
        jobtimming = self.__loadjob__()

        if self.mode == 'train':
            return data[:train_len].copy(), data_stamp[:train_len].copy(), jobtimming[:train_len].copy()  
        elif self.mode == 'valid':
            return data[train_len : train_len + valid_len].copy(), data_stamp[train_len : train_len + valid_len].copy(), jobtimming[train_len : train_len + valid_len].copy()
        else:  # self.mode == 'test'
            return data[train_len + valid_len:].copy(), data_stamp[train_len + valid_len:].copy(), jobtimming[train_len + valid_len:].copy()  
        
    def __loadjob__(self):
        df = pq.read_table("data/processed_data/clean_data_r28n4_1min.parquet")
        df = df.to_pandas(ignore_metadata=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        job_timing = df.values[:, 1:5] # job_sub, job_start, job_end. （values:(0,1)）, job_counts
        return job_timing


    def __getitem__(self, idx):
        '''
        return single sample
        '''
        start = idx * self.stride
        end = start + self.in_len
        
        x = self.data[start:end, self.in_var] # (in_len, features)
        y = self.data[end : end+self.out_len, self.out_var]
        
        # whether adding time feats
        if self.timeenc > -1:
            x_time = self.date[start:end]
            y_time = self.date[end: end+self.out_len]
            job_time = self.jobtiming[start:end]
            # import pdb; pdb.set_trace()
            return x, x_time, y, y_time, job_time.astype('float32')
            # return x, x_time, y, y_time, idx
        else:
            return x, y

    def __len__(self):
        if self.filter_data_flag and (self.mode == "train"):
            return len(self.training_indices) - self.in_len - self.out_len + 1
        else:
            return int(np.floor((self.data.shape[0] - self.in_len - self.out_len) // self.stride)) + 1
    
def get_dataloaders(logger, cfg):
    train_dataset = TS_Dataset(logger, cfg.data_path, cfg.train_val_test_ratio, in_len=cfg.in_len, out_len=cfg.out_len, mode='train', norm_type=cfg.norm_type, if_col_norm=cfg.if_col_norm, timeenc=cfg.timeenc, freq=cfg.freq, stride=cfg.stride, in_var=cfg.in_var, out_var=cfg.out_var)

    valid_dataset = TS_Dataset(logger, cfg.data_path, cfg.train_val_test_ratio, in_len=cfg.in_len, out_len=cfg.out_len, mode='valid', norm_type=cfg.norm_type, if_col_norm=cfg.if_col_norm, timeenc=cfg.timeenc, freq=cfg.freq, stride=cfg.stride, in_var=cfg.in_var, out_var=cfg.out_var)

    test_dataset = TS_Dataset(logger, cfg.data_path, cfg.train_val_test_ratio, in_len=cfg.in_len, out_len=cfg.out_len, mode='test', norm_type=cfg.norm_type, if_col_norm=cfg.if_col_norm, timeenc=cfg.timeenc, freq=cfg.freq, stride=cfg.stride, in_var=cfg.in_var, out_var=cfg.out_var)

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
  
  