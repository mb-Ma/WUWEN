import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from models.FEDformer.units.timefeatures import time_features
from models.FEDformer.units.augmentation import run_augmentation_single
import pickle as pkl
import warnings


warnings.filterwarnings('ignore')


class Dataset_Electricity(Dataset):
    def __init__(self, args, flag='train', size=None,
                 features='S', scale=True, timeenc=0, freq='t', seasonal_patterns=None):
        # size [past_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.past_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            #电价预测中，为了加上日前特征，参数设定的96点推理96点在数据集中实际上生成的是192点推理92点，为了适配192点推理92点，模型在初始化参数时让self.past_len+1，在加载数据集时先去掉+1
            self.past_len = (size[0]-1)*self.args.unit_len
            self.label_len = int(size[1]*self.args.unit_len)
            self.pred_len = size[2]*self.args.unit_len

        self.train_len=self.args.data_len_train*self.args.unit_len
        self.valid_len=self.args.data_len_valid*self.args.unit_len
        self.test_len=self.args.data_len_test*self.args.unit_len
        # 此处认为所有的数据都被观测到了，因此mask都为1

        self.kill_dim = len(self.args.feature_cols_mask2D)
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.flag=flag
        self.set_type = type_map[flag]
        self.features = features
        self.target_cols = args.target_cols #目标列
        self.feature_cols_da=args.feature_cols_da #日内特征
        self.feature_cols_mask1D=args.feature_cols_mask1D #日前特征
        self.feature_cols_mask2D=args.feature_cols_mask2D #日前特征
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.data_path = args.data_path
        self.__read_data__()
    def fix_time_format(self,index_string):
        if "24:00" in index_string:
            date, time = index_string.split(" ")
            new_date = pd.to_datetime(date) + pd.Timedelta(days=1)
            return f"{new_date.strftime('%Y-%m-%d')} 00:00"
        return index_string
    def __read_data__(self):
        self.scaler = StandardScaler()
        # df_raw = pd.read_csv('./datasets/ETTm1/ETTm1.csv')
        with open(self.data_path,'rb') as file:
            df_raw = pkl.load(file)
        # 确保数据是连续的
        df_raw = pd.DataFrame(np.ascontiguousarray(df_raw.values), 
                            index=df_raw.index, 
                            columns=df_raw.columns)
        #目标列在最后一列
        #数据集index格式处理
        df_raw=df_raw.reindex(columns=self.feature_cols_mask2D+self.feature_cols_mask1D+self.feature_cols_da+self.target_cols)
        # index = df_raw.index.map(lambda x: f"{x[0]} {x[1]}")
        # fixed_index = map(self.fix_time_format, index)
        # df_raw.index = pd.to_datetime(list(fixed_index)) - pd.Timedelta(minutes=15)
        df_raw.index.name='date'
        if self.args.data_mode==0:
            df_raw=df_raw.loc[:self.args.data_len_end]
            df_raw=df_raw.reset_index()

            border1s = [0, self.train_len-self.past_len, self.train_len+self.valid_len-self.past_len] 
            border2s = [self.train_len, self.train_len+self.valid_len, self.train_len+self.valid_len+self.test_len]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
            #计算可以被滚动的起始点index
            # sample_interval = [self.args.unit_len//self.args.unit_len,self.args.unit_len//self.args.unit_len,self.args.unit_len//self.args.unit_len][self.set_type]
            # self.sample_index = np.arange(0, border2-border1-self.past_len-self.pred_len+1, sample_interval)    

            #数据集划分
            border1s = [-self.past_len-self.test_len-self.valid_len-self.train_len, -self.past_len-self.test_len-self.valid_len, -self.test_len-self.past_len] 
            border2s = [-self.test_len-self.valid_len, -self.test_len, None]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
            if self.features == 'M' or self.features == 'MS':
                cols_data = df_raw.columns[1:]
                df_data = df_raw[cols_data]
            elif self.features == 'S':
                df_data = df_raw[self.target_cols]
            
            #日内特征和目标维度掩码
            cols = list(cols_data)
            # 需要掩盖两天的数据（日内特征）
            mask2D = [cols.index(key) for key in self.feature_cols_mask2D]
            # 需要掩盖一天的数据（日前价格）
            mask1D = [cols.index(key) for key in self.target_cols+self.feature_cols_mask1D]
            self.mask2D = mask2D
            self.mask1D = mask1D

            if self.scale:
                train_data = df_data[border1s[0]:border2s[0]]
                self.scaler.fit(train_data.values)
                data = self.scaler.transform(df_data.values)
            else:
                data = df_data.values

            df_stamp = df_raw[['date']][border1:border2]
            df_stamp['date'] = pd.to_datetime(df_stamp.date)
            if self.timeenc == 0:
                df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
                df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
                df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
                df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
                df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
                df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
                data_stamp = df_stamp.drop(['date'], 1).values
            elif self.timeenc == 1:
                data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
                data_stamp = data_stamp.transpose(1, 0)

            self.data_x = data[border1:border2]
            self.data_y = data[border1:border2]

            if self.set_type == 0 and self.args.augmentation_ratio > 0:
                self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

            self.data_stamp = data_stamp
        elif self.args.data_mode==1:
            train_row_start = df_raw.index.get_loc(self.args.data_time_train_start)
            train_row_end = df_raw.index.get_loc(self.args.data_time_train_end)
            valid_row_start = df_raw.index.get_loc(self.args.data_time_valid_start)
            valid_row_end = df_raw.index.get_loc(self.args.data_time_valid_end)
            test_row_start = df_raw.index.get_loc(self.args.data_time_test_start)
            test_row_end = df_raw.index.get_loc(self.args.data_time_test_end)
            df_raw=df_raw.reset_index()
            border1s = [train_row_start, valid_row_start-self.past_len, test_row_start-self.past_len]
            border2s = [train_row_end+1, valid_row_end+1, test_row_end +1]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
            sample_interval = [self.args.unit_len//self.args.unit_len,self.args.unit_len,self.args.unit_len][self.set_type]
            self.sample_index = np.arange(0, border2-border1-self.past_len-self.pred_len+1, sample_interval)    
            if self.features == 'M' or self.features == 'MS':
                cols_data = df_raw.columns[1:]
                df_data = df_raw[cols_data]
            elif self.features == 'S':
                df_data = df_raw[self.target_cols]
            #日内特征和目标维度掩码
            cols = list(cols_data)
            # 需要掩盖两天的数据（日内特征）
            mask2D = [cols.index(key) for key in self.feature_cols_mask2D]
            # 需要掩盖一天的数据（日前价格）
            mask1D = [cols.index(key) for key in self.target_cols+self.feature_cols_mask1D]
            self.mask2D = mask2D
            self.mask1D = mask1D
            if self.scale:
                train_data = df_data[border1s[0]:border2s[0]]
                self.scaler.fit(train_data.values)
                data = self.scaler.transform(df_data.values)
            else:
                data = df_data.values

            df_stamp = df_raw[['date']][border1:border2]
            df_stamp['date'] = pd.to_datetime(df_stamp.date)
            if self.timeenc == 0:
                df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
                df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
                df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
                df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
                df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
                df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
                data_stamp = df_stamp.drop(['date'], 1).values
            elif self.timeenc == 1:
                data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
                data_stamp = data_stamp.transpose(1, 0)
            self.origin_data=np.array(df_data[border1:border2]) #原始的数据，用于给schrodinger提供未经tranform处理的原始
            self.data_x = data[border1:border2]
            self.data_y = data[border1:border2]

            if self.set_type == 0 and self.args.augmentation_ratio > 0:
                self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

            self.data_stamp = data_stamp


    def __getitem__(self, index):
        # 验证集、测试集每隔一天推理96点
        # index = self.sample_index[id]
        s_begin = index
        s_end = s_begin + self.past_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        # 这里显式复制，避免修改 seq_x 影响 seq_y
        seq_x = self.data_x[s_begin:s_end+self.args.unit_len].copy()
        seq_y = self.data_y[r_begin:r_end].copy()
        seq_x_mark = self.data_stamp[s_begin:s_end+self.args.unit_len].copy()
        seq_y_mark = self.data_stamp[r_begin:r_end].copy()

        # mask 日内特征的最后2天
        seq_x[-self.args.unit_len*2:, self.mask2D] = 0
        # mask 日前价格
        seq_x[-self.args.unit_len:, self.mask1D] = 0

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.past_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

#每隔96点滚动进行一次推理
class Dataset_common2(Dataset):
    def __init__(self, args, flag='train', size=None,
                 features='S', scale=True, timeenc=0, freq='t', seasonal_patterns=None):
        # size [past_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.past_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.past_len = size[0]*self.args.unit_len
            self.label_len = int(size[1]*self.args.unit_len)
            self.pred_len = size[2]*self.args.unit_len

        self.train_len=self.args.data_len_train*self.args.unit_len
        self.valid_len=self.args.data_len_valid*self.args.unit_len
        self.test_len=self.args.data_len_test*self.args.unit_len
        # 此处认为所有的数据都被观测到了，因此mask都为1

        self.kill_dim = len(self.args.feature_cols_mask2D)
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.flag=flag
        self.set_type = type_map[flag]
        self.features = features
        self.target_cols = args.target_cols #目标列
        self.feature_cols_da=args.feature_cols_da #日内特征
        self.feature_cols_mask1D=args.feature_cols_mask1D #日前特征
        self.feature_cols_mask2D=args.feature_cols_mask2D #日前特征
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.data_path = args.data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        # df_raw = pd.read_csv('./datasets/ETTm1/ETTm1.csv')
        with open(self.data_path,'rb') as file:
            df_raw = pkl.load(file)
        
        #TimeXer 的目标列在最后一列

        df_raw=df_raw.reindex(columns=self.feature_cols_mask2D+self.feature_cols_mask1D+self.feature_cols_da+self.target_cols)
            
    
        if self.args.data_mode==0:
            df_raw=df_raw.reset_index()
            df_raw=df_raw.loc[:self.args.data_len_end]
            border1s = [0, self.train_len-self.past_len, self.train_len+self.valid_len-self.past_len] 
            border2s = [self.train_len, self.train_len+self.valid_len, self.train_len+self.valid_len+self.test_len]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
            #计算可以被滚动的起始点index

            sample_interval = [self.args.unit_len//self.args.unit_len,self.args.unit_len,self.args.unit_len][self.set_type]
            self.sample_index = np.arange(0, border2-border1-self.past_len-self.pred_len+1, sample_interval)    

            #数据集划分
            border1s = [-self.past_len-self.test_len-self.valid_len-self.train_len, -self.past_len-self.test_len-self.valid_len, -self.test_len-self.past_len] 
            border2s = [-self.test_len-self.valid_len, -self.test_len, None]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]  
            if self.features == 'M' or self.features == 'MS':
                cols_data = df_raw.columns[1:]
                df_data = df_raw[cols_data]
            elif self.features == 'S':
                df_data = df_raw[self.target_cols]

            if self.scale:
                train_data = df_data[border1s[0]:border2s[0]]
                self.scaler.fit(train_data.values)
                data = self.scaler.transform(df_data.values)
            else:
                data = df_data.values

            df_stamp = df_raw[['date']][border1:border2]
            df_stamp['date'] = pd.to_datetime(df_stamp.date)
            if self.timeenc == 0:
                df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
                df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
                df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
                df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
                df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
                df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
                data_stamp = df_stamp.drop(['date'], 1).values
            elif self.timeenc == 1:
                data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
                data_stamp = data_stamp.transpose(1, 0)

            self.data_x = data[border1:border2]
            self.data_y = data[border1:border2]

            if self.set_type == 0 and self.args.augmentation_ratio > 0:
                self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

            self.data_stamp = data_stamp
        elif self.args.data_mode==1:
            train_row_start = df_raw.index.get_loc(self.args.data_time_train_start)
            train_row_end = df_raw.index.get_loc(self.args.data_time_train_end)
            valid_row_start = df_raw.index.get_loc(self.args.data_time_valid_start)
            valid_row_end = df_raw.index.get_loc(self.args.data_time_valid_end)
            test_row_start = df_raw.index.get_loc(self.args.data_time_test_start)
            test_row_end = df_raw.index.get_loc(self.args.data_time_test_end)
            df_raw=df_raw.reset_index()
            border1s = [train_row_start, valid_row_start-self.past_len, test_row_start-self.past_len]
            border2s = [train_row_end+1, valid_row_end+1, test_row_end +1]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
            sample_interval = [self.args.unit_len//self.args.unit_len,self.args.unit_len,self.args.unit_len][self.set_type]
            self.sample_index = np.arange(0, border2-border1-self.past_len-self.pred_len+1, sample_interval)    
            if self.features == 'M' or self.features == 'MS':
                cols_data = df_raw.columns[1:]
                df_data = df_raw[cols_data]
            elif self.features == 'S':
                df_data = df_raw[self.target_cols]

            if self.scale:
                train_data = df_data[border1s[0]:border2s[0]]
                self.scaler.fit(train_data.values)
                data = self.scaler.transform(df_data.values)
            else:
                data = df_data.values

            df_stamp = df_raw[['date']][border1:border2]
            df_stamp['date'] = pd.to_datetime(df_stamp.date)
            if self.timeenc == 0:
                df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
                df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
                df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
                df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
                df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
                df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
                data_stamp = df_stamp.drop(['date'], 1).values
            elif self.timeenc == 1:
                data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
                data_stamp = data_stamp.transpose(1, 0)
            self.origin_data=np.array(df_data[border1:border2]) #原始的数据，用于给schrodinger提供未经tranform处理的原始
            self.data_x = data[border1:border2]
            self.data_y = data[border1:border2]

            if self.set_type == 0 and self.args.augmentation_ratio > 0:
                self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

            self.data_stamp = data_stamp


    def __getitem__(self, id):
        #验证集、测试集每隔一天推理96点
        index = self.sample_index[id]
        s_begin = index
        s_end = s_begin + self.past_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.sample_index)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

# 每隔1点滚动推理96点
class Dataset_common(Dataset):
    def __init__(self, args, flag='train', size=None,
                 features='S', scale=True, timeenc=0, freq='t', seasonal_patterns=None):
        # size [past_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.past_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.past_len = size[0]*self.args.unit_len
            self.label_len = int(size[1]*self.args.unit_len)
            self.pred_len = size[2]*self.args.unit_len

        self.train_len=self.args.data_len['train']*self.args.unit_len
        self.valid_len=self.args.data_len['valid']*self.args.unit_len
        self.test_len=self.args.data_len['test']*self.args.unit_len
        # 此处认为所有的数据都被观测到了，因此mask都为1

        self.kill_dim = len(self.args.feature_cols_mask2D)
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.flag=flag
        self.set_type = type_map[flag]
        self.features = features
        self.target_cols = args.target_cols #目标列
        self.feature_cols_da=args.feature_cols_da #日内特征
        self.feature_cols_mask1D=args.feature_cols_mask1D #日前特征
        self.feature_cols_mask2D=args.feature_cols_mask2D #日前特征
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.data_path = args.data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(self.data_path)
        # with open(self.data_path,'rb') as file:
        #     df_raw = pkl.load(file)
        
        #TimeXer 的目标列在最后一列
        df_raw.set_index('timestamp',inplace=True)
        df_raw.index = pd.to_datetime(df_raw.index)
        df_raw=df_raw.reindex(columns=self.feature_cols_mask2D+self.feature_cols_mask1D+self.feature_cols_da+self.target_cols)
        df_raw=df_raw.reset_index()
    
        if self.args.data_mode==0:
            df_raw=df_raw.reset_index()
            df_raw=df_raw.loc[:self.args.data_len['end']]
            border1s = [0, self.train_len-self.past_len, self.train_len+self.valid_len-self.past_len] 
            border2s = [self.train_len, self.train_len+self.valid_len, self.train_len+self.valid_len+self.test_len]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
            #计算可以被滚动的起始点index
            sample_interval = [self.args.unit_len//96,self.args.unit_len,self.args.unit_len][self.set_type]
            self.sample_index = np.arange(0, border2-border1-self.past_len-self.pred_len+1, sample_interval)    

            #数据集划分
            border1s = [-self.past_len-self.test_len-self.valid_len-self.train_len, -self.past_len-self.test_len-self.valid_len, -self.test_len-self.past_len] 
            border2s = [-self.test_len-self.valid_len, -self.test_len, None]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]  
            if self.features == 'M' or self.features == 'MS':
                cols_data = df_raw.columns[1:]
                df_data = df_raw[cols_data]
            elif self.features == 'S':
                df_data = df_raw[self.target_cols]

            if self.scale:
                train_data = df_data[border1s[0]:border2s[0]]
                self.scaler.fit(train_data.values)
                data = self.scaler.transform(df_data.values)
            else:
                data = df_data.values

            df_stamp = df_raw[['timestamp']][border1:border2]
            df_stamp['timestamp'] = pd.to_datetime(df_stamp.date)
            if self.timeenc == 0:
                df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
                df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
                df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
                df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
                df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
                df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
                data_stamp = df_stamp.drop(['timestamp'], 1).values
            elif self.timeenc == 1:
                data_stamp = time_features(pd.to_datetime(df_stamp['timestamp'].values), freq=self.freq)
                data_stamp = data_stamp.transpose(1, 0)

            self.data_x = data[border1:border2]
            self.data_y = data[border1:border2]

            if self.set_type == 0 and self.args.augmentation_ratio > 0:
                self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

            self.data_stamp = data_stamp
        elif self.args.data_mode==1:
            train_row_start = df_raw.index.get_loc(self.args.data_time_train_start)
            train_row_end = df_raw.index.get_loc(self.args.data_time_train_end)
            valid_row_start = df_raw.index.get_loc(self.args.data_time_valid_start)
            valid_row_end = df_raw.index.get_loc(self.args.data_time_valid_end)
            test_row_start = df_raw.index.get_loc(self.args.data_time_test_start)
            test_row_end = df_raw.index.get_loc(self.args.data_time_test_end)
            df_raw=df_raw.reset_index()
            border1s = [train_row_start, valid_row_start-self.past_len, test_row_start-self.past_len]
            border2s = [train_row_end+1, valid_row_end+1, test_row_end +1]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
 
            sample_interval = [self.args.unit_len//self.args.unit_len,self.args.unit_len,self.args.unit_len][self.set_type]
            self.sample_index = np.arange(0, border2-border1-self.past_len-self.pred_len+1, sample_interval)    
            if self.features == 'M' or self.features == 'MS':
                cols_data = df_raw.columns[1:]
                df_data = df_raw[cols_data]
            elif self.features == 'S':
                df_data = df_raw[self.target_cols]

            if self.scale:
                train_data = df_data[border1s[0]:border2s[0]]
                self.scaler.fit(train_data.values)
                data = self.scaler.transform(df_data.values)
            else:
                data = df_data.values

            df_stamp = df_raw[['timestamp']][border1:border2]
            df_stamp['timestamp'] = pd.to_datetime(df_stamp.date)
            if self.timeenc == 0:
                df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
                df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
                df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
                df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
                df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
                df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
                data_stamp = df_stamp.drop(['timestamp'], 1).values
            elif self.timeenc == 1:
                data_stamp = time_features(pd.to_datetime(df_stamp['timestamp'].values), freq=self.freq)
                data_stamp = data_stamp.transpose(1, 0)
            self.origin_data=np.array(df_data[border1:border2]) #原始的数据，用于给schrodinger提供未经tranform处理的原始
            self.data_x = data[border1:border2]
            self.data_y = data[border1:border2]

            if self.set_type == 0 and self.args.augmentation_ratio > 0:
                self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

            self.data_stamp = data_stamp
        elif self.args.data_mode==2:
            # 按比例划分数据集
            df_raw = df_raw.reset_index()
            total_data_len = len(df_raw)
            
            # 根据比例计算各数据集的长度
            train_ratio, valid_ratio, test_ratio = self.args.train_val_test_ratio
            train_len_ratio = int(total_data_len * train_ratio)
            valid_len_ratio = int(total_data_len * valid_ratio)
            test_len_ratio = total_data_len - train_len_ratio - valid_len_ratio  # 确保总和等于总长度
            
            # 计算边界
            border1s = [0, train_len_ratio-self.past_len, train_len_ratio+valid_len_ratio-self.past_len]
            border2s = [train_len_ratio, train_len_ratio+valid_len_ratio, train_len_ratio+valid_len_ratio+test_len_ratio]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
            
            # 计算可以被滚动的起始点index
            sample_interval = [self.args.unit_len//self.args.unit_len, self.args.unit_len//self.args.unit_len, self.args.unit_len//self.args.unit_len][self.set_type]
            self.sample_index = np.arange(0, border2-border1-self.past_len-self.pred_len+1, sample_interval)
            
            # 数据集划分
            if self.features == 'M' or self.features == 'MS':
                cols_data = df_raw.columns[1:]
                df_data = df_raw[cols_data]
            elif self.features == 'S':
                df_data = df_raw[self.target_cols]

            if self.scale:
                train_data = df_data[border1s[0]:border2s[0]]
                self.scaler.fit(train_data.values)
                data = self.scaler.transform(df_data.values)
            else:
                data = df_data.values

            df_stamp = df_raw[['timestamp']][border1:border2]
            df_stamp['timestamp'] = pd.to_datetime(df_stamp.timestamp)
            if self.timeenc == 0:
                df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
                df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
                df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
                df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
                df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
                df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
                data_stamp = df_stamp.drop(['timestamp'], 1).values
            elif self.timeenc == 1:
                data_stamp = time_features(pd.to_datetime(df_stamp['timestamp'].values), freq=self.freq)
                data_stamp = data_stamp.transpose(1, 0)
            
            self.origin_data = np.array(df_data[border1:border2])  # 原始的数据，用于给schrodinger提供未经transform处理的原始
            self.data_x = data[border1:border2]
            self.data_y = data[border1:border2]

            if self.set_type == 0 and self.args.augmentation_ratio > 0:
                self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

            self.data_stamp = data_stamp
            


    def __getitem__(self, index):
        #验证集、测试集每隔一天推理96点
        # index = self.sample_index[id]
        s_begin = index
        s_end = s_begin + self.past_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.past_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

# class Dataset_Custom(Dataset):
#     def __init__(self, root_path, flag='train', size=None,
#                  features='S', data_path='ETTh1.csv',
#                  target='OT', scale=True, timeenc=0, freq='h'):
#         # size [seq_len, label_len, pred_len]
#         # info
#         if size == None:
#             self.seq_len = 24 * 4 * 4
#             self.label_len = 24 * 4
#             self.pred_len = 24 * 4
#         else:
#             self.seq_len = size[0]
#             self.label_len = size[1]
#             self.pred_len = size[2]
#         # init
#         assert flag in ['train', 'test', 'val']
#         type_map = {'train': 0, 'val': 1, 'test': 2}
#         self.set_type = type_map[flag]

#         self.features = features
#         self.target = target
#         self.scale = scale
#         self.timeenc = timeenc
#         self.freq = freq

#         self.root_path = root_path
#         self.data_path = data_path
#         self.__read_data__()

#     def __read_data__(self):
#         self.scaler = StandardScaler()
#         with open(self.data_path,'rb') as file:
#             df_raw = pkl.load(file)
#         df_raw=df_raw.reset_index()
#         '''
#         df_raw.columns: ['date', ...(other features), target feature]
#         '''
#         cols = list(df_raw.columns)
#         cols.remove(self.target)
#         cols.remove('date')
#         df_raw = df_raw[['date'] + cols + [self.target]]
#         # print(cols)
#         num_train = int(len(df_raw) * 0.7)
#         num_test = int(len(df_raw) * 0.2)
#         num_vali = len(df_raw) - num_train - num_test
#         border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
#         border2s = [num_train, num_train + num_vali, len(df_raw)]
#         border1 = border1s[self.set_type]
#         border2 = border2s[self.set_type]

#         if self.features == 'M' or self.features == 'MS':
#             cols_data = df_raw.columns[1:]
#             df_data = df_raw[cols_data]
#         elif self.features == 'S':
#             df_data = df_raw[[self.target]]

#         if self.scale:
#             train_data = df_data[border1s[0]:border2s[0]]
#             self.scaler.fit(train_data.values)
#             data = self.scaler.transform(df_data.values)
#         else:
#             data = df_data.values

#         df_stamp = df_raw[['date']][border1:border2]
#         df_stamp['date'] = pd.to_datetime(df_stamp.date)
#         if self.timeenc == 0:
#             df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
#             df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
#             df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
#             df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
#             data_stamp = df_stamp.drop(['date'], 1).values
#         elif self.timeenc == 1:
#             data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
#             data_stamp = data_stamp.transpose(1, 0)

#         self.data_x = data[border1:border2]
#         self.data_y = data[border1:border2]
#         self.data_stamp = data_stamp

#     def __getitem__(self, index):
#         s_begin = index
#         s_end = s_begin + self.seq_len
#         r_begin = s_end - self.label_len
#         r_end = r_begin + self.label_len + self.pred_len

#         seq_x = self.data_x[s_begin:s_end]
#         seq_y = self.data_y[r_begin:r_end]
#         seq_x_mark = self.data_stamp[s_begin:s_end]
#         seq_y_mark = self.data_stamp[r_begin:r_end]

#         return seq_x, seq_y, seq_x_mark, seq_y_mark

#     def __len__(self):
#         return len(self.sample_index)

#     def inverse_transform(self, data):
#         return self.scaler.inverse_transform(data)