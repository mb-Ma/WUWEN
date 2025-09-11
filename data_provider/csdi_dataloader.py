#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @FileName      : diffusion_dataset
# @Time          : 2024-11-20 11:07:32
# @Author        : ZhangHonglin
# @Email         : honglin-24@mails.tsinghua.edu.cn
# @description   : 
"""

import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from datetime import datetime, timedelta


class Diffusion_Dataset(Dataset):
    def __init__(self, config, data, mode):
        unit_len = config['data_config']["unit_len"]
        self.pred_length = config['data_config']["pred_len"]*unit_len
        self.past_length = config['data_config']["past_len"] * unit_len
        self.target_dim=config['data_config']["target_cols"]
        self.seq_length = self.past_length + self.pred_length
        self.kill_dim = len(config['data_config']["feature_cols_mask2D"])
        self.kill_dim1D = len(config['data_config']["feature_cols_mask1D"])
        self.dim = len(config['data_config']["columns"])
        if config['data_config']["data_mode"] == 1:
            if mode == "train" or mode == "examine":
                start = config['data_config']["data_time"]["train"]["start"]
                end = config['data_config']["data_time"]["train"]["end"]
            else:
                # 直接获取past_length * unit_len长度的历史数据
                start_idx = data.index.get_loc(config['data_config']["data_time"][mode]["start"])
                start_idx = max(0, start_idx - self.past_length )
                start = data.index[start_idx].strftime("%Y-%m-%d %H:%M:%S")
                end = config['data_config']["data_time"][mode]["end"]

        if mode == "train" or mode == "examine":
            if config['data_config']["data_len"]["train"] == -1:
                start_len = len(data)
                end_len = (
                    config['data_config']["data_len"]["valid"]
                    + config['data_config']["data_len"]["test"]
                    + config['data_config']["data_len"]["shift"]
                ) * unit_len
            else:
                start_len = (
                    config['data_config']["data_len"]["train"]
                    + config['data_config']["data_len"]["valid"]
                    + config['data_config']["data_len"]["test"]
                    + config['data_config']["data_len"]["shift"]
                ) * unit_len
                end_len = start_len - config['data_config']["data_len"]["train"] * unit_len
        elif mode == "valid":
            start_len = (
                config['data_config']["data_len"]["valid"]
                + config['data_config']["data_len"]["test"]
                + config['data_config']["data_len"]["shift"]
            ) * unit_len + self.past_length
            end_len = (
                start_len
                - config['data_config']["data_len"]["valid"] * unit_len
                - self.past_length
            )
        else:
            start_len = (
                config['data_config']["data_len"]["test"]
                + config['data_config']["data_len"]["shift"]
            ) * unit_len + self.past_length
            end_len = (
                start_len
                - config['data_config']["data_len"]["test"] * unit_len
                - self.past_length
            )

        if config['data_config']["data_mode"] == 0:
            if end_len == 0:
                self.main_data = data.iloc[-start_len:]
            else:
                self.main_data = data.iloc[-start_len:-end_len]
            step = config['data_config']["training_step"]
        elif config['data_config']["data_mode"] == 1:
            self.main_data = data.loc[start:end]
            step = config['data_config']["training_step"]
        elif config['data_config']["data_mode"] == 2:
            # 使用比例划分训练集、验证集和测试集
            total_len = len(data)
            train_ratio, val_ratio, test_ratio = config['data_config']["train_val_test_ratio"]
            
            # 计算各部分的长度
            train_len = int(total_len * train_ratio)
            val_len = int(total_len * val_ratio)
            test_len = total_len - train_len - val_len  # 剩余部分作为测试集
            
            if mode == "train":
                # 训练集：从开始到train_len
                self.main_data = data.iloc[:train_len]
            elif mode == "valid":
                # 验证集：从train_len到train_len+val_len
                self.main_data = data.iloc[train_len:train_len+val_len]
            elif mode == "test":
                # 测试集：从train_len+val_len到结束
                self.main_data = data.iloc[train_len+val_len:]
            else:
                # 其他模式：使用全部数据
                self.main_data = data
            
            step = config['data_config']["training_step"]
        else:
            self.main_data = data.iloc[-start_len:]
            step = config['data_config']["training_step"]

        # 此处认为所有的数据都被观测到了，因此mask都为1
        self.mask_data = pd.DataFrame(np.ones_like(self.main_data.values)).to_numpy()
        self.main_data = self.main_data.to_numpy()


        if mode == "train":
            total = len(self.main_data)
            index_end = len(self.main_data) - self.seq_length + 1
            self.use_index = np.arange(0, total - self.seq_length + step, step)
        elif  mode=='examine':
        # elif mode == 'examine' or mode == 'test':
            step = unit_len
            total = len(self.main_data)
            index_end = len(self.main_data) - self.seq_length + 1
            # self.use_index = np.arange(0, total - self.seq_length + step, step)
            self.use_index = np.arange(0, total- self.seq_length+1, 1)
        else:
            index_end = len(self.main_data) - self.seq_length + 1
            self.use_index = np.arange(0, index_end, unit_len)
            # self.use_index = np.arange(0, index_end, 1)
        # 日内数据掩码
        # 此处要对齐，数据结构为电价+日内数据+日前数据
        self.kill_mask = np.ones([self.seq_length, self.dim])
        self.kill_mask[- 2*self.pred_length :, 1 : (1 + self.kill_dim)] = 0
        self.kill_mask[- self.pred_length :, (1 + self.kill_dim) : (1 + self.kill_dim+self.kill_dim1D)] = 0

    def __getitem__(self, orgindex):
        index = self.use_index[orgindex]  # 坐标转化
        target_mask = self.mask_data[
            index : index + self.seq_length
        ].copy()  # 目标掩码，告诉模型哪里是要预测的电价真值
        if self.target_dim == 1:
            target_mask[-self.pred_length :, 0] = 0.0
        else:
            target_mask[-self.pred_length :, :] = 0.0

        # 确保数据是连续的numpy数组，并且是float32类型
        observed_data = np.ascontiguousarray(self.main_data[index : index + self.seq_length] * self.kill_mask).astype(np.float32)
        observed_mask = np.ascontiguousarray(self.mask_data[index : index + self.seq_length]).astype(np.float32)
        gt_mask = np.ascontiguousarray(target_mask).astype(np.float32)
        timepoints = np.ascontiguousarray(np.arange(self.seq_length) * 1.0).astype(np.float32)
        feature_id = np.ascontiguousarray(np.arange(self.main_data.shape[1]) * 1.0).astype(np.float32)

        s = {
            "observed_data": observed_data,
            "observed_mask": observed_mask,
            "gt_mask": gt_mask,
            "timepoints": timepoints,
            "feature_id": feature_id,
        }

        return s

    def __len__(self):
        return len(self.use_index)


def time_shift(time_str, shift):
    time0 = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
    time0 = time0 + timedelta(days=shift)
    return time0.strftime("%Y-%m-%d %H:%M:%S")
