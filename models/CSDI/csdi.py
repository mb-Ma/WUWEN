#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @FileName      : Diffusion
# @Time          : 2024-10-31 15:16:50
# @Author        : ZhangHonglin
# @Email         : honglin-24@mails.tsinghua.edu.cn
# @description   : 扩散模型interface
"""

import pandas as pd
import pickle
import numpy as np
import torch
import pytz
import shutil
import os

from tqdm import tqdm
from torch.optim import Adam
from datetime import datetime
from torch.utils.data import DataLoader
from datetime import datetime, timedelta
from models.CSDI.csdi_model_architecture import Forecasting
from data_provider.csdi_dataloader import Diffusion_Dataset
from utils.metrics import MAE_torch, RMSE_torch, MAPE_torch, MAE_np, RMSE_np, MAPE_np


def time_shift(time_str, shift):
    time0 = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
    time0 = time0 + timedelta(days=shift)
    return time0.strftime("%Y-%m-%d %H:%M:%S")

class csdi:
    def __init__(self, configs) -> None:
        # 处理main.py的配置结构，转换为CSDI需要的格式
        if not isinstance(configs, dict):
            # 如果是Hydra配置对象，转换为字典
            configs = dict(configs)
        
        # 添加CSDI需要的字段
        if 'root' not in configs:
            configs['root'] = './'
        if 'config_path' not in configs:
            configs['config_path'] = 'conf/baseline/csdi_UTS.yaml'
        
        # 确保配置结构完整
        if 'training_config' not in configs:
            configs['training_config'] = configs
        if 'data_config' not in configs:
            configs['data_config'] = configs
        if 'model_config' not in configs:
            configs['model_config'] = configs
        
        self.config = configs

        # 设置所有随机数种子
        seed = configs['training_config'].get('seed', 2024)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.model = None

        self.mean = None  # 训练集均值
        self.std = None  # 训练集方差

        self.foldername = None  # 结果存储位置

        self.target_dim = len(configs['data_config']["columns"])  # 预测目标的维度
        self.device = configs['training_config']["device"]  # 训练设备

        self.dataloader = {"train": None, "valid": None, "test": None}  # 数据加载器

        self.data_info = []  # 存储数据的列名信息，便于复现
        self.norm_info = []  # 存储各次训练的归一化信息
        self.data_process = []  # 存储数据处理操作

        self.roll_run = configs['training_config']["roll_run"]  # 是否要进行滚动训练
        self.roll_step = configs['training_config']["roll_step"]  # 滚动训练的单次平移长度为多少个unit_step

        self.test_index = None  # 测试集的时间戳

        # 创建文件夹用于存储，文件夹名为 "数据集名字_时间"
        beijing_tz = pytz.timezone("Asia/Shanghai")
        current_time = datetime.now().astimezone(beijing_tz).strftime("%Y-%m-%d_%H:%M:%S")
        foldername = (
            self.config['training_config']["save_path"]
            + self.config['data_config']["data_path"].split("/")[-2] + "/"
            + current_time
            + "/"
        )
        print("model folder:", foldername)
        os.makedirs(foldername, exist_ok=True)
        self.foldername = foldername

        # 转存三个配置文件，便于复现
        def copy_file_to_folder(file_path, folder_name):
            file_name = os.path.basename(file_path)
            destination_path = os.path.join(folder_name, file_name)
            shutil.copy(file_path, destination_path)

        copy_file_to_folder(os.path.join(configs['root'], configs['config_path']), self.foldername)

    def init_model(self) -> None:
        """
        初始化模型，也可以用于建立新的模型
        """
        # 直接创建模型，CSDI模型架构会动态修改配置字典
        self.model = Forecasting(self.config, self.device, self.target_dim).to(self.config['training_config']["device"])

        def print_model_info(model):
            total_params = 0
            for _, param in model.named_parameters():
                if param.requires_grad:
                    num_params = param.numel()
                    total_params += num_params
            print(f"Total Trainable Parameters: {total_params}")

        print_model_info(self.model)
    
    def process_data(self, rolls_step=0) -> None:
        """数据加载和处理
        构造特征或构造dataset，在初始化和进行滚动测试时可以使用
        raw_dataset -> processed_dataset
        """
        file_path = self.config['data_config']["data_path"]
       
        raw_dataset = pd.read_csv(file_path)
        raw_dataset.set_index('timestamp',inplace=True)
        raw_dataset.index = pd.to_datetime(raw_dataset.index)
        
        # 重新索引原始数据列
        raw_dataset = raw_dataset.reindex(self.config['data_config']['target_cols']+
                                        self.config['data_config']['feature_cols_mask2D']+
                                        self.config['data_config']['feature_cols_mask1D']+
                                        self.config['data_config']['feature_cols_da'], axis=1)
  
        # 校对数据的列名是否对齐
        config_temp = self.config

        # 滚动测试时调整切割时间戳
        config_temp['data_config']["data_len"]["shift"] = (
            self.config['data_config']["data_len"]["shift"] - rolls_step
        )
        config_temp['data_config']["data_time"]["train"]["start"] = time_shift(
            self.config['data_config']["data_time"]["train"]["start"], rolls_step
        )
        config_temp['data_config']["data_time"]["train"]["end"] = time_shift(
            self.config['data_config']["data_time"]["train"]["end"], rolls_step
        )
        config_temp['data_config']["data_time"]["valid"]["start"] = time_shift(
            self.config['data_config']["data_time"]["valid"]["start"], rolls_step
        )
        config_temp['data_config']["data_time"]["valid"]["end"] = time_shift(
            self.config['data_config']["data_time"]["valid"]["end"], rolls_step
        )
        config_temp['data_config']["data_time"]["test"]["start"] = time_shift(
            self.config['data_config']["data_time"]["test"]["start"], rolls_step
        )
        config_temp['data_config']["data_time"]["test"]["end"] = time_shift(
            self.config['data_config']["data_time"]["test"]["end"], rolls_step
        )

        unit_len = config_temp['data_config']["unit_len"]
        if config_temp['data_config']["data_mode"] == 0:
            end_time = config_temp['data_config']["data_len"]["end"]
            processed_dataset = raw_dataset.loc[:end_time]
            if config_temp['data_config']["data_len"]["train"] == -1:
                if config_temp['data_config']["data_len"]["shift"] != 0:
                    processed_dataset = processed_dataset.iloc[
                        : -config_temp['data_config']["data_len"]["shift"] * unit_len
                    ]
            else:
                total_len = (
                    config_temp['data_config']["data_len"]["train"]
                    + config_temp['data_config']["data_len"]["valid"]
                    + config_temp['data_config']["data_len"]["test"]
                    + config_temp['data_config']["data_len"]["shift"]
                ) * unit_len

                shift = config_temp['data_config']["data_len"]["shift"]
                if shift != 0:
                    processed_dataset = processed_dataset.iloc[
                        -total_len : -shift * unit_len
                    ]
                else:
                    processed_dataset = processed_dataset.iloc[-total_len:]

                train_temp_len = (
                    total_len - config_temp['data_config']["data_len"]["train"] * unit_len
                )
                train_dataset = processed_dataset.iloc[-total_len:-train_temp_len]

            test_start = (
                len(processed_dataset)
                - (
                    config_temp['data_config']["data_len"]["test"] +
                    config_temp['data_config']["data_len"]["shift"]
                )
                * unit_len
            )
            test_end = (
                len(processed_dataset)
                - (
                    config_temp['data_config']["data_len"]["shift"]
                )
                * unit_len
            )
            self.test_index = processed_dataset.iloc[test_start:test_end].index
        else:
            times = [
                config_temp['data_config']["data_time"]["train"]["start"],
                config_temp['data_config']["data_time"]["train"]["end"],
                config_temp['data_config']["data_time"]["valid"]["start"],
                config_temp['data_config']["data_time"]["valid"]["end"],
                config_temp['data_config']["data_time"]["test"]["start"],
                config_temp['data_config']["data_time"]["test"]["end"],
            ]
            times_datetime = [
                datetime.strptime(value, "%Y-%m-%d %H:%M:%S") for value in times
            ]
            earliest_time = min(times_datetime)
            latest_time = max(times_datetime)
            processed_dataset = raw_dataset.loc[earliest_time:latest_time]

            train_dataset = processed_dataset.loc[
                config_temp['data_config']["data_time"]["train"][
                    "start"
                ] : config_temp['data_config']["data_time"]["train"]["end"]
            ]

            self.test_index = processed_dataset.loc[
                config_temp['data_config']["data_time"]["test"][
                    "start"
                ] : config_temp['data_config']["data_time"]["test"]["end"]
            ].index

        # 进行归一化处理，并且存储归一化值，以便于再次使用模型进行推理
        processed_dataset = self.normalize(train_dataset, processed_dataset)

        self.generate_dataloader(processed_dataset, config_temp)

    def normalize(self, train_dataset, processed_dataset) -> None:
        self.data_process.append("Normalization")

        if self.config['training_config']["model_path"] == "":
            col_mean = np.mean(train_dataset.to_numpy(), axis=0, keepdims=True)
            col_std = np.std(train_dataset.to_numpy(), axis=0, keepdims=True)
            col_std[col_std == 0] = 1

            # 将归一化参数存储到模型中
            self.model.mean = col_mean
            self.model.std = col_std

        else:
            # 如果已经加载好了模型，则直接采用模型本身的均值和方差进行归一化
            col_mean = self.model.mean
            col_std = self.model.std

        self.mean = self.model.mean
        self.std = self.model.std

        return (processed_dataset - col_mean) / col_std

    def generate_dataloader(self, processed_dataset, config):
        """
        生成数据加载器
        """
        train_dataset = Diffusion_Dataset(config, processed_dataset, "train")
        valid_dataset = Diffusion_Dataset(config, processed_dataset, "valid")
        test_dataset = Diffusion_Dataset(config, processed_dataset, "test")

        self.dataloader["train"] = DataLoader(
            train_dataset,
            batch_size=config['training_config']["batch_size"],
            shuffle=True,
        )
        self.dataloader["valid"] = DataLoader(
            valid_dataset,
            batch_size=config['training_config']["batch_size"],
            shuffle=False,
        )
        self.dataloader["test"] = DataLoader(
            test_dataset, batch_size=config['training_config']["batch_size"], shuffle=False
        )

    def train(self) -> None:
        """训练模型"""
        valid_epoch_interval = 1

        for k in range(self.roll_run + 1):
            print(f"-----------第{k}次训练和推理-------------")

            self.init_model()  # 初始化一个新的模型
            self.process_data(k)  # 处理数据，一般为归一化，可以添加特征工程

            # 配置优化器
            config_temp = self.config['training_config']
            optimizer = Adam(
                self.model.parameters(),
                lr=config_temp["lr"],
                weight_decay=1e-6,
            )
            p1 = int(config_temp["p1"])
            p2 = int(config_temp["p2"])
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[p1, p2], gamma=config_temp["lr_gamma"]
            )

            if self.foldername != "":
                output_path = self.foldername + f"model_{k}.pth"  # 存储最终训练模型
                best_path = (
                    self.foldername + f"best_model_{k}.pth"
                )  # 存储验证效果最优对应的模型

            train_losses = []  # 存储训练过程的loss
            valid_losses = []  # 存储验证过程的loss

            best_valid_loss = 1e10  # 存储最优的验证loss
            for epoch_no in range(config_temp["epochs"]):
                avg_loss = 0
                self.model.train()

                with tqdm(
                    self.dataloader["train"], mininterval=5.0, maxinterval=50.0
                ) as it:
                    for batch_no, train_batch in enumerate(it, start=1):
                        optimizer.zero_grad()  # 清除优化器梯度
                        
                        # 计算 loss 并进行模型更新
                        loss = self.model(train_batch)
                        
                        # 检查loss是否为NaN
                        if torch.isnan(loss):
                            print(f"Warning: NaN loss detected at epoch {epoch_no}, batch {batch_no}")
                            continue  # 跳过这个batch
                        
                        loss.backward()
                        
                        # # 梯度裁剪，防止梯度爆炸
                        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        
                        avg_loss += loss.item()
                        optimizer.step()
                        
                        # 进度条显示信息
                        it.set_postfix(
                            ordered_dict={
                                "loss": loss.item(),
                                "avg_epoch_loss": avg_loss / batch_no,
                                "epoch": epoch_no,
                            },
                            refresh=False,
                        )

                    lr_scheduler.step()

                train_losses.append(avg_loss / batch_no)

                if (
                    self.dataloader["valid"] is not None
                    and (epoch_no + 1) % valid_epoch_interval == 0
                ):
                    output = self.predict(
                        self.dataloader["valid"],
                        self.config['training_config']["valid_nsample"],
                    )
                    self.valid_test(output)
                    (
                        all_target,
                        all_evalpoint,
                        all_observed_point,
                        all_observed_time,
                        all_generated_samples,
                    ) = output
                    
                    mae = self.evaluate(
                        MAE_torch,
                        all_target, all_evalpoint, all_observed_point, 
                        all_observed_time, all_generated_samples,
                        self.mean, self.std
                    )
                    
                    rmse = self.evaluate(
                        RMSE_torch,
                        all_target, all_evalpoint, all_observed_point, 
                        all_observed_time, all_generated_samples,
                        self.mean, self.std
                    )

                    # 存储验证效果最好的模型
                    if best_valid_loss > mae:
                        best_valid_loss = mae
                        print(
                            "\n best loss is updated to ",
                            mae,
                            "at",
                            epoch_no,
                        )
                        if self.foldername != "":
                            self.save_model(self.model, best_path)
                valid_losses.append(mae)
            self.load_model(best_path)
            # 训练完成后进行测试集推理
            output = self.predict(self.dataloader["test"], self.config['model_config']["nsample"])
            self.output_save(output)
            (
                all_target,
                all_evalpoint,
                all_observed_point,
                all_observed_time,
                all_generated_samples,
            ) = output

            mae = self.evaluate(
                MAE_torch,
                all_target,
                all_evalpoint,
                all_observed_point,
                all_observed_time,
                all_generated_samples,
                self.mean,
                self.std,
            )

            rmse = self.evaluate(
                RMSE_torch,
                all_target,
                all_evalpoint,
                all_observed_point,
                all_observed_time,
                all_generated_samples,
                self.mean,
                self.std,
            )

            mape = self.evaluate(
                MAPE_torch,
                all_target,
                all_evalpoint,
                all_observed_point,
                all_observed_time,
                all_generated_samples,
                self.mean,
                self.std,
            )

            print(f"第{k}次测试集MAE：", mae)
            print(f"第{k}次测试集RMSE：", rmse)
            print(f"第{k}次测试集MAPE：", mape)

            # 存储单次训练出的模型
            self.save_model(self.model, output_path)

            # 存储 loss 结果
            losses_path = self.foldername + f"losses_{k}.pkl"
            losses_data = {"train_losses": train_losses, "valid_losses": valid_losses}
            with open(losses_path, "wb") as f:
                pickle.dump(losses_data, f)

    def predict(self, dataloader, nsample):
        """
        根据给定的数据加载器，使用当前的模型进行批量预测
        """

        with torch.no_grad():
            self.model.eval()

            all_target = []
            all_observed_point = []
            all_observed_time = []
            all_evalpoint = []
            all_generated_samples = []
            with tqdm(dataloader, mininterval=5.0, maxinterval=50.0) as it:
                for batch_no, test_batch in enumerate(it, start=1):
                    output = self.model.evaluate(test_batch, nsample)

                    samples, c_target, eval_points, observed_points, observed_time = (
                        output
                    )
                    samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                    c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                    eval_points = eval_points.permute(0, 2, 1)
                    observed_points = observed_points.permute(0, 2, 1)

                    all_target.append(c_target.cpu())
                    all_evalpoint.append(eval_points.cpu())
                    all_observed_point.append(observed_points.cpu())
                    all_observed_time.append(observed_time.cpu())
                    all_generated_samples.append(samples.cpu())

                    it.set_postfix(
                        ordered_dict={
                            "batch_no": batch_no,
                        },
                        refresh=True,
                    )

        all_target = torch.cat(all_target, dim=0)
        all_evalpoint = torch.cat(all_evalpoint, dim=0)
        all_observed_point = torch.cat(all_observed_point, dim=0)
        all_observed_time = torch.cat(all_observed_time, dim=0)
        all_generated_samples = torch.cat(all_generated_samples, dim=0)

        return (
            all_target,
            all_evalpoint,
            all_observed_point,
            all_observed_time,
            all_generated_samples,
        )
    def test(self):
        """
        根据给定的数据加载器，使用当前的模型进行批量预测
        """

        self.load_model()
        self.process_data(0)
        dataloader=self.dataloader["test"]
        nsample=self.config['model_config']["nsample"]
        with torch.no_grad():
            self.model.eval()

            all_target = []
            all_observed_point = []
            all_observed_time = []
            all_evalpoint = []
            all_generated_samples = []
            with tqdm(dataloader, mininterval=5.0, maxinterval=50.0) as it:
                for batch_no, test_batch in enumerate(it, start=1):
                    output = self.model.evaluate(test_batch, nsample)

                    samples, c_target, eval_points, observed_points, observed_time = (
                        output
                    )
                    samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                    c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                    eval_points = eval_points.permute(0, 2, 1)
                    observed_points = observed_points.permute(0, 2, 1)

                    all_target.append(c_target.cpu())
                    all_evalpoint.append(eval_points.cpu())
                    all_observed_point.append(observed_points.cpu())
                    all_observed_time.append(observed_time.cpu())
                    all_generated_samples.append(samples.cpu())

                    it.set_postfix(
                        ordered_dict={
                            "batch_no": batch_no,
                        },
                        refresh=True,
                    )

        all_target = torch.cat(all_target, dim=0)
        all_evalpoint = torch.cat(all_evalpoint, dim=0)
        all_observed_point = torch.cat(all_observed_point, dim=0)
        all_observed_time = torch.cat(all_observed_time, dim=0)
        all_generated_samples = torch.cat(all_generated_samples, dim=0)
        output=(all_target,all_evalpoint,all_observed_point,all_observed_time,all_generated_samples)
        self.output_save(output)
        return (
            all_target,
            all_evalpoint,
            all_observed_point,
            all_observed_time,
            all_generated_samples,
        )
    def evaluate(self, metric_func, *data):
        """
        对预测结果进行评测，使用utils.metrics中的指标函数
        """
        # 解析输入数据
        all_target, all_evalpoint, all_observed_point, all_observed_time, all_generated_samples = data[:5]
        
        # 获取归一化参数
        mean = data[5] if len(data) > 5 else self.mean
        std = data[6] if len(data) > 6 else self.std
        
        # 去归一化
        targets = all_target * std + mean
        samples = all_generated_samples * std + mean
        
        # 提取预测部分（最后pred_len个时间步）
        pred_length = int(self.config['data_config']["unit_len"] * self.config['data_config']['pred_len'])
        targets = targets[:, -pred_length:, 0]  # (B, pred_len)
        samples = samples[:, :, -pred_length:, 0]  # (B, nsample, pred_len)
        
        # 使用中位数作为预测值（第50个百分位数）
        if samples.shape[1] > 1:  # 如果有多个样本
            pred_values = torch.quantile(samples, 0.5, dim=1)  # (B, pred_len)
        else:
            pred_values = samples.squeeze(1)  # (B, pred_len)
        
        # 确保数据是连续的
        targets = targets.contiguous()
        pred_values = pred_values.contiguous()
        
        # 调用指标函数
        if metric_func in [MAE_torch, RMSE_torch, MAPE_torch]:
            # 使用PyTorch版本的指标函数
            return metric_func(pred_values, targets)
        elif metric_func in [MAE_np, RMSE_np, MAPE_np]:
            # 使用NumPy版本的指标函数
            return metric_func(pred_values.cpu().numpy(), targets.cpu().numpy())
        else:
            # 兼容原有的指标函数调用方式
            return metric_func(*data)

    def valid_test(self, output):
        (
            all_target,
            all_evalpoint,
            all_observed_point,
            all_observed_time,
            all_generated_samples,
        ) = output
        targets = all_target * self.std + self.mean
        samples = all_generated_samples * self.std + self.mean
        targets = targets[:, -int(self.config['data_config']["unit_len"]*self.config['data_config']['pred_len']) :, 0]
        targets = targets.reshape(-1).unsqueeze(1).numpy()
        samples = samples[:, :, -int(self.config['data_config']["unit_len"]*self.config['data_config']['pred_len']) :, 0]

        qlist = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        quantiles = []
        for q in qlist:
            quantiles.append(torch.quantile(samples, q, dim=1).numpy())

        samples = np.transpose(samples, (0, 2, 1))
        samples = samples.reshape(-1, self.config['training_config']["valid_nsample"])
        samples = np.concatenate(
            (quantiles[4].reshape(-1)[:, np.newaxis], samples.numpy()), axis=1
        )
        
        # 使用新的指标函数计算MAE和RMSE
        pred_values = samples[:, 0]  # 使用中位数预测值
        mae = MAE_np(pred_values, targets.flatten())
        rmse = RMSE_np(pred_values, targets.flatten())
        
        print(f"验证集去归一化MAE: {mae}, RMSE: {rmse}")
        
        datas = np.concatenate((targets, samples), axis=1)
        cols = ["Real", "Prediction"] + [
            f"Sample_{x}" for x in range(self.config['training_config']["valid_nsample"])
        ]
        output_pd = pd.DataFrame(datas, columns=cols)
        times = pd.DataFrame(range(len(targets)), columns=["timestamp"])
        output_pd = pd.concat((times, output_pd), axis=1)
        with open(self.foldername + "valid_results.pkl", 'wb') as file:
            pickle.dump(output_pd, file)
    def output_save(self, output) -> None:
        """
        存储标准的pd格式的预测数据（以及其原始文件，方便更进一步分析）
        时间 + 预测 + 真值
        """
        (
            all_target,
            all_evalpoint,
            all_observed_point,
            all_observed_time,
            all_generated_samples,
        ) = output
        if self.config['training_config']['inverse']:
            targets = all_target * self.std + self.mean
            samples = all_generated_samples * self.std + self.mean
        else:
            targets = all_target
            samples = all_generated_samples

        targets = targets[:, -int(self.config['data_config']["unit_len"]*self.config['data_config']['pred_len']) :, 0]
        targets = targets.reshape(-1).unsqueeze(1).numpy()
        samples = samples[:, :, -int(self.config['data_config']["unit_len"]*self.config['data_config']['pred_len']) :, 0]

        qlist = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        quantiles = []
        for q in qlist:
            quantiles.append(torch.quantile(samples, q, dim=1).numpy())

        samples = np.transpose(samples, (0, 2, 1))
        samples = samples.reshape(-1, self.config['model_config']["nsample"])
        samples = np.concatenate(
            (quantiles[4].reshape(-1)[:, np.newaxis], samples.numpy()), axis=1
        )
        
        # 使用新的指标函数计算MAE和RMSE
        pred_values = samples[:, 0]  # 使用中位数预测值
        mae = MAE_np(pred_values, targets.flatten())
        rmse = RMSE_np(pred_values, targets.flatten())
        
        print(f"测试集MAE: {mae}, RMSE: {rmse}")
        
        datas = np.concatenate((targets, samples), axis=1)
        cols = ["Real", "Prediction"] + [
            f"Sample_{x}" for x in range(self.config['model_config']["nsample"])
        ]
        output_pd = pd.DataFrame(datas, columns=cols)
        times = pd.DataFrame(self.test_index, columns=["timestamp"])
        output_pd = pd.concat((times, output_pd), axis=1)
        with open(self.foldername + "results.pkl", 'wb') as file:
            pickle.dump(output_pd, file)

        # 存储原始数据
        with open(
            self.foldername
            + "raw_results_nsample"
            + str(self.config['model_config']["nsample"])
            + ".pkl",
            "wb",
        ) as f:
            pickle.dump(
                [
                    all_generated_samples,
                    all_target,
                    all_evalpoint,
                    all_observed_point,
                    all_observed_time,
                    self.std,
                    self.mean,
                ],
                f,
            )

    def save_model(self, model, save_path) -> None:
        """
        保存模型参数和归一化参数
        """
        mean = self.mean
        std = self.std
        param = {
            'model': model.state_dict(),
            'mean': mean,
            'std': std
        }
        torch.save(param, save_path)

    def load_model(self, best_model_path=None) -> None:
        """
        加载模型参数
        """
        model_path = self.config['training_config'].get("model_path", None)
        
        # 如果提供了best_model_path，说明是训练结束后加载最优模型
        if best_model_path is not None:
            model_path = best_model_path
            print(f"训练结束，加载最优模型: {model_path}")
        elif model_path is None or not os.path.exists(model_path):
            raise ValueError(f"模型路径无效: {model_path}")
        self.model = Forecasting(self.config, self.device, self.target_dim).to(self.config['training_config']["device"])
        param = torch.load(model_path, map_location=self.config['training_config']["device"], weights_only=False)
        self.model.mean = param['mean']
        self.model.std = param['std']
        self.model.load_state_dict(param['model'])
        self.model.to(self.config['training_config']["device"])



