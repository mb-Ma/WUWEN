from data_provider.data_factory import data_provider
from models.TimeMixer import TimeMixer_architectures
from models.TimeMixer.units.tools import EarlyStopping, adjust_learning_rate, visual
from models.TimeMixer.units.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import argparse
import pytz
import pickle as pkl
from datetime import datetime
from utils.metrics import MAE_np, RMSE_np, MAPE_np, SPEARMAN_np, PEARSON_np,R2_np, SMAPE_np
warnings.filterwarnings('ignore')

class Exp_Basic(object):
    def __init__(self, config):
        self.configs=config
        self.model_config=config['model_config']
        self.training_config=config['training_config']
        self.data_config=config['data_config']
        # 合并所有配置项
        merged_config = {**self.model_config, **self.training_config,**self.data_config}
        def flatten_dict(d, parent_key=''):
            items = {}
            for k, v in d.items():
                new_key = f"{parent_key}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.update(flatten_dict(v, new_key + '_'))
                else:
                    items[new_key] = v
            return items

        merged_config = flatten_dict(merged_config)
        self.args = argparse.Namespace(**merged_config)
        #对于电价数据集，为了能够使用未来一天的日前特征，将原先的past_len推理pred_len修改为past_len+1推理pred_len，再在dataloader中进行相应的处理
        if self.args.data=='Electricity_shanxi' or self.args.data=='Electricity_shandong':
            self.args.past_len+=1
        self.model_dict = {
            # 'TimesNet': TimesNet,
            # 'Autoformer': Autoformer,
            # 'Transformer': Transformer,
            # 'Nonstationary_Transformer': Nonstationary_Transformer,
            # 'DLinear': DLinear,
            # 'FEDformer': FEDformer,
            # 'Informer': Informer,
            # 'LightTS': LightTS,
            # 'Reformer': Reformer,
            # 'ETSformer': ETSformer,
            # 'PatchTST': PatchTST,
            # 'Pyraformer': Pyraformer,
            # 'MICN': MICN,
            # 'Crossformer': Crossformer,
            # 'FiLM': FiLM,
            # 'iTransformer': iTransformer,
            # # 'Koopa': Koopa,
            # # 'TiDE': TiDE,
            # # 'FreTS': FreTS,
            # # 'MambaSimple': MambaSimple,
            'TimeMixer': TimeMixer_architectures,
            # 'TSMixer': TSMixer,
            # 'SegRNN': SegRNN,
            # 'TemporalFusionTransformer': TemporalFusionTransformer,
            # "SCINet": SCINet,
            # 'TimeXer': TimeXer
        }

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):

        device = torch.device(self.args.device)  # 直接使用 self.args.device
        print(f"Use GPU: {self.args.device}")
        # print(f"Available GPUs: {torch.cuda.device_count()}")
        # if self.args.use_gpu:
        #     os.environ["CUDA_VISIBLE_DEVICES"] = str(
        #         self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
        #     device = torch.device('cuda:{}'.format(self.args.gpu))
        #     print('Use GPU: cuda:{}'.format(self.args.gpu))
        # else:
        #     device = torch.device('cpu')
        #     print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

class timemixer(Exp_Basic):
    def __init__(self, config):
        super(timemixer, self).__init__(config)
        self.config=config
        self.model_config=config['model_config']
        self.training_config=config['training_config']
        self.data_config=config['data_config']
        # 合并所有配置项
        merged_config = {**self.model_config, **self.training_config,**self.data_config}
        def flatten_dict(d, parent_key=''):
            items = {}
            for k, v in d.items():
                new_key = f"{parent_key}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.update(flatten_dict(v, new_key + '_'))
                else:
                    items[new_key] = v
            return items

        merged_config = flatten_dict(merged_config)
        self.args = argparse.Namespace(**merged_config)
        #对于电价数据集，为了能够使用未来一天的日前特征，将原先的past_len推理pred_len修改为past_len+1推理pred_len，再在dataloader中进行相应的处理
        if self.args.data=='Electricity_shanxi' or self.args.data=='Electricity_shandong':
            self.args.past_len+=1
        print('args:')
        print(self.args)
        # 创建文件夹用于存储，文件夹名为 "数据集名字_时间"
        beijing_tz = pytz.timezone("Asia/Shanghai")
        current_time = (
            datetime.now().astimezone(beijing_tz).strftime("%Y-%m-%d_%H:%M:%S")
        )

        foldername = (
            self.config['training_config']["save_path"]
            +  self.config['data_config']["data_path"].split("/")[-2] + "/"
            + current_time
            + "/"
        )

        print("model folder:", foldername)
        foldername = os.path.join(config['root'], foldername)
        os.makedirs(foldername, exist_ok=True)
        self.foldername = foldername
    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float().to(torch.device(self.args.device))

        # if self.args.use_multi_gpu and self.args.use_gpu:
        #     model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -(self.args.pred_len*self.args.unit_len):, :]).float()
                dec_inp = torch.cat([batch_y[:, :48, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len*self.args.unit_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len*self.args.unit_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        path=self.foldername
        # path = os.path.join(self.args.checkpoints, setting)
        # if not os.path.exists(path):
        #     os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len*self.args.unit_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :int(self.args.label_len*self.args.unit_len), :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        if self.args.features=='MS':
                            f_dim = -1   #这里存疑，如果是S不应该也是f_dim=1么  #解决了，S只有一列
                            outputs = outputs[:, -self.args.pred_len*self.args.unit_len:, f_dim:]
                            batch_y = batch_y[:, -self.args.pred_len*self.args.unit_len:, f_dim:].to(self.device)
                        else:
                            outputs = outputs[:, -self.args.pred_len*self.args.unit_len:, :]
                            batch_y = batch_y[:, -self.args.pred_len*self.args.unit_len:, :].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    if self.args.features=='MS':
                        f_dim = -1  
                        outputs = outputs[:, -self.args.pred_len*self.args.unit_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len*self.args.unit_len:, f_dim:].to(self.device)
                    else:
                        outputs = outputs[:, -self.args.pred_len*self.args.unit_len:, :]
                        batch_y = batch_y[:, -self.args.pred_len*self.args.unit_len:, :].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'best_model_0.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        #训练结束执行测试,获取真实值和预测值
        print('————————————————————————————testing——————————————————————')

        self.test(test=0)

        return self.model

    def test(self,test=0):
        #直接执行推理
            

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(self.args.model_path))
            # self.model.load_state_dict(torch.load(os.path.join(self.args.model_path, 'best_model.pth')))
        test_data, test_loader = self._get_data(flag='test')
        preds = []
        trues = []
        folder_path = self.foldername
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len*self.args.unit_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :int(self.args.label_len*self.args.unit_len), :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len*self.args.unit_len:, :]
                batch_y = batch_y[:, -self.args.pred_len*self.args.unit_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if self.args.features == 'MS':
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)
        
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # 使用新的指标计算函数
        mae = MAE_np(preds, trues)
        rmse = RMSE_np(preds, trues)
        mape = MAPE_np(preds, trues)
        r2 = R2_np(preds, trues)
        spearman = SPEARMAN_np(preds, trues)
        pearson = PEARSON_np(preds, trues)
        smape = SMAPE_np(preds, trues)
        # 保留原有的 metric 函数用于兼容性
        mae_old, mse, rmse_old, mape_old, mspe = metric(preds, trues)
        
        print('MAE: {:.6f}, RMSE: {:.6f}, MAPE: {:.6f}, R2: {:.6f}, Spearman: {:.6f}, Pearson: {:.6f}'.format(mae, rmse, mape, r2, spearman, pearson))
        print('MSE: {:.6f}, MSPE: {:.6f}, SMAPE: {:.6f}'.format(mse, mspe, smape))
        f = open("result_timexer.txt", 'a')
        f.write(self.args.model_path + "  \n")
        f.write('MAE: {:.6f}, RMSE: {:.6f}, MAPE: {:.6f}, R2: {:.6f}, Spearman: {:.6f}, Pearson: {:.6f}, MSE: {:.6f}, MSPE: {:.6f}, SMAPE: {:.6f}'.format(mae, rmse, mape, r2, spearman, pearson, mse, mspe, smape))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe, r2, spearman, pearson, smape]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        # 额外保存result.npz文件
        np.savez(
            os.path.join(folder_path, "result.npz"),
            real_y=trues,
            pred_y=preds,
        )