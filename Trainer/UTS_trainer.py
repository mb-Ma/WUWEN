from Trainer.basic_trainer import Basic_Trainer
from torch.optim import lr_scheduler
import torch
import torch.nn as nn
from torch import optim
from utils.metrics import Metrics
from tqdm import tqdm
import numpy as np
from loguru import logger as log
import copy

import warnings
warnings.filterwarnings('ignore')

def _log_metric(metrics: dict, prefix: str):
    if not isinstance(metrics, dict):
        raise ValueError("metrics should be dict format")
    try:
        formatted = ", ".join(f"{k}={v:.3f}" for k, v in metrics.items())
    except Exception as e:
        raise ValueError(f"format error: {e}")
    return f"{prefix}{formatted}"

class UTS_trainer(Basic_Trainer):
    def __init__(self, model, scaler, args):
        super(UTS_trainer, self).__init__(model, args)
        self.scaler = scaler

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self, loss_name='MSE'):
        if loss_name == 'MSE':
            return nn.MSELoss()
        elif loss_name == 'MAE':
            return nn.L1Loss()
        else:
            raise NotImplementedError("No required loss function")
    
    def train(self, train_loader, valid_loader):
        # metric config 
        _v_metric = {_: 0 for _ in ["rmse", "mae", "mape"]}

        # config early stopping
        best_loss = float("inf")
        not_improved_count = 0
        best_model = None
        
        # optimizer
        optimizer = self._select_optimizer()
        # the scheduler for the learning rate.
        train_steps = len(train_loader)
        scheduler = lr_scheduler.OneCycleLR(optimizer=optimizer,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.epochs,
                                            max_lr=self.args.learning_rate)
        # loss function
        loss_func = self._select_criterion(self.args.loss_name)
        self.model.train()
        for epoch in range(self.args.epochs):
            # clear train_loss cache
            train_loss = [] # store each iteration's loss
            with tqdm(train_loader, desc=f"Epoch {epoch}") as tq:
                _log = {}
                for batch in tq:
                    optimizer.zero_grad()
                    batch_x, x_time, batch_y, y_time = batch
                    x_time = x_time.to(self.device)
                    y_time = y_time.to(self.device)
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    pred_y = self.model((batch_x, x_time))
                    # if reverse the prediction to original dimension
                    if not self.args.is_norm_loss:
                        if self.args.if_col_norm:
                            pred_y = self.scaler.inver_transform_col(
                                pred_y
                            )
                            batch_y = self.scaler.inver_transform_col(batch_y)
                        else:
                            pred_y = self.scaler.inver_transform(pred_y)
                            batch_y = self.scaler.inver_transform(batch_y)
                    
                    loss = loss_func(pred_y, batch_y)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    train_loss.append(loss.detach().item())
                    _loss = {"loss": np.mean(train_loss)}
                    _log = {**_loss, **_v_metric} # unpack 
                    tq.set_postfix(_log)

            if epoch % self.args.skip_epoch == 0 and valid_loader is not None:
                    log.info(f"Epoch {epoch} result is: {_log}")
                    _v_metric = self.valid(valid_loader)

            if self.args.early_stoping:
                if _v_metric["mae"] < best_loss:
                    best_loss = _v_metric["mae"]
                    best_model = copy.deepcopy(self.model.state_dict())
                    not_improved_count = 0
                else:
                    not_improved_count += 1
                    if not_improved_count >= self.args.patience:
                        log.info(f"Stop training at epoch {epoch}")
                        self.model.load_state_dict(best_model)
                        break

        return self.model

    def valid(self, valid_loader):
        all_preds = [] # just for convenience, not the efficient method
        all_labels = []
        with torch.no_grad():
            for batch in valid_loader:
                batch_x, x_time, batch_y, y_time = batch
                x_time = x_time.to(self.device)
                y_time = y_time.to(self.device)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                pred_y = self.model((batch_x, x_time))
                # if reverse the prediction to original dimension
                if self.args.if_col_norm:
                    pred_y = self.scaler.inver_transform_col(
                        pred_y.cpu()
                    )
                    batch_y = self.scaler.inver_transform_col(batch_y.cpu())
                else:
                    pred_y = self.scaler.inver_transform(pred_y)
                    batch_y = self.scaler.inver_transform(batch_y)
                
                all_preds.append(pred_y.cpu())
                all_labels.append(batch_y.cpu())
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        mae, rmse, mape = Metrics(all_preds, all_labels)
        
        metrics = {"mae":mae.item(), "rmse":rmse.item(), "mape": mape.item()}
        _log = " ".join(
                [f"{name}: {metric:.3f}" for name, metric in metrics.items()]
            )
        log.info(_log)
        
        return metrics

    def infer(self, test_loader):
        self.model.eval()

        all_preds = [] # just for convenience, not the efficient method
        all_labels = []

        with torch.no_grad():
            for batch in test_loader:
                batch_x, x_time, batch_y, y_time = batch
                x_time = x_time.to(self.device)
                y_time = y_time.to(self.device)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                pred_y = self.model((batch_x, x_time))

                # if reverse the prediction to original dimension
                if self.args.if_col_norm:
                    pred_y = self.scaler.inver_transform_col(
                        pred_y.cpu()
                    )
                    batch_y = self.scaler.inver_transform_col(batch_y.cpu())
                else:
                    pred_y = self.scaler.inver_transform(pred_y)
                    batch_y = self.scaler.inver_transform(batch_y)
                all_preds.append(pred_y.cpu())
                all_labels.append(batch_y.cpu())
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        mae, rmse, mape = Metrics(all_preds, all_labels)
        metrics = {"mae":mae.item(), "rmse":rmse.item(), "mape": mape.item()}
        _log = _log_metric(metrics, prefix="Testset's result is: ")
        log.info(_log)

        return metrics, all_labels, all_preds

if __name__=="__main__":
    args = {'gpu': 0}
    mts = UTS_trainer(args)