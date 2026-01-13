from Trainer.basic_trainer import Basic_Trainer
from torch.optim import lr_scheduler
import torch
import torch.nn as nn
from torch import optim
from utils.metrics import Metrics
from tqdm import tqdm
import numpy as np
from loguru import logger as log
import pandas as pd
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

class DILATELoss(nn.Module):
    """
    DILATE loss: combination of shape term (smooth-DTW) and temporal term (TDI) for time-series forecasting.
    
    Args:
        alpha: float in [0,1], weight between shape loss and temporal loss.
        gamma: float >0, smoothing parameter for soft‐DTW.
        normalize: bool, whether to normalize series before computing.
    """
    def __init__(self, alpha=0.5, gamma=0.01, normalize=True):
        super().__init__()
        assert 0.0 <= alpha <= 1.0
        self.alpha = alpha
        self.gamma = gamma
        self.normalize = normalize

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: Tensor of shape (batch_size, horizon) or (batch_size, horizon, channels)
            y_true: Tensor of same shape as y_pred
        Returns:
            loss: scalar tensor
        """
        # assume 2-D (batch, T) for simplicity; extend to multi-channel if needed
        if y_pred.ndim == 3:
            # collapse channels by summing losses
            batch_size, T, C = y_pred.shape
            loss = 0.0
            for c in range(C):
                loss = loss + self._compute_loss(y_pred[..., c], y_true[..., c])
            loss = loss / C
            return loss
        else:
            return self._compute_loss(y_pred, y_true)

    def _compute_loss(self, y_pred, y_true):
        # optionally normalize each sample to zero mean/unit std to avoid scale issues
        if self.normalize:
            mu = y_true.mean(dim=1, keepdim=True)
            sigma = y_true.std(dim=1, keepdim=True) + 1e-8
            y_true_n = (y_true - mu) / sigma
            y_pred_n = (y_pred - mu) / sigma
        else:
            y_true_n = y_true
            y_pred_n = y_pred

        # shape loss: smooth DTW between y_pred_n and y_true_n
        shape_loss = self._soft_dtw(y_pred_n, y_true_n, gamma=self.gamma)

        # temporal loss: temporal distortion index (TDI) – penalizes mis‐alignment in time
        temporal_loss = self._temporal_loss(y_pred_n, y_true_n, gamma=self.gamma)

        # combined loss
        loss = self.alpha * shape_loss + (1.0 - self.alpha) * temporal_loss
        return loss / (y_pred.size(1) ** 2)

    def _soft_dtw(self, X, Y, gamma):
        """
        Compute soft-DTW between two sequences X and Y both of size (batch_size, T).
        Returns mean soft-DTW over batch.
        """
        batch_size, T = X.shape
        # compute cost matrix: shape (batch, T, T)
        # cost[i,j] = (X[:,i] - Y[:,j])^2
        X_exp = X.unsqueeze(2)  # (batch, T, 1)
        Y_exp = Y.unsqueeze(1)  # (batch, 1, T)
        D = (X_exp - Y_exp) ** 2  # (batch, T, T)

        # initialize R matrix for dynamic programming
        R = torch.zeros((batch_size, T+2, T+2), device=X.device, dtype=X.dtype) + float('inf')
        R[:, 0, 0] = 0.0

        # for numeric stability, we can pad indices starting at 1
        for i in range(1, T+1):
            for j in range(1, T+1):
                r0 = -R[:, i-1, j-1]
                r1 = -R[:, i-1, j]
                r2 = -R[:, i, j-1]
                rmax = torch.maximum(torch.maximum(r0, r1), r2)
                softmin = - (gamma * (torch.log(
                    torch.exp((r0 - rmax)/gamma) + torch.exp((r1 - rmax)/gamma) + torch.exp((r2 - rmax)/gamma)
                ) + rmax / gamma))
                R[:, i, j] = D[:, i-1, j-1] + softmin

        # soft‐DTW value is R[:, T, T]
        sdtw = R[:, T, T]
        return sdtw.mean()

    def _temporal_loss(self, X, Y, gamma):
        """
        Temporal loss term: sums over all pairs (i,j) cost * alignment weight,
        where cost = |i-j| and alignment weight = soft‐alignment matrix W computed in soft‐DTW.
        This penalizes time shifts.
        """
        batch_size, T = X.shape
        # compute cost matrix for value differences and time differences
        X_exp = X.unsqueeze(2)  # (batch, T, 1)
        Y_exp = Y.unsqueeze(1)  # (batch, 1, T)
        D = (X_exp - Y_exp) ** 2  # (batch, T, T)
        # time difference matrix
        time_mat = torch.abs(torch.arange(T, device=X.device).unsqueeze(1) - torch.arange(T, device=X.device).unsqueeze(0)).float()
        time_mat = time_mat.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, T, T)

        # compute soft‐alignment weights W via soft‐DTW “gradients”
        # Simplified version: W_ij ∝ exp(−D_ij / gamma)
        W = torch.exp(- D / gamma)
        W = W / (W.sum(dim=(1,2), keepdim=True) + 1e-8)  # normalize per batch

        # temporal error is sum_{i,j} W_ij * time_mat_ij
        temp_err = (W * time_mat).sum(dim=(1,2))
        return temp_err.mean()

class TweedieLoss(nn.Module):
    """
    Tweedie Loss for 1 < p < 2 (Compound Poisson-Gamma)
    """
    def __init__(self, power=1.5, eps=1e-6):
        super().__init__()
        self.power = power
        self.eps = eps

    def forward(self, y_pred, y_true):
        mu = torch.clamp(y_pred, min=self.eps)
        y = torch.clamp(y_true, min=0)

        p = self.power

        # Tweedie log-likelihood (up to a constant)
        loss = (y * mu**(1 - p)) / (1 - p) - (mu**(2 - p)) / (2 - p)

        return -loss.mean()

class FocalMSELoss(nn.Module):
    def __init__(self, gamma=0.6, eps=1e-6, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred, target):
        error = pred - target
        abs_error = torch.abs(error) + self.eps
        
        weight = abs_error ** self.gamma    # focal weight
        loss = weight * (error ** 2)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

class quantile_loss_multi(nn.Module):
    def __init__(self):
        super().__init__()
        self.quantiles = [0.1, 0.5, 0.9]
    def forward(self, y_pred, y_true):
        """
        y_pred: (batch, pred_len, num_q)
        y_true: (batch, pred_len)
        quantiles: list or tensor, e.g. [0.1, 0.5, 0.9]
        """
        assert y_pred.ndim == 3
        B, H, Q = y_pred.shape
        q = torch.tensor(self.quantiles, device=y_pred.device).view(1, 1, Q)
        y_true = y_true.expand_as(y_pred)   # (B, H, Q)
        diff = y_true - y_pred
        loss = torch.maximum(q * diff, (q - 1) * diff)    # (B, H, Q)
        return loss.mean()

class MTS_trainer(Basic_Trainer):
    def __init__(self, model, scaler, args):
        super(MTS_trainer, self).__init__(model, args)
        self.scaler = scaler

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self, loss_name='MSE'):
        if loss_name == 'MSE':
            return nn.MSELoss()
        elif loss_name == 'MAE':
            return nn.L1Loss()
        elif loss_name == "huber":
            return nn.HuberLoss(delta=1.0)
        elif loss_name == "tweedie":
            return TweedieLoss(1.5)
        elif loss_name == 'DILATELoss':
            return DILATELoss()
        elif loss_name == "focal":
            return FocalMSELoss()
        elif loss_name == "quantile":
            return quantile_loss_multi()
        else:
            raise NotImplementedError("No required loss function")
        
    def _build_regime_feat(self, window_x):
        """
        window_x: (Batch, in_len, feature_dim)
        返回 (batch, in_len, regime_dim)
        """
        # w1 = 5
        # w2 = 9
        # w3 = 21
        # B, L, D = window_x.shape
        # gpu = window_x[:, :, -2]   # (B, L)

        # # 1) rolling mean (window w1)
        # # use convolution for fast windowed smoothing
        # kernel = torch.ones(1, 1, w1, device=gpu.device) / w1
        # gpu_ = gpu.unsqueeze(1)              # (B,1,L)
        # roll_mean = torch.nn.functional.conv1d(
        #     gpu_, kernel, padding=w1//2
        # ).squeeze(1)                          # (B,L)

        # # 2) energy_rate = diff(rolling mean)
        # energy_rate = torch.diff(roll_mean, dim=1, prepend=roll_mean[:, :1])

        # # 3) local variance (window w2)
        # # E[x^2] - (E[x])^2
        # kernel2 = torch.ones(1, 1, w2, device=gpu.device) / w2
        # mean2 = torch.nn.functional.conv1d(gpu_, kernel2, padding=w2//2).squeeze(1)
        # mean_sq = torch.nn.functional.conv1d(gpu_**2, kernel2, padding=w2//2).squeeze(1)
        # local_var = mean_sq - mean2**2

        # # 4) local CUSUM (window w3)
        # # local mean
        # kernel3 = torch.ones(1, 1, w3, device=gpu.device) / w3
        # local_mean = torch.nn.functional.conv1d(gpu_, kernel3, padding=w3//2).squeeze(1)
        # cusum = torch.cumsum(gpu - local_mean, dim=1)
        # cusum_signal = torch.diff(cusum, dim=1, prepend=cusum[:, :1])

        # # stack → (B, L, 3)
        # reg = torch.stack([energy_rate, local_var, cusum_signal], dim=-1)
        return None
    
    def train(self, train_loader, valid_loader, test_loader):
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
                    batch_x, x_time, batch_y, y_time, job_timing = batch
                    x_time = x_time.to(self.device)
                    y_time = y_time.to(self.device)
                    reg = self._build_regime_feat(batch_x.float())#.to(self.device)
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    job_timing = job_timing.to(self.device)
                    if self.args.model_name == "timexer":
                        pred_y = self.model(batch_x, x_time, job_timing, reg)
                    else:
                        pred_y = self.model((batch_x, x_time, job_timing))
                    # if reverse the prediction to original dimension
                    # import pdb; pdb.set_trace()
                    if not self.args.is_norm_loss:
                        if self.args.if_col_norm:
                            pred_y = self.scaler.inver_transform_col(
                                pred_y, self.args.out_var
                            )
                            batch_y = self.scaler.inver_transform_col(batch_y, self.args.out_var)
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

            #-------------------------------------
            # _, _, _ = self.infer(test_loader)

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
                batch_x, x_time, batch_y, y_time, job_timing = batch
                x_time = x_time.to(self.device)
                y_time = y_time.to(self.device)
                reg = self._build_regime_feat(batch_x.float())#.to(self.device)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                job_timing = job_timing.to(self.device)
                if self.args.model_name == "timexer":
                    pred_y = self.model(batch_x, x_time, job_timing, reg)
                else:
                    pred_y = self.model((batch_x, x_time, job_timing))
                # if reverse the prediction to original dimension
                if self.args.if_col_norm:
                    pred_y = self.scaler.inver_transform_col(
                        pred_y.cpu(), self.args.out_var
                    )
                    batch_y = self.scaler.inver_transform_col(batch_y.cpu(), self.args.out_var)
                else:
                    pred_y = self.scaler.inver_transform(pred_y)
                    batch_y = self.scaler.inver_transform(batch_y)
                
                all_preds.append(pred_y.cpu())
                all_labels.append(batch_y.cpu())
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        mae, rmse, mape, smape = Metrics(all_preds, all_labels)
        
        metrics = {"mae":mae.item(), "rmse":rmse.item(), "smape": smape.item()}
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
                batch_x, x_time, batch_y, y_time, job_timing = batch
                x_time = x_time.to(self.device)
                y_time = y_time.to(self.device)
                reg = self._build_regime_feat(batch_x.float())#.to(self.device)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                job_timing = job_timing.to(self.device)
                if self.args.model_name == "timexer":
                    pred_y = self.model(batch_x, x_time, job_timing, reg)
                else:
                    pred_y = self.model((batch_x, x_time, job_timing))

                # if reverse the prediction to original dimension
                if self.args.if_col_norm:
                    pred_y = self.scaler.inver_transform_col(
                        pred_y.cpu(), self.args.out_var
                    )
                    batch_y = self.scaler.inver_transform_col(batch_y.cpu(), self.args.out_var)
                else:
                    pred_y = self.scaler.inver_transform(pred_y)
                    batch_y = self.scaler.inver_transform(batch_y)
                all_preds.append(pred_y.cpu())
                all_labels.append(batch_y.cpu())
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        mae, rmse, mape, smape = Metrics(all_preds, all_labels)
        metrics = {"mae":mae.item(), "rmse":rmse.item(), "smape": smape.item()}
        _log = _log_metric(metrics, prefix="Testset's result is: ")
        log.info(_log)

        return metrics, all_labels, all_preds

if __name__=="__main__":
    args = {'gpu': 0}
    mts = MTS_trainer(args)