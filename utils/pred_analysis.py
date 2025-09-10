import pandas as pd
import numpy as np
from tabulate import tabulate
from utils.metrics import MAE_np, RMSE_np, MAPE_np, R2_np, SPEARMAN_np, PEARSON_np, SMAPE_np
def pred_metrics(file_path='../experiments/timemixer/processed_data/2025-09-10_22:13:31/', col_names=None):
    # 加载数据
    data = np.load(file_path+'result.npz')
    pred = data['pred_y']  # (样本数, 序列长度, 变量数)
    true = data['real_y']
    print(pred.shape, true.shape)

    n_vars = pred.shape[2]
    if col_names is None:
        # 默认变量名
        col_names = [f'var_{i}' for i in range(n_vars)]

    # 单变量（每一列）指标
    table = []
    for i in range(n_vars):
        pred_col = pred[:, :, i].reshape(-1)
        true_col = true[:, :, i].reshape(-1)
        mae = MAE_np(pred_col, true_col)
        rmse = RMSE_np(pred_col, true_col)
        mape = MAPE_np(pred_col, true_col)
        r2 = R2_np(pred_col, true_col)
        spearman = SPEARMAN_np(pred_col, true_col)
        pearson = PEARSON_np(pred_col, true_col)
        smape = SMAPE_np(pred_col, true_col)
        table.append([
            col_names[i], 
            f"{mae:.4f}", 
            f"{rmse:.4f}", 
            f"{mape:.2f}%", 
            f"{smape:.2f}%", 
            f"{r2:.4f}", 
            f"{spearman:.4f}", 
            f"{pearson:.4f}"
        ])
    headers = ["变量", "MAE", "RMSE", "MAPE", "SMAPE", "R2", "Spearman", "Pearson"]
    print("单变量表现：")
    print(tabulate(table, headers=headers, tablefmt="grid", showindex=True))

    # 多变量整体指标
    pred_all_flat = pred.reshape(-1)
    true_all_flat = true.reshape(-1)
    mae = MAE_np(pred_all_flat, true_all_flat)
    rmse = RMSE_np(pred_all_flat, true_all_flat)
    mape = MAPE_np(pred_all_flat, true_all_flat)
    r2 = R2_np(pred_all_flat, true_all_flat)
    spearman = SPEARMAN_np(pred_all_flat, true_all_flat)
    pearson = PEARSON_np(pred_all_flat, true_all_flat)
    smape = SMAPE_np(pred_all_flat, true_all_flat)
    print("\n多变量整体表现：")
    print(tabulate([[
        f"{mae:.4f}", 
        f"{rmse:.4f}", 
        f"{mape:.2f}%", 
        f"{smape:.2f}%", 
        f"{r2:.4f}", 
        f"{spearman:.4f}", 
        f"{pearson:.4f}"
    ]], headers=["MAE", "RMSE", "MAPE", "SMAPE", "R2", "Spearman", "Pearson"], tablefmt="grid"))

    return mae, rmse, mape, r2, spearman, pearson, smape

pred_metrics()