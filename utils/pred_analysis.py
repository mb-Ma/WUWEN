import pandas as pd
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
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

# pred_metrics()

def pred_plot(
    file_path='./experiments/timemixer/processed_data/2025-09-11_10:38:28/',
    col_names=None,
    mode='flatten',  # 'flatten' 或 'samples'
    sample_indices=None,  # 指定要可视化的样本索引，mode='samples'时生效
    inverse=True,  # 是否反归一化
    var_idx=1,  # 指定要可视化的变量索引
    n_samples=5,  # mode='samples'时默认展示前n个样本
    past_offset=12,  # 过去序列的偏移量，默认为12
    train_start="2025-07-29 09:42:00",  # 训练集起始时间
    train_end="2025-08-19 09:21:30"     # 训练集结束时间
):
    """
    mode:
        'flatten'：将所有样本和时间步展开为一条曲线
        'samples'：每个样本单独一个子图，展示多个样本，并标注过去序列
    sample_indices:
        指定要展示的样本索引列表，若为None则默认展示前n_samples个
    var_idx:
        指定要展示的变量索引
    past_offset:
        过去序列的样本偏移量
    train_start, train_end:
        用于计算均值和方差的训练集时间范围
    """
    # 读取原始数据，计算指定变量的均值和方差
    data = pd.read_csv('./data/processed_data/1_datacenter.csv')
    data = data.set_index('timestamp')
    # 获取所有变量名
    all_columns = list(data.columns)
    if var_idx >= len(all_columns):
        raise ValueError(f"var_idx超出范围，最大为{len(all_columns)-1}")
    var_name = all_columns[var_idx]
    mean = data[var_name].loc[train_start:train_end].mean()
    std = data[var_name].loc[train_start:train_end].std()

    # 加载预测结果
    npz_data = np.load(file_path + 'result.npz')
    pred = npz_data['pred_y']  # (样本数, 序列长度, 变量数)
    true = npz_data['real_y']
    # 反归一化
    pred = pred * std + mean
    true = true * std + mean
    print(pred.shape, true.shape)

    if mode == 'flatten':
        pred_flat = pred[:, :, var_idx].flatten()
        true_flat = true[:, :, var_idx].flatten()
        plt.figure(figsize=(15, 8))
        plt.plot(pred_flat[:300], label='Pred')
        plt.plot(true_flat[:300], label='True')
        plt.legend()
        plt.title(f'Flattened Prediction vs True (var_idx={var_idx})')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.show()
        plt.savefig('../pred_plot/pred_plot.png')
    elif mode == 'samples':
        # 每个样本单独一个子图，拼接过去序列和未来序列，前past_offset为过去，后为未来
        n = pred.shape[0]
        seq_len = pred.shape[1]
        if sample_indices is None:
            sample_indices = list(range(min(n_samples, n)))
        num_plots = len(sample_indices)
        plt.figure(figsize=(15, 3 * num_plots))
        for i, idx in enumerate(sample_indices):
            plt.subplot(num_plots, 1, i + 1)
            # 检查是否有过去序列
            if idx - past_offset >= 0:

                past_true = true[idx - past_offset, :, var_idx]
                future_true = true[idx, :, var_idx]
                future_pred = pred[idx, :, var_idx]

                x = list(range(past_offset + seq_len))
                plt.plot(range(past_offset), past_true, label='Past True', color='tab:orange')
                plt.plot(range(past_offset, past_offset + seq_len), future_true, label='Future True', color='tab:green')
                plt.plot(range(past_offset, past_offset + seq_len), future_pred, label='Future Pred', color='tab:blue')
                plt.axvspan(0, past_offset-1, color='lightgray', alpha=0.3, label='Past Sequence')
                plt.axvspan(past_offset-0.5, past_offset+seq_len-1, color='lightblue', alpha=0.15, label='Future Sequence')
                plt.axvline(past_offset-0.5, color='gray', linestyle='--')
            else:
                # 没有过去序列
                plt.text(0.5, 0.5, 'No past sequence', transform=plt.gca().transAxes, fontsize=10, color='gray', ha='center')
            plt.legend()
            plt.title(f'Sample {idx} (var_idx={var_idx})')
            plt.xlabel('Time Step')
            plt.ylabel('Value')
        plt.tight_layout()
        plt.show()
        plt.savefig('../pred_plot/pred_plot.png')
    else:
        raise ValueError("mode 只支持 'flatten' 或 'samples'")
# 示例绘图
# pred_plot(mode='samples',var_idx=1, inverse=True, sample_indices=[k for k in range(150,172)])