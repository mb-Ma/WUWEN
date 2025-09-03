import torch
import torch.nn as nn


class GRU(nn.Module):
    def __init__(self, args):
        super(GRU, self).__init__()
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.gru = nn.GRU(args.in_len, args.hidden_size, args.num_layers, batch_first=True)
        self.fc = nn.Linear(args.hidden_size, args.out_len)

    def forward(self, x):
        # x: [batch, seq_len, feats]
        out, _ = self.gru(x)                 # out: [batch, seq_len, hidden_size]
        out = out[:, -1, :]                  # 取最后一个时间步的输出
        out = self.fc(out)                   # 输出预测
        return out.unsequeeze(-1) # (B, L, 1)