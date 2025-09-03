import torch
import torch.nn as nn


class GRU(nn.Module):
    def __init__(self, args):
        super(GRU, self).__init__()
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.timeenc = args.timeenc
        if args.timeenc > -1:
            self.d_embedding = nn.Embedding(32, args.T_embed_dim)
            self.w_embedding = nn.Embedding(7, args.T_embed_dim)
            self.h_embedding = nn.Embedding(24, args.T_embed_dim)
            self.m_embedding = nn.Embedding(60, args.T_embed_dim)
        self.gru = nn.GRU(4*args.T_embed_dim+1, args.hidden_size, args.num_layers, batch_first=True)
        self.fc = nn.Linear(args.hidden_size, args.out_len)

    def forward(self, input):
        # x: [batch, seq_len, feats]
        
        if self.timeenc > -1:
            x, x_time = input
            d_time = self.d_embedding(x_time[:, :, 0])
            w_time = self.w_embedding(x_time[:, :, 1])
            h_time = self.h_embedding(x_time[:, :, 2])
            m_time = self.m_embedding(x_time[:, :, 3])
            time_emb = torch.cat([d_time, w_time, h_time, m_time], dim=-1)
            x = torch.cat([x, time_emb], dim=-1)
        else:
            x = input

        out, _ = self.gru(x)                 # out: [batch, seq_len, hidden_size]
        out = out[:, -1, :]                  # 取最后一个时间步的输出
        out = self.fc(out)                   # 输出预测
        return out.unsqueeze(-1) # (B, L, 1)