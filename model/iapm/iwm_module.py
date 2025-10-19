import torch
import torch.nn as nn

class IWMModule(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, num_parts=7):
        super().__init__()
        self.num_parts = num_parts
        # 原论文：2个全连接层+softmax
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)  # 每个部件输出1个权重
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, part_feats):
        """
        part_feats: (B, P, C) - PAP输出的部件特征
        返回：part_weights (B, P) - 可解释权重（和为1）
        """
        B, P, C = part_feats.shape
        # 逐部件计算权重
        weights = []
        for i in range(P):
            feat = part_feats[:, i, :]  # (B, C)
            out = self.fc1(feat)  # (B, hidden_dim)
            out = self.relu(out)
            out = self.fc2(out)  # (B, 1)
            weights.append(out)
        
        weights = torch.cat(weights, dim=1)  # (B, P)
        part_weights = self.softmax(weights)  # 归一化权重
        return part_weights