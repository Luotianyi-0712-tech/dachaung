import torch
import torch.nn as nn
import torch.nn.functional as F

class PAPModule(nn.Module):
    def __init__(self, num_parts=7):
        super().__init__()
        self.num_parts = num_parts  # 原论文7个部件（头、上躯干、下躯干、大腿、小腿、脚、全身）
    
    def generate_part_mask(self, feat_map):
        """生成部件注意力掩码（横向分割特征图）
        feat_map: ViT输出特征图，shape=(B, C, H, W)（如B×768×12×4，对应384×128输入）
        返回：mask (B, num_parts, C, H, W)，每个部件对应位置为1，其余为0
        """
        B, C, H, W = feat_map.shape
        mask = torch.zeros(B, self.num_parts, C, H, W, device=feat_map.device)
        
        # 横向分割H维度（按人体比例分配部件高度）
        part_heights = [int(H * r) for r in [0.1, 0.2, 0.15, 0.2, 0.15, 0.1, 1.0]]  # 7个部件比例
        cum_heights = torch.cumsum(torch.tensor(part_heights[:-1]), dim=0)  # 前6个部件的累计高度（全身是整个H）
        
        for i in range(self.num_parts - 1):  # 前6个局部部件
            start_h = 0 if i == 0 else cum_heights[i-1]
            end_h = cum_heights[i]
            # print(f"Part {i+1}: {start_h}~{end_h}")
            mask[:, i, :, start_h:end_h, :] = 1.0
        mask[:, -1, :, :, :] = 1.0  # 全身部件（第7个）
        
        return mask
    
    def forward(self, feat_map):
        """
        feat_map: (B, C, H, W) - ViT图像编码器输出的特征图
        返回：part_feats (B, num_parts, C) - 每个部件的全局池化特征
        """
        B, C, H, W = feat_map.shape
        mask = self.generate_part_mask(feat_map)  # (B, P, C, H, W)，P=num_parts
        
        # 逐部件应用掩码并全局maxpool
        part_feats = []
        for i in range(self.num_parts):
            masked_feat = feat_map * mask[:, i, :, :, :]  # (B, C, H, W)
            part_feat = F.adaptive_max_pool2d(masked_feat, (1, 1)).squeeze(-1).squeeze(-1)  # (B, C)
            part_feats.append(part_feat)
        
        return torch.stack(part_feats, dim=1)  # (B, P, C)