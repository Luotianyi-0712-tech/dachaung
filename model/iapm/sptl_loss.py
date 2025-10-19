import torch
import torch.nn as nn
import torch.nn.functional as F

class SPTLLoss(nn.Module):
    def __init__(self, alpha=1.2, num_parts=7):
        """
        初始化显著部件三元损失（SPTL）
        参数说明：
            alpha: 距离边际参数，论文最优值为1.2（式9）
            num_parts: 人体部件总数，论文中P=7（头、上躯干、下躯干、大腿、小腿、脚、全身）
        """
        super().__init__()
        self.alpha = alpha  # 论文式9中的距离边际参数α
        self.num_parts = num_parts  # 论文中的P（人体部件总数）
        self.eps = 1e-8  # 防止数值计算中除零或梯度爆炸

    def forward(self, part_feats, part_weights, labels):
        """
        计算显著部件三元损失
        输入参数（严格对应论文定义）：
            part_feats: (B, P, C) - 各部件特征，对应论文中的e_{a|i}, e_{pos|i}, e_{neg|i}
                        B=批量大小，P=部件数，C=特征维度（论文中d=256）
            part_weights: (B, P) - IWM生成的可解释权重，对应论文中的w_{a|i}, w_{pos|i}, w_{neg|i}
                        需满足每个样本的部件权重和为1（IWM模块已通过softmax保证）
            labels: (B,) - 行人身份标签，用于区分正/负样本对
        返回：
            sptl_loss: 标量 - 总显著部件三元损失（论文式10）
        """
        B, P, C = part_feats.shape
        total_loss = 0.0  # 累积所有部件的损失（论文式10）

        # 论文核心：直接使用IWM输出的权重，无需额外sigmoid归一化（IWM已通过softmax保证权重分布）
        # 验证权重合法性（可选，训练初期可开启以确保IWM输出正确）
        # weight_sum = part_weights.sum(dim=1, keepdim=True)
        # assert torch.allclose(weight_sum, torch.ones_like(weight_sum), atol=1e-3), \
        #     "IWM输出权重和不为1，违反论文定义"

        for i in range(self.num_parts):
            # 1. 提取第i个部件的特征和权重（对应论文中单个部件的三元组计算）
            feat_i = part_feats[:, i, :]  # (B, C) - 第i个部件的所有样本特征
            weight_i = part_weights[:, i].unsqueeze(1)  # (B, 1) - 第i个部件的所有样本权重

            # 2. 计算批量内所有样本对的L2距离（论文式7、8中的||e_a - e_pos||_2）
            dist_matrix = torch.cdist(feat_i, feat_i, p=2)  # (B, B) - 样本对距离矩阵

            # 3. 权重加权：论文式7、8中的w_a * w_pos * 距离（权重越大，距离对损失影响越强）
            weighted_dist = dist_matrix * (weight_i * weight_i.T)  # (B, B) - 加权距离矩阵

            # 4. 生成掩码：区分正样本、负样本、自身样本（排除自身对比）
            mask_same_id = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()  # (B,B) - 同类身份掩码
            mask_self = torch.eye(B, device=part_feats.device).float()  # (B,B) - 自身样本掩码
            mask_pos = mask_same_id - mask_self  # (B,B) - 正样本掩码（同类且非自身）
            mask_neg = 1 - mask_same_id  # (B,B) - 负样本掩码（不同类）

            # 5. 筛选有效锚点：同时存在正、负样本的锚点（避免无样本对导致的计算错误）
            has_pos = (mask_pos.sum(dim=1) > self.eps).float()  # (B,) - 每个锚点是否有正样本
            has_neg = (mask_neg.sum(dim=1) > self.eps).float()  # (B,) - 每个锚点是否有负样本
            valid_anchors = has_pos * has_neg  # (B,) - 有效锚点标记（1=有效，0=无效）
            
            # print(f"部件{i}有效锚点数量: {valid_anchors.sum().item()}")

            if valid_anchors.sum() < 1:
                continue  # 无有效锚点时，当前部件不贡献损失

            # 6. 挖掘最难正样本：锚点与同类样本的最大加权距离（论文式7：d_{a,pos|i} = max(...)）
            pos_dist = weighted_dist * mask_pos  # (B,B) - 仅保留正样本对的加权距离
            # 非正样本位置设为极小值（确保max只取正样本）
            pos_dist = pos_dist + (-1e9 * (1 - mask_pos))
            hardest_pos_dist = pos_dist.max(dim=1)[0]  # (B,) - 每个锚点的最难正样本距离

            # 7. 挖掘最难负样本：锚点与不同类样本的最小加权距离（论文式8：d_{a,neg|i} = min(...)）
            neg_dist = weighted_dist * mask_neg  # (B,B) - 仅保留负样本对的加权距离
            # 非负样本位置设为极大值（确保min只取负样本）
            neg_dist = neg_dist + (1e9 * (1 - mask_neg))
            hardest_neg_dist = neg_dist.min(dim=1)[0]  # (B,) - 每个锚点的最难负样本距离（原代码核心修正点）

            # 8. 计算单个部件的三元损失（论文式9：L_i^{SPTL} = max(0, d_pos - d_neg + α)）
            distance_diff = hardest_pos_dist - hardest_neg_dist + self.alpha  # (B,)
            # 仅有效锚点贡献损失，无效锚点设为0（relu后自然无损失）
            distance_diff = distance_diff * valid_anchors
            part_triplet_loss = F.relu(distance_diff)  # (B,) - 单个锚点的损失（非负）

            # 9. 累加当前部件的损失（论文式10：对所有部件求和）
            # 除以有效锚点数：避免批量中有效锚点数量差异导致的损失波动
            loss_i = part_triplet_loss.sum() / (valid_anchors.sum() + self.eps)
            total_loss += loss_i

        # 论文式10：直接返回所有部件损失之和（无需除以部件数，原代码误解点）
        # 若所有部件均无有效锚点，返回0避免梯度NaN
        return total_loss if total_loss > 0 else torch.tensor(0.0, device=part_feats.device)