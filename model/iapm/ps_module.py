import torch
import torch.nn as nn
import torch.nn.functional as F

class PSModule(nn.Module):
    """局部分割约束模块（复现原论文1.2节PS模块设计）"""
    def __init__(self, in_channels=2048, num_classes=8):
        """
        Args:
            in_channels: 输入特征图通道数（原论文ResNet50输出为2048，适配BLIP时需改为768）
            num_classes: 分割类别数（原论文8类：背景、头、躯干、前臂、后臂、大腿、小腿、脚）
        """
        super().__init__()
        self.num_classes = num_classes
        # 原论文结构：3×3反卷积（步长2，上采样）→ 1×1卷积（逐像素分类）
        self.deconv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=in_channels // 2,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1  # 保证上采样后尺寸为2H×2W
        )
        self.conv1x1 = nn.Conv2d(
            in_channels=in_channels // 2,
            out_channels=num_classes,
            kernel_size=1,
            stride=1,
            padding=0
        )
        # 分割损失（交叉熵，原论文式(6)）
        self.seg_loss = nn.CrossEntropyLoss(reduction="none")
        self.loss = 0.0  # 存储当前批次的分割损失

    def forward(self, feat_map, pseudo_labels=None):
        """
        Args:
            feat_map: 输入特征图（B, C, H, W），来自ResNet50/BLIP-ViT
            pseudo_labels: 伪标签（B, 2H, 2W），训练时传入，测试时为None
        Returns:
            feat_map: 经过PS约束后的特征图（B, C, H, W）（仅训练时生效约束，测试时直通）
            self.loss: 分割损失（仅训练时计算）
        """
        B, C, H, W = feat_map.shape
        # 1. 上采样+分类预测（原论文式(6)前向过程）
        x = F.relu(self.deconv(feat_map))  # (B, C//2, 2H, 2W)
        seg_logits = self.conv1x1(x)       # (B, num_classes, 2H, 2W)

        # 2. 训练时计算分割损失（原论文式(6)：各类损失取均值）
        if self.training and pseudo_labels is not None:
            # 交叉熵损失：需将logits展平为(B*2H*2W, num_classes)，标签展平为(B*2H*2W)
            flat_logits = seg_logits.permute(0, 2, 3, 1).reshape(-1, self.num_classes)  # (B*2H*2W, 8)
            flat_labels = pseudo_labels.reshape(-1)  # (B*2H*2W)
            # 逐类计算平均损失（避免大部件占比过高）
            class_losses = []
            for cls in range(self.num_classes):
                cls_mask = (flat_labels == cls).float()
                if cls_mask.sum() > 0:  # 跳过无样本的类别
                    cls_loss = (self.seg_loss(flat_logits, flat_labels) * cls_mask).sum() / cls_mask.sum()
                    class_losses.append(cls_loss)
            self.loss = torch.mean(torch.stack(class_losses)) if class_losses else torch.tensor(0.0).to(feat_map.device)
        else:
            self.loss = torch.tensor(0.0).to(feat_map.device)

        # 3. 测试时不修改特征图，训练时通过损失反向传播约束特征分割
        return feat_map