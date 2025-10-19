import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

class PAPModule(nn.Module):
    """
    Part Attention Pooling 模块
    将全局特征图分成若干部件，并通过注意力加权汇总每个部件的特征
    """
    def __init__(self, in_channels, num_parts=7):
        super().__init__()
        self.num_parts = num_parts
        self.conv1x1 = nn.Conv2d(in_channels, num_parts, kernel_size=1)  # 预测每个部件的注意力图

    def forward(self, feat_map):
        """
        feat_map: (B, C, H, W) 骨干网络输出的特征图
        return:
            part_feats: (B, num_parts, C) 每个部件的特征向量
            attn_maps: (B, num_parts, H, W) 每个部件的注意力图
        """
        B, C, H, W = feat_map.shape
        attn_maps = self.conv1x1(feat_map)  # (B, num_parts, H, W)
        attn_maps = F.softmax(attn_maps.view(B, self.num_parts, -1), dim=-1).view(B, self.num_parts, H, W)

        part_feats = []
        for i in range(self.num_parts):
            attn = attn_maps[:, i:i+1]  # (B, 1, H, W)
            part_feat = (feat_map * attn).sum(dim=(2, 3)) / (attn.sum(dim=(2, 3)) + 1e-6)  # (B, C)
            part_feats.append(part_feat)

        return torch.stack(part_feats, dim=1), attn_maps


class PSModule(nn.Module):
    """
    Part Segmentation 模块
    预测像素级的部件分割图（论文中8个部件类别）
    """
    def __init__(self, in_channels, num_classes=8):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, 256, kernel_size=4, stride=2, padding=1)
        self.conv1x1 = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, feat_map):
        """
        feat_map: (B, C, H, W)
        return: seg_logits: (B, num_classes, 2H, 2W)
        """
        x = self.deconv(feat_map)  # (B, 256, 2H, 2W)
        seg_logits = self.conv1x1(x)  # (B, num_classes, 2H, 2W)
        return seg_logits


class IWMModule(nn.Module):
    """
    Interpretable Weight Map 模块
    生成可解释的注意力权重图（论文中用于可视化和损失约束）
    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, feat_map):
        """
        feat_map: (B, C, H, W)
        return: weight_map: (B, 1, H, W)
        """
        weight_map = torch.sigmoid(self.conv1x1(feat_map))  # 权重在 [0,1]
        return weight_map


class ResNet50_IAPM(nn.Module):
    """
    ResNet50 + IAPM 模型
    """
    def __init__(self, pretrained=False):
        super().__init__()
        # 1. 加载 ResNet50 骨干网络
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # 2. IAPM 模块
        self.pap = PAPModule(in_channels=2048, num_parts=7)  # 原论文7个部件
        self.ps = PSModule(in_channels=2048, num_classes=8)   # 原论文8类分割
        self.iwm = IWMModule(in_channels=2048)

    def forward(self, x):
        """
        x: (B, 3, H, W) 输入图像
        return:
            global_feat: (B, 2048) 全局特征
            part_feats: (B, 7, 2048) 部件特征
            seg_logits: (B, 8, 2H, 2W) 分割预测
            weight_map: (B, 1, H/32, W/32) 可解释权重图
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feat_map = self.layer4(x)  # (B, 2048, H/32, W/32)

        # 全局特征（平均池化）
        global_feat = F.adaptive_avg_pool2d(feat_map, 1).view(feat_map.size(0), -1)  # (B, 2048)

        # IAPM 模块
        part_feats, attn_maps = self.pap(feat_map)  # (B, 7, 2048), (B, 7, H/32, W/32)
        seg_logits = self.ps(feat_map)              # (B, 8, 2H/32, 2W/32)
        weight_map = self.iwm(feat_map)             # (B, 1, H/32, W/32)

        return global_feat, part_feats, seg_logits, weight_map


if __name__ == "__main__":
    # 测试模型
    model = ResNet50_IAPM(pretrained=False)
    x = torch.randn(2, 3, 384, 128)  # 原论文输入尺寸
    global_feat, part_feats, seg_logits, weight_map = model(x)
    print("global_feat:", global_feat.shape)
    print("part_feats:", part_feats.shape)
    print("seg_logits:", seg_logits.shape)
    print("weight_map:", weight_map.shape)