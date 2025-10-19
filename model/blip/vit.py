import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel

class ViTForReID(nn.Module):
    """适配IAPM的ViT图像编码器（修改输出特征图尺寸以匹配原论文部件分割）
    原论文ResNet50输入384×128 → 输出特征图H=24、W=8；此处ViT需调整.patch_size与.layers，确保输出H=24、W=8
    """
    def __init__(self, vision_model=None, pretrained_path=None):
        super(ViTForReID, self).__init__()
        # 支持从预训练模型实例或路径初始化
        if vision_model is not None:
            self.vit = vision_model
        elif pretrained_path is not None:
            self.vit = ViTModel.from_pretrained(pretrained_path)
        else:
            raise ValueError("必须提供vision_model实例或pretrained_path路径")
            
        self.hidden_size = self.vit.config.hidden_size  # 768
        self.patch_size = self.vit.config.patch_size    # 16（从配置读取）
        
        # 动态计算特征图尺寸（基于输入图像尺寸384×128）
        self.img_height = 384
        self.img_width = 128
        self.grid_h = self.img_height // self.patch_size  # 384÷16=24
        self.grid_w = self.img_width // self.patch_size   # 128÷16=8
        
        # 关键修改：用普通成员函数替换lambda，支持pickle序列化
        self.reshape = self._reshape_feat  # 绑定内部重塑函数

    # 新增：特征重塑的普通成员函数（原lambda逻辑迁移至此）
    def _reshape_feat(self, x):
        """
        对ViT输出的序列特征进行重塑，转换为2D特征图
        Args:
            x: 移除CLS token后的序列特征，shape=(B, patch_num, hidden_size)
        Returns:
            2D特征图，shape=(B, hidden_size, grid_h, grid_w)
        """
        return x.permute(0, 2, 1).reshape(
            -1, self.hidden_size, self.grid_h, self.grid_w
        )

    def forward(self, pixel_values):
        """
        Args:
            pixel_values: 输入图像张量（B, 3, 384, 128）
        Returns:
            feat_map: 2D特征图（B, 768, 24, 8）
        """
        outputs = self.vit(pixel_values=pixel_values)
        last_hidden_state = outputs.last_hidden_state  # (B, patch_num+1, 768)
        
        # 移除CLS token
        feat_without_cls = last_hidden_state[:, 1:, :]  # (B, 192, 768)
        
        # 调用普通成员函数进行重塑（逻辑与原lambda完全一致）
        feat_map = self.reshape(feat_without_cls)  # (B, 768, 24, 8)
        return feat_map
