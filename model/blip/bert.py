import torch
import torch.nn as nn
from transformers import BertModel

class BertForReID(nn.Module):
    """适配ReID任务的Bert文本编码器（支持从BLIP的text_model初始化）"""
    def __init__(self, text_model=None, pretrained_path=None):
        super().__init__()
        # 支持两种初始化方式：1. 直接传入BLIP的text_model实例 2. 从路径加载Bert模型
        if text_model is not None:
            self.bert = text_model  # 直接使用BLIP预训练好的文本模型
        elif pretrained_path is not None:
            self.bert = BertModel.from_pretrained(pretrained_path)  # 从路径加载
        else:
            raise ValueError("必须提供text_model实例或pretrained_path路径")
        
        self.hidden_size = self.bert.config.hidden_size  # 768（与BLIP文本模型一致）

    def forward(self, input_ids, attention_mask):
        """
        前向传播：输出文本全局特征和序列特征
        Args:
            input_ids: 文本token ID (B, L)
            attention_mask: 注意力掩码 (B, L)
        Returns:
            text_feat: 全局特征（CLS token）(B, 768)
            text_seq_feat: 序列特征（所有token）(B, L, 768)
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        text_seq_feat = outputs.last_hidden_state  # (B, L, 768)：所有token的特征
        text_feat = text_seq_feat[:, 0, :]  # (B, 768)：取CLS token作为全局特征
        return text_feat, text_seq_feat
