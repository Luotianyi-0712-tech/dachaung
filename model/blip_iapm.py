import torch
import torch.nn as nn
from model.blip.blip import BlipForReID
from model.iapm.pap_module import PAPModule
from model.iapm.iwm_module import IWMModule
from model.iapm.ps_module import PSModule
from model.iapm.sptl_loss import SPTLLoss

class BlipIAPMReID(nn.Module):
    def __init__(self, blip_pretrained_path, num_parts=7, part_feat_dim=768, num_ids=751, device=None):
        super().__init__()
        # 关键修复：1. 显式指定设备，避免权重加载到错误设备
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 2. 加载预训练BLIP模型（调用新增的from_pretrained方法，对齐原论文多模态逻辑）
        self.blip = BlipForReID.from_pretrained(
            pretrained_path=blip_pretrained_path,
            device=device,  # 传递设备参数，加速加载
        )
        
        # 3. 后续IAPM模块初始化不变（保持原论文逻辑）
        self.pap_module = PAPModule(num_parts=num_parts)  # 7个部件，原论文最优
        self.ps_module = PSModule(in_channels=part_feat_dim)  # 减少部件重叠
        self.iwm_module = IWMModule(in_dim=part_feat_dim, num_parts=num_parts)  # 可解释权重
        self.cross_modal_attn = nn.MultiheadAttention(embed_dim=part_feat_dim, num_heads=8, batch_first=True)
        self.id_head = nn.Linear(part_feat_dim, num_ids)  # 身份分类头，原论文式(4)
        
        # 4. 注册类别中心和计数器（支持[]操作，被state_dict管理）
        self.register_buffer(
            "class_centers", 
            torch.zeros(num_ids, part_feat_dim, device=device)  # 形状：[身份数, 特征维度]
        )
        self.register_buffer(
            "class_counts", 
            torch.zeros(num_ids, dtype=torch.long, device=device)  # 计数：每个身份的样本数
        )

    def forward(self, image_inputs, text_inputs, labels=None, pseudo_labels=None):
        # 文本特征
        text_feat, text_seq_feat = self.blip.text_encoder(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"]
        )

        # 图像特征 + IAPM 部件提取
        vit_feat = self.blip.vision_encoder(image_inputs["pixel_values"])
        vit_feat = self.ps_module(vit_feat, pseudo_labels)  # PS 约束
        part_feats = self.pap_module(vit_feat)  # 部件特征
        part_weights = self.iwm_module(part_feats)  # 可解释权重

        # 跨模态融合
        weighted_part_feats = part_feats * part_weights.unsqueeze(-1)
        cross_feat, _ = self.cross_modal_attn(
            query=weighted_part_feats,
            key=text_seq_feat,
            value=text_seq_feat,
            key_padding_mask=~text_inputs["attention_mask"].bool()
        )
        cross_feat = cross_feat.mean(dim=1)

        if self.training:
            # 计算 ID 损失
            logits_id = self.id_head(cross_feat)
            losses = {
                "id_loss": nn.CrossEntropyLoss()(logits_id, labels),
                "sptl_loss": SPTLLoss(alpha=1.2)(part_feats, part_weights, labels),
                "ps_loss": self.ps_module.loss
            }
            return logits_id, part_weights, part_feats, cross_feat, losses
        else:
            return cross_feat, part_weights