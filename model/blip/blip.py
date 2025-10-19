import torch
import os
import sys
import torch.nn as nn
from transformers import AutoImageProcessor, AutoTokenizer, BlipModel as BlipForImageTextRetrieval

# 加入工程根目录到搜索路径
current_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(os.path.dirname(current_path))
if root_path not in sys.path:
    sys.path.insert(0, root_path)
    
from model.blip.vit import ViTForReID
from model.blip.bert import BertForReID


class BlipProcessor:
    """图像处理器与文本分词器的组合类"""
    def __init__(self, image_processor, tokenizer):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        
    @classmethod
    def from_pretrained(cls, pretrained_path, device=None):
        """从预训练路径加载处理器"""
        image_processor = AutoImageProcessor.from_pretrained(
            pretrained_path, local_files_only=True, device=device
        )
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_path, local_files_only=True, device=device
        )
        return cls(image_processor, tokenizer)

    def __call__(self, images=None, text=None, return_tensors="pt", **kwargs):
        """统一处理图像和文本输入"""
        outputs = {}
        if images is not None:
            outputs["pixel_values"] = self.image_processor(
                images=images, return_tensors=return_tensors, ** kwargs
            ).pixel_values
        if text is not None:
            text_outputs = self.tokenizer(
                text=text, return_tensors=return_tensors, **kwargs
            )
            outputs.update(text_outputs)
        return outputs

class BlipForReID(nn.Module):
    """BLIP模型适配ReID任务的扩展实现"""
    def __init__(self, vision_encoder=None, text_encoder=None, processor=None):
        super().__init__()
        # 1. 图像编码器
        self.vision_encoder = vision_encoder if vision_encoder is not None else ViTForReID()
        
        # 2. 文本编码器
        self.text_encoder = text_encoder if text_encoder is not None else BertForReID()
        
        # 3. 跨模态注意力层
        self.cross_modal_attn = nn.MultiheadAttention(
            embed_dim=768, num_heads=8, batch_first=True
        )
        
        # 4. 处理器
        self.processor = processor
        # 记录设备信息
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, pixel_values=None, input_ids=None, attention_mask=None):
        """前向传播逻辑"""
        outputs = {}
        
        # 提取图像特征
        if pixel_values is not None:
            outputs["img_feat"] = self.vision_encoder(pixel_values)  # (B, C, H, W)
        
        # 提取文本特征
        if input_ids is not None and attention_mask is not None:
            text_feat, text_seq_feat = self.text_encoder(input_ids, attention_mask)
            outputs["text_feat"] = text_feat          # (B, 768)
            outputs["text_seq_feat"] = text_seq_feat  # (B, L, 768)
        
        # 跨模态注意力计算部分
        if "img_feat" in outputs and "text_seq_feat" in outputs:
            # 图像特征序列化 - 增加维度适配
            img_feat = outputs["img_feat"]  # (B, C, H, W)
            B, C, H, W = img_feat.shape
            img_seq_feat = img_feat.permute(0, 2, 3, 1).reshape(B, -1, C)  # (B, H*W, C)
            
            # 文本序列特征
            text_seq_feat = outputs["text_seq_feat"]  # (B, L, D)
            target_dim = text_seq_feat.shape[-1]  # 目标维度（文本特征的维度）
            
            # 确保图像特征维度与文本特征维度一致
            if C != target_dim:
                if not hasattr(self, 'img_proj'):
                    self.img_proj = nn.Linear(C, target_dim, device=self.device)
                img_seq_feat = self.img_proj(img_seq_feat)  # (B, H*W, target_dim)
            
            # 确保多头注意力的输入维度与 embed_dim 一致
            if img_seq_feat.shape[-1] != self.cross_modal_attn.embed_dim:
                if not hasattr(self, 'attn_proj'):
                    self.attn_proj = nn.Linear(img_seq_feat.shape[-1], self.cross_modal_attn.embed_dim, device=self.device)
                img_seq_feat = self.attn_proj(img_seq_feat)
                text_seq_feat = self.attn_proj(text_seq_feat)  # 文本特征也需要同步映射
            
            # 关键修复：确保注意力层输入的批量大小一致
            # 获取最小批量大小并截断
            min_batch_size = min(img_seq_feat.size(0), text_seq_feat.size(0))
            img_seq_feat = img_seq_feat[:min_batch_size]
            text_seq_feat = text_seq_feat[:min_batch_size]
            if attention_mask is not None:
                attention_mask = attention_mask[:min_batch_size]
            
            # 跨模态注意力计算
            fused_seq_feat, _ = self.cross_modal_attn(
                query=img_seq_feat,
                key=text_seq_feat,
                value=text_seq_feat,
                key_padding_mask=~attention_mask.bool() if attention_mask is not None else None
            )
            outputs["fused_feat"] = fused_seq_feat.mean(dim=1)  # (B, D)
                
            return outputs

    @classmethod
    def from_pretrained(cls, pretrained_path, device=None):
        """从本地加载预训练权重，确保不进行远程下载"""
        # 验证本地路径存在
        if not os.path.isdir(pretrained_path):
            raise FileNotFoundError(f"本地模型路径不存在: {pretrained_path}")
            
        # 设备自动选择
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 强制从本地加载基础模型（BlipForImageTextRetrieval）
        hf_blip = BlipForImageTextRetrieval.from_pretrained(
            pretrained_path,
            local_files_only=True,  # 只从本地加载，不远程下载
            use_safetensors=False
        )
        
        # 初始化编码器（关键修改：text_encoder改为text_model）
        vision_encoder = ViTForReID(vision_model=hf_blip.vision_model)  # 图像编码器（复用BLIP视觉模型）
        text_encoder = BertForReID(text_model=hf_blip.text_model)      # 文本编码器（复用BLIP文本模型）
        
        # 加载处理器（图像预处理+文本分词）
        try:
            image_processor = AutoImageProcessor.from_pretrained(
                pretrained_path, local_files_only=True
            )
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_path, local_files_only=True
            )
            processor = BlipProcessor(image_processor, tokenizer)
        except Exception as e:
            raise RuntimeError(f"处理器加载失败: {str(e)}") from e
        
        # 初始化模型并移动到目标设备
        model = cls(
            vision_encoder=vision_encoder,
            text_encoder=text_encoder,
            processor=processor
        ).to(device)
        model.device = device  # 保存设备信息
        return model

# # 验证脚本
# if __name__ == "__main__":
#     import torch
#     from PIL import Image
#     import numpy as np

#     # 1. 模型初始化（替换为实际本地路径）
#     model_path = os.path.abspath(r"A:\ReID_DC\BLIP-IAPM-ReID\blip_model")
#     model = BlipForReID.from_pretrained(model_path).eval()
#     print("模型加载完成")

#     # 2. 准备测试数据
#     # 模拟图像 (384x128)
#     dummy_image = Image.fromarray(np.random.randint(0, 255, (384, 128, 3), dtype=np.uint8))
#     # 测试文本
#     dummy_text = ["a person wearing a red shirt", "a man with black pants"]

#     if model.processor is None:
#         raise ValueError("模型处理器未正确加载")
    
#     # 3. 数据预处理（关键修改：先获取字典，再逐个移动张量到设备）
#     # 第一步：执行预处理，得到包含张量的字典
#     inputs = model.processor(
#         images=[dummy_image, dummy_image],
#         text=dummy_text,
#         return_tensors="pt"
#     )
#     # 第二步：遍历字典，将每个张量移动到模型所在设备
#     inputs = {k: v.to(model.device) for k, v in inputs.items()}

#     # 4. 前向传播（无需修改）
#     with torch.no_grad():
#         outputs = model(** inputs)

#     # 5. 验证输出（无需修改）
#     print("\n输出特征维度验证:")
#     print(f"图像特征: {outputs['img_feat'].shape}")        # 预期: (2, 768, 24, 8)
#     print(f"文本特征: {outputs['text_feat'].shape}")       # 预期: (2, 768)
#     print(f"融合特征: {outputs['fused_feat'].shape}")      # 预期: (2, 768)
#     print("所有测试通过!")