import torch
import os
import sys

# 加入工程根目录到搜索路径
current_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(current_path)
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from model.blip.vit import ViTForReID
from model.iapm.pap_module import PAPModule
from model.iapm.ps_module import PSModule
from codes.config import Config

def convert_resnet50_iapm_to_vit(resnet50_iapm_path, save_vit_iapm_path):
    """
    将原论文ResNet50的IAPM权重转换为ViT适配的权重（确保PAP/PS/IWM模块初始化与原论文一致）
    Args:
        resnet50_iapm_path: 原论文ResNet50-IAPM权重路径（resnet50_iapm_init.pth）
        save_vit_iapm_path: 转换后ViT-IAPM权重保存路径
    """
    # 1. 加载原论文ResNet50-IAPM权重
    resnet50_iapm_weights = torch.load(resnet50_iapm_path, map_location="cpu")
    print("权重文件包含的键:", resnet50_iapm_weights.keys())
    print(f"Loaded ResNet50-IAPM weights from {resnet50_iapm_path}")
    
    # 2. 初始化ViT与IAPM模块（适配工程结构）
    vit_model = ViTForReID()
    pap_module = PAPModule(num_parts=7)  # 原论文7个部件（2.1节）
    ps_module = PSModule(in_channels=768)  # ViT输出通道768，替换原论文ResNet50的2048
    
    # 3. 权重转换逻辑（核心：保持IAPM模块参数与原论文一致，仅适配特征维度）
    converted_weights = {}
    
    # -------------------------- 3.1 PAP模块权重转换（关键修改：复用原论文PAP权重）--------------------------
    # 初始化的PAP权重，未复用原论文权重；现在提取原权重中"pap.xxx"的参数
    pap_state_dict = pap_module.state_dict()
    for key in pap_state_dict.keys():
        # 原权重键格式：pap.conv1x1.weight → 工程PAP模块键格式：conv1x1.weight
        resnet_pap_key = f"pap.{key}"  # 拼接原权重的PAP键前缀
        if resnet_pap_key in resnet50_iapm_weights:
            pap_state_dict[key] = resnet50_iapm_weights[resnet_pap_key]
            print(f"PAP模块权重加载成功：{resnet_pap_key} → {key}")
    converted_weights["pap_module"] = pap_state_dict
    
    # -------------------------- 3.2 PS模块权重转换（关键修改：读取顶层"ps.xxx"权重）--------------------------
    # 顶层"ps_module"键，实际原权重无此键；现在提取原权重中"ps.xxx"的参数
    vit_ps_weights = ps_module.state_dict()
    # 3.2.1 反卷积层权重转换（原ResNet50输入2048→输出1024；ViT输入768→输出384）
    # 原权重反卷积键：ps.deconv.weight / ps.deconv.bias
    resnet_deconv_w_key = "ps.deconv.weight"
    resnet_deconv_b_key = "ps.deconv.bias"
    if resnet_deconv_w_key in resnet50_iapm_weights and resnet_deconv_b_key in resnet50_iapm_weights:
        # 调整反卷积层通道数：原2048→1024 → 适配ViT的768→384
        resnet_deconv_w = resnet50_iapm_weights[resnet_deconv_w_key]  # 原形状：[1024, 2048, 3, 3]
        # 1. 先调整输入通道：2048→768（截取前768个输入通道）
        resnet_deconv_w_adjust_in = resnet_deconv_w[:, :768, :, :]  # 形状变为[1024, 768, 3, 3]
        # 2. 再调整输出通道：1024→384（截取前384个输出通道）
        # 注：原代码用interpolate是多余的（卷积核尺寸已为3×3），直接截取通道即可
        vit_deconv_w = resnet_deconv_w_adjust_in[:384, :, :, :]  # 最终形状：[384, 768, 3, 3]
        vit_deconv_b = resnet50_iapm_weights[resnet_deconv_b_key][:384]  # 偏置截取前384个
        
        vit_ps_weights["deconv.weight"] = vit_deconv_w
        vit_ps_weights["deconv.bias"] = vit_deconv_b
        print(f"PS模块反卷积权重加载成功：{resnet_deconv_w_key} → deconv.weight（通道2048→768, 1024→384）")
    
    # 3.2.2 1×1卷积层权重转换（原论文输出8类，工程一致，仅调整输入通道）
    # 原权重1×1卷积键：ps.conv1x1.weight / ps.conv1x1.bias
    resnet_conv1x1_w_key = "ps.conv1x1.weight"
    resnet_conv1x1_b_key = "ps.conv1x1.bias"
    if resnet_conv1x1_w_key in resnet50_iapm_weights and resnet_conv1x1_b_key in resnet50_iapm_weights:
        resnet_conv1x1_w = resnet50_iapm_weights[resnet_conv1x1_w_key]  # 原形状：[8, 1024, 1, 1]（输出8类，输入1024）
        # 调整输入通道：1024→384（PS模块反卷积输出通道）
        vit_conv1x1_w = resnet_conv1x1_w[:, :384, :, :]  # 最终形状：[8, 384, 1, 1]
        vit_conv1x1_b = resnet50_iapm_weights[resnet_conv1x1_b_key]  # 偏置无需调整（输出8类不变）
        
        vit_ps_weights["conv1x1.weight"] = vit_conv1x1_w
        vit_ps_weights["conv1x1.bias"] = vit_conv1x1_b
        print(f"PS模块1×1卷积权重加载成功：{resnet_conv1x1_w_key} → conv1x1.weight（输入通道1024→384）")
    
    converted_weights["ps_module"] = vit_ps_weights
    
    # 3.3 ViT权重初始化（原论文无ViT权重，用预训练ViT初始化，此处仅占位；若有预训练权重可补充加载）
    converted_weights["vit_model"] = vit_model.state_dict()
    
    # 4. 保存转换后的权重（用于工程IAPM模块初始化，与原论文参数兼容）
    torch.save(converted_weights, save_vit_iapm_path)
    print(f"\nConverted ViT-IAPM weights saved to {save_vit_iapm_path}")

def load_converted_weights(model, converted_weight_path):
    """加载转换后的权重到工程模型（确保与原论文IAPM初始化一致）"""
    converted_weights = torch.load(converted_weight_path, map_location="cpu")
    # 加载PAP/PS模块权重（与原论文一致）
    model.pap_module.load_state_dict(converted_weights["pap_module"])
    model.ps_module.load_state_dict(converted_weights["ps_module"])
    # 部分加载ViT权重（仅初始化与原论文相关的层；若后续补充ViT预训练权重，此处逻辑无需修改）
    vit_state_dict = model.blip.vision_encoder.state_dict()
    vit_converted = {k: v for k, v in converted_weights["vit_model"].items() if k in vit_state_dict}
    vit_state_dict.update(vit_converted)
    model.blip.vision_encoder.load_state_dict(vit_state_dict)
    print(f"Loaded converted weights from {converted_weight_path} (aligned with paper's IAPM init)")
    return model

if __name__ == "__main__":
    cfg = Config()
    # 转换原论文ResNet50-IAPM权重为ViT适配权重
    convert_resnet50_iapm_to_vit(
        resnet50_iapm_path=os.path.join(cfg.PRETRAINED_DIR, "resnet50_iapm_init.pth"),
        save_vit_iapm_path=os.path.join(cfg.PRETRAINED_DIR, "vit_iapm_init.pth")
    )