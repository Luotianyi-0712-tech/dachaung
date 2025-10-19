import os
import json
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import sys

# 加入工程根目录到搜索路径
current_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(current_path)
if root_path not in sys.path:
    sys.path.insert(0, root_path)
    
from model.blip.blip import BlipProcessor

def resize_images(raw_dir, save_dir, target_size=(384, 128)):
    """将图像Resize为原论文指定的384×128"""
    os.makedirs(save_dir, exist_ok=True)
    for img_name in tqdm(os.listdir(raw_dir)):
        if img_name.endswith(('.jpg', '.png')):
            img_path = os.path.join(raw_dir, img_name)
            img = Image.open(img_path).convert('RGB')
            img = img.resize(target_size, Image.Resampling.BILINEAR)
            img.save(os.path.join(save_dir, img_name))

def generate_text_captions(img_dir, save_path, blip_processor, blip_model, device):
    """用BLIP生成图像的文本描述（适配行人特征）"""
    img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]
    captions = {}
    
    blip_model.eval()
    with torch.no_grad():
        for img_path in tqdm(img_paths):
            img = Image.open(img_path).convert('RGB')
            inputs = blip_processor(images=img, return_tensors="pt").to(device)
            # 生成行人相关描述（如服装颜色、款式、配件）
            out = blip_model.generate(**inputs, max_length=50, num_beams=3)
            caption = blip_processor.decode(out[0], skip_special_tokens=True, use_fast=False)
            captions[img_path] = caption
    
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(captions, f, indent=2)

if __name__ == "__main__":
    # 配置参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_data_dir = "./data/raw/Market-1501/bounding_box_train"
    save_img_dir = "./data/processed/Market-1501/images"
    save_caption_path = "./data/processed/Market-1501/captions.json"
    
    # 加载BLIP预处理器和模型（用于生成文本标注）
    local_model_path = r"A:\ReID_DC\BLIP-IAPM-ReID\blip_model"  # 替换为你的本地模型文件夹路径
    blip_processor = BlipProcessor.from_pretrained(local_model_path)
    blip_model = BlipForConditionalGeneration.from_pretrained(
        local_model_path,
        torch_dtype=torch.float32  # 可选，加速推理
    )
    blip_model = blip_model.to(device)  # type: ignore
    
    # 1. Resize图像
    resize_images(raw_data_dir, save_img_dir)
    # 2. 生成文本标注
    generate_text_captions(save_img_dir, save_caption_path, blip_processor, blip_model, device)