import torch
from transformers import BlipProcessor
from model.blip.blip import BlipModel

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipModel.from_pretrained("./output/blip_iapm_best").eval()

# 文本特征提取
text = "A person wearing a red shirt and black pants"
inputs = processor(text=text, return_tensors="pt")
with torch.no_grad():
    text_feat = model.get_text_features(**inputs)  # (1, 768)

# 图像特征提取
img = Image.open("./gallery/0001_c2s3_000551_01.jpg").convert("RGB")
inputs = processor(images=img, return_tensors="pt")
with torch.no_grad():
    img_feat = model.get_image_features(**inputs)  # (1, 768)

# 计算相似度
sim = torch.cosine_similarity(text_feat, img_feat).item()
print("文本-图像相似度:", sim)