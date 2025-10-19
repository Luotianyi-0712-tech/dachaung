import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from model.resnet import ResNet50_IAPM

# 1. 配置
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "./output/resnet50_iapm_best.pth"

# 2. 图像预处理（和训练时保持一致）
transform = transforms.Compose([
    transforms.Resize((384, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. 加载模型
model = ResNet50_IAPM().to(device)
model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
model.eval()

# 4. 读取图片并提取特征
def extract_feature(img_path):
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        global_feat, _, _, _ = model(img)  # (1, 2048)
    return F.normalize(global_feat, dim=1)  # L2归一化

query_feat = extract_feature("./query/0001_c1s1_000151_01.jpg")  # 查询图片特征
gallery_feat = extract_feature("./gallery/0001_c2s3_000551_01.jpg")  # 候选图片特征