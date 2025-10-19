import os
import sys
import torch

# 加入工程根目录到搜索路径
current_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(current_path)
if root_path not in sys.path:
    sys.path.insert(0, root_path)

# 导入带 IAPM 的 ResNet50
from model.resnet import ResNet50_IAPM

# 官方 ResNet50 权重路径
src_path = os.path.join(root_path, "pretrained", "resnet50_official.pth")
# 转换后保存路径
dst_path = os.path.join(root_path, "pretrained", "resnet50_iapm_init.pth")

# 加载官方权重
src_state_dict = torch.load(src_path, map_location="cpu", weights_only=False)

# 创建 IAPM 模型（自动初始化新增层）
model = ResNet50_IAPM()
dst_state_dict = model.state_dict()

# 复制可匹配的权重
for k, v in src_state_dict.items():
    if k in dst_state_dict and dst_state_dict[k].shape == v.shape:
        dst_state_dict[k] = v
    else:
        print(f"跳过权重：{k}")

# 保存转换后的权重
torch.save(dst_state_dict, dst_path)
print(f"转换完成！权重已保存到：{dst_path}")

if __name__ == "__main__":
    model = ResNet50_IAPM()
    model.load_state_dict(torch.load("./pretrained/resnet50_iapm_init.pth", map_location="cpu", weights_only=False), strict=False)
    print("权重加载成功！")