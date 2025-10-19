# 模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('cubeai/blip-image-captioning-base', cache_dir=r"A:\DC（行人重识别）\BLIP-IAPM-ReID\cache")
print(f"模型已下载到：{model_dir}")