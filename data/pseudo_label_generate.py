import os
import sys
import numpy as np
from PIL import Image, ImageDraw
import torch
from torchvision.models.detection import keypointrcnn_resnet50_fpn, maskrcnn_resnet50_fpn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from tqdm import tqdm
import logging
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

# 加入工程根目录到搜索路径
current_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(current_path)
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from codes.config import Config

# 字体
plt.rcParams["font.family"] = ["SimHei", "sans-serif"]
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def safe_mean(keypoints, indices):
    """安全计算可见关键点的均值，忽略不可见点"""
    visible_keypoints = keypoints[indices][keypoints[indices, 2] > 0]
    if len(visible_keypoints) == 0:
        return None
    return np.mean(visible_keypoints, axis=0)

def visualize_pseudo_labels(raw_img, pseudo_label, keypoints=None, num_samples=5, save_vis_dir=None):
    """
    可视化伪标签和原始图像，检查部位划分是否合理
    
    Args:
        raw_img: 原始图像 (PIL Image)
        pseudo_label: 伪标签图像 (PIL Image)
        keypoints: 关键点信息 (可选)
        num_samples: 要可视化的样本数
        save_vis_dir: 可视化结果保存目录 (可选)
    """
    # 定义各部位的颜色映射
    part_colors = [
        (0, 0, 0),        # 0: background - 黑色
        (255, 0, 0),      # 1: head - 红色
        (0, 255, 0),      # 2: torso - 绿色
        (0, 0, 255),      # 3: forearm - 蓝色
        (255, 255, 0),    # 4: upper_arm - 黄色
        (255, 0, 255),    # 5: thigh - 品红
        (0, 255, 255),    # 6: calf - 青色
        (128, 0, 0)       # 7: foot - 深红色
    ]
    part_names = ["background", "head", "torso", "forearm", "upper_arm", "thigh", "calf", "foot"]
    
    # 创建颜色映射用于matplotlib显示
    cmap = ListedColormap(np.array(part_colors) / 255.0)
    
    # 将伪标签恢复到原始图像尺寸
    pseudo_label_resized = pseudo_label.resize(raw_img.size, Image.Resampling.NEAREST)
    pseudo_np = np.array(pseudo_label_resized)
    
    # 创建叠加可视化图像
    overlay = raw_img.copy()
    overlay_np = np.array(overlay)
    mask = pseudo_np > 0  # 非背景区域
    
    # 对每个部位添加半透明颜色
    for i in range(1, len(part_colors)):
        part_mask = pseudo_np == i
        if np.any(part_mask):
            overlay_np[part_mask] = (
                overlay_np[part_mask] * 0.6 + 
                np.array(part_colors[i]) * 0.4
            ).astype(np.uint8)
    overlay = Image.fromarray(overlay_np)
    
    # 如果有关键点，绘制关键点
    if keypoints is not None:
        draw = ImageDraw.Draw(overlay)
        for kp in keypoints:
            x, y, vis = kp
            if vis > 0:  # 只绘制可见关键点
                draw.ellipse([(x-3, y-3), (x+3, y+3)], fill=(255, 255, 255))
    
    # 显示图像
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(raw_img)
    axes[0].set_title("原始图像")
    axes[0].axis('off')
    
    axes[1].imshow(pseudo_np, cmap=cmap, vmin=0, vmax=7)
    axes[1].set_title("伪标签")
    axes[1].axis('off')
    
    axes[2].imshow(overlay)
    axes[2].set_title("部位划分叠加效果")
    axes[2].axis('off')
    
    handles = [Rectangle((0,0),1,1, facecolor=np.array(color)/255.0) 
               for color in part_colors]
               
    plt.legend(handles, part_names, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # 保存可视化结果（如果指定了目录）
    if save_vis_dir:
        os.makedirs(save_vis_dir, exist_ok=True)
        save_path = os.path.join(save_vis_dir, f"vis_{random.randint(1000, 9999)}.png")
        plt.savefig(save_path, bbox_inches='tight')
        logger.info(f"可视化结果已保存至: {save_path}")
    
    # plt.show()
    # plt.close()

def generate_pseudo_labels(raw_img_dir, save_label_dir, device="cuda", vis_samples=5, vis_dir=None):
    """
    生成PS模块所需的伪标签（同时调用关键点模型和分割模型）
    新增参数：
        vis_samples: 要可视化的样本数量
        vis_dir: 可视化结果保存目录
    """
    # 1. 初始化关键点检测模型
    try:
        keypoint_model = keypointrcnn_resnet50_fpn(
            pretrained=False,
            weights=None,
            box_detections_per_img=100,
            rpn_pre_nms_top_n_test=100,
            rpn_post_nms_top_n_test=100,
            keypoint_backbone=resnet_fpn_backbone('resnet50', False),
            num_keypoints=17,
            num_classes=2  # 1类人体+1类背景
        ).to(device)
        # 加载关键点模型权重
        keypoint_weight_path = "./keypointrcnn_resnet50_fpn_coco/keypointrcnn_resnet50_fpn_coco-9f466800.pth"
        keypoint_model.load_state_dict(torch.load(keypoint_weight_path, map_location=device))
        keypoint_model.eval()
        logger.info(f"成功加载关键点模型: {keypoint_weight_path}")
    except Exception as e:
        logger.error(f"关键点模型加载失败: {str(e)}")
        return

    # 2. 初始化掩码分割模型
    try:
        mask_model = maskrcnn_resnet50_fpn(
            pretrained=False,
            weights=None,
            num_classes=91
        ).to(device)
        # 加载分割模型权重
        mask_weight_path = "./maskrcnn_resnet50_fpn_coco/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth"
        mask_model.load_state_dict(torch.load(mask_weight_path, map_location=device))
        mask_model.eval()
        logger.info(f"成功加载分割模型: {mask_weight_path}")
    except Exception as e:
        logger.error(f"分割模型加载失败: {str(e)}")
        return
    
    # 3. 定义COCO人体关键点与8类分割的映射
    coco_person_parts = {
        "background": 0,
        "head": 1,          # 头部关键点（nose, eyes, ears）
        "torso": 2,         # 躯干关键点（shoulders, hips）
        "forearm": 3,       # 前臂关键点（elbows, wrists）
        "upper_arm": 4,     # 后臂关键点（shoulders, elbows）
        "thigh": 5,         # 大腿关键点（hips, knees）
        "calf": 6,          # 小腿关键点（knees, ankles）
        "foot": 7           # 脚部关键点（ankles, toes）
    }
    
    # 4. 创建保存目录
    os.makedirs(save_label_dir, exist_ok=True)
    if vis_dir:
        os.makedirs(vis_dir, exist_ok=True)
    
    # 5. 批量生成伪标签
    img_names = [f for f in os.listdir(raw_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    logger.info(f"发现{len(img_names)}张图像，开始生成伪标签...")
    
    # 选择要可视化的图像索引
    if vis_samples > 0 and len(img_names) > 0:
        vis_indices = set(random.sample(range(len(img_names)), min(vis_samples, len(img_names))))
    else:
        vis_indices = set()
    
    with torch.no_grad():
        for img_idx, img_name in enumerate(tqdm(img_names, desc="Generating pseudo-labels")):
            try:
                img_path = os.path.join(raw_img_dir, img_name)
                # 读取图像（保留原始尺寸用于可视化）
                raw_img = Image.open(img_path).convert('RGB')
                # 预处理（Resize为384×128用于模型推理）
                img = raw_img.resize((128, 384))  # (W, H)
                img_tensor = torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
                
                # 6. 分别使用两个模型推理
                # 6.1 关键点模型推理
                keypoint_outputs = keypoint_model(img_tensor)
                # 6.2 分割模型推理
                mask_outputs = mask_model(img_tensor)
                
                # 7. 处理分割结果（获取人体掩码）
                person_mask = (mask_outputs[0]["labels"] == 1).nonzero().squeeze(1)
                visualize_keypoints = None  # 用于可视化的关键点
                
                if len(person_mask) == 0:
                    # 无人体检测结果时，标签设为全背景
                    pseudo_label = np.zeros((384, 128), dtype=np.uint8)
                    if img_idx % 100 == 0:
                        logger.warning(f"图像 {img_name} 未检测到人体")
                else:
                    # 取置信度最高的人体mask
                    best_mask_idx = mask_outputs[0]["scores"][person_mask].argmax()
                    person_seg_mask = mask_outputs[0]["masks"][person_mask[best_mask_idx]].squeeze(0).cpu().numpy() > 0.5  # (384, 128)
                    
                    # 8. 处理关键点结果
                    kp_person_mask = (keypoint_outputs[0]["labels"] == 1).nonzero().squeeze(1)
                    if len(kp_person_mask) == 0 or "keypoints" not in keypoint_outputs[0]:
                        logger.warning(f"图像 {img_name} 未检测到人体关键点，设为全背景标签")
                        pseudo_label = np.zeros((384, 128), dtype=np.uint8)
                        pseudo_label_img = Image.fromarray(pseudo_label).resize((16, 48), Image.Resampling.NEAREST)
                        save_path = os.path.join(save_label_dir, os.path.splitext(img_name)[0] + ".png")
                        pseudo_label_img.save(save_path)
                        
                        # 检查是否需要可视化
                        if img_idx in vis_indices:
                            visualize_pseudo_labels(raw_img, pseudo_label_img, None, vis_samples, vis_dir)
                        continue

                    # 获取置信度最高的人体关键点
                    best_kp_idx = keypoint_outputs[0]["scores"][kp_person_mask].argmax()
                    person_keypoints = keypoint_outputs[0]["keypoints"][kp_person_mask[best_kp_idx]].cpu().numpy()  # (17, 3)
                    
                    # 保存用于可视化的关键点（转换回原始图像尺寸）
                    orig_w, orig_h = raw_img.size
                    scale_w = orig_w / 128  # 原始宽度 / 处理宽度
                    scale_h = orig_h / 384  # 原始高度 / 处理高度
                    visualize_keypoints = person_keypoints.copy()
                    visualize_keypoints[:, 0] *= scale_w  # x坐标缩放
                    visualize_keypoints[:, 1] *= scale_h  # y坐标缩放
                    
                    # 9. 基于关键点划分8类部件
                    pseudo_label = np.zeros((384, 128), dtype=np.uint8)
                    H, W = pseudo_label.shape
                    
                    # 计算各部位关键点均值（带可见性判断）
                    shoulder = safe_mean(person_keypoints, [5, 6])  # 左右肩
                    hip = safe_mean(person_keypoints, [11, 12])      # 左右髋
                    knee = safe_mean(person_keypoints, [13, 14])     # 左右膝
                    ankle = safe_mean(person_keypoints, [15, 16])    # 左右脚踝
                    elbow = safe_mean(person_keypoints, [7, 8])      # 左右肘
                    wrist = safe_mean(person_keypoints, [9, 10])     # 左右手腕
                    
                    # 9.1 头部（y < 肩部y坐标）
                    if shoulder is not None:
                        shoulder_y = shoulder[1]
                        pseudo_label[:int(shoulder_y), :] = coco_person_parts["head"]
                    
                    # 9.2 躯干（肩部y < y < 髋部y）
                    if shoulder is not None and hip is not None:
                        shoulder_y = shoulder[1]
                        hip_y = hip[1]
                        if shoulder_y < hip_y:
                            pseudo_label[int(shoulder_y):int(hip_y), :] = coco_person_parts["torso"]
                    
                    # 9.3 大腿（髋部y < y < 膝盖y）
                    if hip is not None and knee is not None:
                        hip_y = hip[1]
                        knee_y = knee[1]
                        if hip_y < knee_y:
                            pseudo_label[int(hip_y):int(knee_y), :] = coco_person_parts["thigh"]
                    
                    # 9.4 小腿（膝盖y < y < 脚踝y）
                    if knee is not None and ankle is not None:
                        knee_y = knee[1]
                        ankle_y = ankle[1]
                        if knee_y < ankle_y:
                            pseudo_label[int(knee_y):int(ankle_y), :] = coco_person_parts["calf"]
                    
                    # 9.5 脚（y > 脚踝y）
                    if ankle is not None:
                        ankle_y = ankle[1]
                        pseudo_label[int(ankle_y):, :] = coco_person_parts["foot"]
                    
                    # 9.6 后臂（肩部x < x < 肘部x）
                    if shoulder is not None and hip is not None and elbow is not None:
                        shoulder_y, shoulder_x = shoulder[1], shoulder[0]
                        hip_y = hip[1]
                        elbow_x = elbow[0]
                        if shoulder_y < hip_y and shoulder_x < elbow_x:
                            y_start, y_end = int(shoulder_y), int(hip_y)
                            x_start, x_end = int(shoulder_x), int(elbow_x)
                            y_start = max(0, y_start)
                            y_end = min(H, y_end)
                            x_start = max(0, x_start)
                            x_end = min(W, x_end)
                            pseudo_label[y_start:y_end, x_start:x_end] = coco_person_parts["upper_arm"]
                    
                    # 9.7 前臂（肘部x < x < 手腕x）
                    if shoulder is not None and hip is not None and elbow is not None and wrist is not None:
                        shoulder_y, hip_y = shoulder[1], hip[1]
                        elbow_x, wrist_x = elbow[0], wrist[0]
                        if shoulder_y < hip_y and elbow_x < wrist_x:
                            y_start, y_end = int(shoulder_y), int(hip_y)
                            x_start, x_end = int(elbow_x), int(wrist_x)
                            y_start = max(0, y_start)
                            y_end = min(H, y_end)
                            x_start = max(0, x_start)
                            x_end = min(W, x_end)
                            pseudo_label[y_start:y_end, x_start:x_end] = coco_person_parts["forearm"]
                    
                    # 9.8 背景（非人体区域）
                    pseudo_label[~person_seg_mask] = coco_person_parts["background"]
                
                # 10. 调整尺寸并保存伪标签
                pseudo_label_img = Image.fromarray(pseudo_label).resize((16, 48), Image.Resampling.NEAREST)
                save_path = os.path.join(save_label_dir, os.path.splitext(img_name)[0] + ".png")
                pseudo_label_img.save(save_path)
                
                # 检查是否需要可视化
                if img_idx in vis_indices:
                    visualize_pseudo_labels(raw_img, pseudo_label_img, visualize_keypoints, vis_samples, vis_dir)
                    
            except Exception as e:
                logger.error(f"处理图像 {img_name} 时出错: {str(e)}")
                continue
    
    logger.info(f"伪标签生成完成，共处理{len(img_names)}张图像，保存至{save_label_dir}")

if __name__ == "__main__":
    cfg = Config()
    dataset_cfg = cfg.DATASETS[cfg.USE_DATASET]
    # 新增可视化参数：5个样本，保存到可视化目录
    vis_dir = os.path.join(dataset_cfg["pseudo_label"], "visualizations")
    generate_pseudo_labels(
        raw_img_dir=dataset_cfg["train_img"],
        save_label_dir=dataset_cfg["pseudo_label"],
        device="cuda" if torch.cuda.is_available() else "cpu",
        vis_samples=5,  # 可视化5个样本
        vis_dir=vis_dir  # 可视化结果保存目录
    )
    print(f"Pseudo-labels generated for {cfg.USE_DATASET}! Saved to {dataset_cfg['pseudo_label']}")
    if vis_dir:
        print(f"Visualizations saved to {vis_dir}")