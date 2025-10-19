import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

def plot_part_weights(img_path, part_weights, highlight_dim=None, save_path=None, part_names=None):
    """
    绘制部件权重可视化图（复现原论文图8、图10）
    Args:
        img_path: 行人图像路径
        part_weights: 7个部件的可解释权重（原论文最优7部件：头、上躯干、下躯干、大腿、小腿、脚、全身）
        highlight_dim: 高亮显示的部件索引（默认不高亮）
        save_path: 保存路径
        part_names: 部件名称列表（默认与原论文一致）
    """
    # 原论文7个部件名称（按PAP模块分割顺序）
    if part_names is None:
        part_names = ["Head", "Upper Torso", "Lower Torso", "Thigh", "Calf", "Foot", "Whole Body"]
    assert len(part_weights) == 7, f"Part weights must be 8-dimensional (as in paper), got {len(part_weights)}"

    # 1. 加载并处理图像（原论文输入尺寸384×128）
    img = Image.open(img_path).convert('RGB')
    img = img.resize((128, 384))  # (W, H)
    img_np = np.array(img)

    # 2. 创建画布（图像+权重条形图）
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    # 2.1 显示行人图像
    ax1.imshow(img_np)
    ax1.axis("off")
    ax1.set_title("Person Image", fontsize=14, fontweight="bold")

    # 2.2 绘制部件权重条形图（横向，与原论文图8一致）
    y_pos = np.arange(len(part_names))
    # 颜色区分：全身部件用红色，其他用蓝色（原论文图8风格）
    colors = ["#1f77b4" if i != 6 else "#d62728" for i in range(7)]
    bars = ax2.barh(y_pos, part_weights, color=colors, alpha=0.8)
    # 添加权重数值标签
    for i, (bar, w) in enumerate(zip(bars, part_weights)):
        color = 'red' if highlight_dim == i else 'blue'
        ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                 f"{w:.3f}", va="center", fontsize=10, color=color)
    # 设置坐标轴
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(part_names, fontsize=11)
    ax2.set_xlabel("Interpretable Weight (IWM Output)", fontsize=12, fontweight="bold")
    ax2.set_xlim(0, max(part_weights) + 0.1)
    ax2.set_title("Part-wise Interpretable Weights (IAPM)", fontsize=14, fontweight="bold")
    ax2.grid(axis="x", alpha=0.3)

    # 3. 添加图像与部件的对应分割线（横向分割逻辑）
    # 原论文384×128图像的7部件横向分割比例（H=384）
    part_heights = [0.1, 0.2, 0.15, 0.2, 0.15, 0.1, 1.0]  # 对应7个部件的高度占比
    cum_heights = np.cumsum([h * 384 for h in part_heights[:-1]])  # 前6个部件的累计高度
    for h in cum_heights:
        ax1.axhline(y=h, color="red", linestyle="--", linewidth=2, alpha=0.7)
        # 在分割线上标注部件名称
        ax1.text(135, h-10, part_names[len(ax1.lines)-1], color="red", fontsize=9, fontweight="bold")

    # 4. 调整布局并保存
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_weight_vs_subjective(query_idx, query_img_path, iwm_weights, subjective_scores, save_path):
    """
    绘制IWM权重与人群主观测评对比图
    Args:
        query_idx: 查询样本索引
        query_img_path: 查询图像路径
        iwm_weights: IWM输出的6个局部部件权重（排除全身）
        subjective_scores: 人群主观测评的6个部件相对得分（排除全身）
        save_path: 保存路径
    """
    part_names = ["Head", "Upper Torso", "Lower Torso", "Thigh", "Calf", "Foot"]
    assert len(iwm_weights) == 6 and len(subjective_scores) == 6, "Must input 6 local parts (exclude whole body)"

    # 1. 创建画布
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))
    # 1.1 显示图像
    img = Image.open(query_img_path).convert('RGB').resize((128, 384))
    ax1.imshow(img)
    ax1.axis("off")
    ax1.set_title(f"Query Sample {query_idx+1}", fontsize=12, fontweight="bold")

    # 1.2 绘制IWM权重
    x_pos = np.arange(len(part_names))
    ax2.bar(x_pos, iwm_weights, color="#1f77b4", alpha=0.8)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(part_names, rotation=45, ha="right")
    ax2.set_ylabel("IWM Weight", fontsize=11, fontweight="bold")
    ax2.set_title("IAPM Interpretable Weights", fontsize=12, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    # 1.3 绘制主观测评得分
    ax3.bar(x_pos, subjective_scores, color="#2ca02c", alpha=0.8)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(part_names, rotation=45, ha="right")
    ax3.set_ylabel("Subjective Relative Score", fontsize=11, fontweight="bold")
    ax3.set_title("Human Subjective Evaluation", fontsize=12, fontweight="bold")
    ax3.grid(axis="y", alpha=0.3)

    # 2. 调整布局并保存
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()