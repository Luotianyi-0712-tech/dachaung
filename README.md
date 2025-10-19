第一次提交
# REID_BLIP
行人重识别（文本+图像）

# BLIP-IAPM-ReID：基于可解释注意力部件模型的多模态行人重识别工程
本工程基于周勇等人的论文《基于可解释注意力部件模型的行人重识别方法》（《自动化学报》2023, 49(10)），融合BLIP多模态模型与原论文提出的IAPM（可解释注意力部件模型），实现“图像-文本”双向检索的行人重识别系统，解决传统ReID可解释性弱、仅依赖图像的局限。


## 一、工程背景与论文对应关系
### 1. 核心目标
- 复现原论文IAPM模型的核心模块（PAP、PS、IWM、SPTL），确保性能与原论文一致（Market-1501 Rank-1≥95.2%、DukeMTMC-reID Rank-1≥88.0%、CUHK03 Rank-1≥72.6%）。
- 扩展原论文的纯图像ReID为多模态ReID（图像+文本），基于BLIP实现跨模态特征融合，提升场景适应性。

### 2. 与原论文模块映射
| 工程模块                | 论文对应章节       | 功能描述                                                                 |
|-------------------------|--------------------|--------------------------------------------------------------------------|
| `model/iapm/pap_module.py` | 2.1节              | 注意力部件对齐池化，横向分割人体为7个部件（原论文图2），解决部件不对齐问题。 |
| `model/iapm/ps_module.py`  | 2.1节              | 局部分割约束，减少部件重叠特征，强化PAP的规范性（依赖COCO伪标签）。         |
| `model/iapm/iwm_module.py` | 2.2节              | 可解释权重生成模块，根据部件显著性生成权重，量化部件对模型的影响（原论文图4）。 |
| `model/iapm/sptl_loss.py`  | 2.2节              | 显著部件三元损失，训练IWM模块，提升识别精度与可解释性（原论文式(7)-(10)）。  |
| `codes/train.py`          | 3.1-3.4节          | 实现原论文总损失函数（式12：L=ID+PS+λSPTL+βC）与训练流程（100轮迭代、学习率调度）。 |
| `codes/test.py`           | 3.3-3.5节          | 计算原论文评价指标（Rank-1、mAP），对比多模态检索性能，可视化IWM权重（原论文图8、10）。 |

参考完整架构文档：docs/architecture.md
相关流程图：docs/diagrams.md

## 二、环境配置
### 1. 硬件要求
- GPU：NVIDIA Tesla P100/RTX 3090及以上（16GB+显存，适配BatchSize=128）。
- CPU：Intel Xeon E5/V3及以上。
- 内存：40GB+（原论文实验配置）。

### 2. 软件依赖
```bash
# 基础依赖
pip install torch==2.0.1 torchvision==0.15.2 transformers==4.30.2 pillow==10.0.0
# 工具依赖
pip install tqdm==4.65.0 tensorboardX==2.6 pandas==2.0.3 matplotlib==3.7.2 scipy==1.10.1
# 数据集处理依赖
pip install opencv-python==4.8.0.76 numpy==1.24.3

BLIP-IAPM-ReID/                # 工程根目录
├─ data/                       # 数据集相关（严格按论文需求整理）
│  ├─ raw/                     # 原始数据集（Market-1501/DukeMTMC-reID/CUHK03）
│  ├─ processed/               # 预处理后数据（图像/文本/伪标签，适配训练）
│  ├─ preprocess.py            # 图像Resize+文本标注生成（适配BLIP）
│  └─ pseudo_label_generate.py # 生成PS模块伪标签（原论文2.1节监督信号）
├─ model/                      # 模型定义（与论文模块一一对应）
│  ├─ blip/                    # BLIP多模态基础模块（适配IAPM的ViT/BERT）
│  ├─ iapm/                    # 原论文IAPM核心模块（PAP/PS/IWM/SPTL）
│  └─ blip_iapm.py             # 融合模型（BLIP+IAPM+跨模态融合）
├─ codes/                      # 核心逻辑（按论文实验流程设计）
│  ├─ train.py                 # 训练脚本（原论文总损失+学习率调度）
│  ├─ test.py                  # 测试脚本（论文评价指标+可解释性可视化）
│  ├─ utils/                   # 工具类（日志/指标/可视化，支持论文实验记录）
│  └─ config.py                # 配置文件（论文参数统一管理，确保可复现）
├─ pretrained/                 # 预训练权重（论文IAPM权重+BLIP权重）
│  ├─ blip-base.pth            # BLIP基础权重
│  ├─ resnet50_iapm_init.pth   # 原论文ResNet50-IAPM权重
│  └─ weight_convert.py        # 权重转换（ResNet50→ViT，适配工程模型）
├─ output/                     # 输出结果（论文实验报告所需）
│  ├─ logs/                    # 训练日志（TensorBoard+文本，支持论文复现）
│  ├─ checkpoints/             # 模型 checkpoint（按论文迭代次数保存）
│  └─ results/                 # 测试结果（指标+图表，与论文格式一致）
└─ README.md                   # 工程说明（论文对应关系+实验流程）

