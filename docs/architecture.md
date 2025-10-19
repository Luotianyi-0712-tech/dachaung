# BLIP-IAPM-ReID 项目整体架构

本工程实现了一个多模态（图像+文本）行人重识别（ReID）系统，融合了 BLIP 的视觉/文本编码能力与 IAPM（可解释注意力部件模型）提出的可解释性部件建模。整体目标是同时获得高检索精度与良好的可解释性（部件级权重）。

- 视觉主干：ViT（来自 BLIP 视觉模型）
- 文本主干：BERT（来自 BLIP 文本模型）
- IAPM 组件：PAP（部件对齐池化）、PS（局部分割约束）、IWM（可解释权重生成）、SPTL（三元部件损失）
- 跨模态融合：Multi-Head Attention（部件特征与文本序列特征）
- 监督目标：ID 分类 + PS 分割 + SPTL 显著部件三元 + Center 类中心约束
- 评估指标：Rank-1 与 mAP

---

## 1. 目录结构（关键模块）

- codes/
  - train.py：训练主循环（损失计算、优化器/调度器、日志、评估、checkpoint）
  - test.py：测试与可解释性可视化示例
  - val.py：验证流程（含可选的 gt_bbox/gt_query 过滤逻辑）
  - utils/
    - data_loader.py：数据集与 DataLoader 封装（图像/文本处理、伪标签加载、采样器、collate）
    - samplers.py：ReID 常用 RandomIdentitySampler
    - metric.py：简化的 Rank-1/mAP 计算
    - logger.py：TensorBoard 与文本日志
    - visualization.py：部件权重与主观评分可视化
  - config.py：集中式配置（数据集路径、超参数、日志输出目录等）
- model/
  - blip/
    - vit.py：ViTForReID，输出 2D 特征图 (B, 768, 24, 8)
    - bert.py：BertForReID，输出文本全局与序列特征 (B, 768), (B, L, 768)
    - blip.py：BlipForReID 封装与处理器 BlipProcessor（AutoImageProcessor+AutoTokenizer）
  - iapm/
    - pap_module.py：PAP 部件对齐池化（横向 7 段）
    - ps_module.py：PS 局部分割约束（反卷积上采样 + 1×1 分类，使用伪标签监督）
    - iwm_module.py：IWM 可解释权重（两层 MLP + softmax）
    - sptl_loss.py：SPTL 显著部件三元损失
  - blip_iapm.py：BLIP + IAPM 融合模型与 ID 分类头
- data/
  - preprocess.py：图像尺寸规范、文本标注生成示例
  - pseudo_label_generate.py：生成/处理 PS 伪标签
- pretrained/：放置本地预训练权重（如 BLIP 权重）
- output/：训练日志、checkpoint 与可视化结果

---

## 2. 数据与预处理

- 图像尺寸：统一为 384×128（H×W），与论文一致。
- 文本标注：以 JSON 形式维护 {image_path: caption}。
- 伪标签：PS 模块使用的分割伪标签（例如 8 类：背景、头、躯干、前臂、后臂、大腿、小腿、脚）。
- 处理器：BlipProcessor 同时封装 AutoImageProcessor 与 AutoTokenizer，用于生成
  - pixel_values: (B, 3, 384, 128)
  - input_ids/attention_mask: (B, L)
- 采样：随机身份采样器 RandomIdentitySampler 可用于每批包含多身份多实例（训练中可按需启用）。

---

## 3. 模型架构（前向与融合）

以 model/blip_iapm.py 中的 BlipIAPMReID 为核心：

1) 文本编码（BERT）
- 输入：input_ids, attention_mask
- 输出：
  - text_feat: (B, 768)
  - text_seq_feat: (B, L, 768)

2) 视觉编码（ViT）
- 输入：pixel_values (B, 3, 384, 128)
- 输出：feat_map (B, 768, 24, 8)

3) PS 局部分割约束（训练时）
- 结构：ConvTranspose2d 上采样 + 1×1 卷积逐像素分类
- 监督：对 2H×2W 的预测与伪标签做逐类平均的 CE 损失
- 前向：对特征图施加约束并返回原尺寸特征图；保存 ps_loss

4) PAP 部件对齐池化
- 方法：沿 H 方向按比例将特征图切分为 7 段（含“全身”一段），并对每段做自适应池化
- 输出：part_feats (B, 7, 768)

5) IWM 可解释权重
- 结构：两层 MLP（in→hidden→1），对 7 个部件分别产生得分后经 softmax 得到权重
- 输出：part_weights (B, 7)，各部件权重和为 1

6) 跨模态注意力融合
- 步骤：将加权后的部件特征作为 query，与 text_seq_feat 作为 key/value 做多头注意力
- 聚合：对注意力输出做时序平均得到 cross_feat (B, 768)

7) ID 分类头
- 线性层将 cross_feat 投影到身份数上，计算 ID 分类损失

---

## 4. 训练目标与总损失

- ID 分类损失：CrossEntropy(logits_id, labels)
- PS 分割损失：ps_module.loss（逐类均衡的 CE）
- SPTL 显著部件三元损失：对每个部件在 batch 内挖掘 hardest positive/negative 距离并施加 margin α（默认 1.2）
- Center 类中心损失：对 cross_feat 与其类别中心做 MSE，类别中心以 EMA 方式更新
- 总损失：L = ID + PS + λ·SPTL + β·Center（默认 λ=1.0, β=5e-4）

---

## 5. 训练流程（codes/train.py）

- 配置：Config 提供数据路径、学习率组、调度器、损失系数与输出目录等；启动时自动创建必要目录
- 数据：ReIDDataLoader 将图像/文本/伪标签组装为 batch；必要时校验标签范围
- 模型：加载 BLIP 权重（本地）、构建 IAPM 模块、注册 class_centers/class_counts 缓存
- 优化：
  - 视觉 backbone 较小学习率（保护预训练）
  - IWM + 文本编码器较大学习率
  - ID 头 + PS 模块较大学习率
  - MultiStepLR 里程碑衰减
- 循环：前向→计算分项损失→合成总损失→反向→step→日志→周期性评估→保存最优/周期性 checkpoint
- 可视化：epoch 末示例化地保存部件权重图与“权重-主观评分”对比图

---

## 6. 验证与测试

- 验证（codes/val.py）：
  - 从验证集提取 cross_feat 并计算距离矩阵
  - 可选使用 gt_bbox/gt_query 进行过滤（仅保留有效样本/匹配）
  - 输出 Rank-1 与 mAP 并记录日志

- 测试（codes/test.py）：
  - 从 gallery/query 中分别提取融合特征与部件权重
  - 计算 Image→Image/Text→Image/Image→Text 的 Rank-1/mAP（示例中复用同一特征接口）
  - 保存部件权重可视化图

评估核心：
- compute_distance_matrix + evaluate_rank（或 utils/metric.py 的简化版本）
- 默认提供 cosine/euclidean 两种度量

---

## 7. 配置与路径

- codes/config.py 统一管理：
  - 数据集路径（Market-1501、DukeMTMC-reID、CUHK03）
  - 训练超参数（batch size、epoch、初始学习率、调度器里程碑、损失系数）
  - 日志/可视化/检查点目录
  - BLIP 本地权重目录与处理器加载方式
- 运行前请根据本地环境检查/同步：
  - data/raw 与 data/processed 的组织（图像、文本、伪标签）
  - pretrained/ 与 blip_model/ 的权重放置

---

## 8. 可解释性与可视化

- visualization.plot_part_weights：绘制 7 个部件的 IWM 权重，支持高亮单部件
- visualization.plot_weight_vs_subjective：对比 6 个局部部件权重与主观评分
- 训练结束后可在 output/results/visualizations/ 下查看结果

---

## 9. 关键类与调用关系（简版）

- BlipProcessor → 统一图像/文本预处理
- BlipForReID → 封装 ViTForReID（vision）与 BertForReID（text）
- BlipIAPMReID → PAP + PS + IWM + Cross-Modal Attn + ID Head + buffers(class_centers/counts)
- SPTLLoss → 显著部件三元损失
- ReIDDataset/DataLoader → 图像/文本/伪标签/采样
- RandomIdentitySampler → 身份均衡采样（按需）

---

如需快速上手：准备数据与 BLIP 权重 → 修改 codes/config.py 中路径 → 运行 codes/train.py，过程中会生成日志、检查点与可视化结果。
