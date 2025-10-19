import os
import sys
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 加入工程根目录到搜索路径
current_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(current_path)
if root_path not in sys.path:
    sys.path.insert(0, root_path)
    
from model.blip_iapm import BlipIAPMReID
from codes.utils.data_loader import ReIDDataLoader
from codes.utils.logger import TensorboardLogger
from codes.utils.visualization import plot_part_weights, plot_weight_vs_subjective
from codes.config import Config
from model.iapm.sptl_loss import SPTLLoss


# -------------------------- 1. 自定义 ReID 指标计算 --------------------------
def compute_distance_matrix(feat1, feat2, metric="cosine"):
    """
    计算特征距离矩阵（支持余弦距离和欧氏距离）
    Args:
        feat1: 特征矩阵1，shape=(N, D)，N为样本数，D为特征维度
        feat2: 特征矩阵2，shape=(M, D)
        metric: 距离度量方式，可选 "cosine"（余弦距离）或 "euclidean"（欧氏距离）
    Returns:
        dist_matrix: 距离矩阵，shape=(N, M)，dist_matrix[i][j] 表示 feat1[i] 与 feat2[j] 的距离
    """
    # 特征归一化（余弦距离计算需先归一化）
    if metric == "cosine":
        feat1 = F.normalize(feat1, p=2, dim=1)
        feat2 = F.normalize(feat2, p=2, dim=1)
    
    # 计算距离：余弦距离 = 1 - 余弦相似度；欧氏距离 = sqrt(||a-b||²)
    if metric == "cosine":
        sim_matrix = torch.matmul(feat1, feat2.T)  # 余弦相似度矩阵，shape=(N, M)
        dist_matrix = 1 - sim_matrix  # 余弦距离（范围0~2，值越小越相似）
    elif metric == "euclidean":
        dist_matrix = torch.cdist(feat1, feat2, p=2)  # 欧氏距离（值越小越相似）
    else:
        raise ValueError(f"不支持的距离度量方式: {metric}，可选 'cosine' 或 'euclidean'")
    
    return dist_matrix

def evaluate_rank(dist_matrix, pids, camids, max_rank=50, use_metric_cuhk03=False):
    """
    计算 ReID 核心指标（Rank-1、mAP）
    Args:
        dist_matrix: 距离矩阵，shape=(N, N)，N为验证集样本总数
        pids: 身份ID列表，shape=(N,)
        camids: 摄像头ID列表，shape=(N,)
        max_rank: 评估的最大Rank值（默认50）
        use_metric_cuhk03: 是否使用CUHK03数据集的特殊评估方式（默认False）
    Returns:
        rank1: Rank-1准确率
        mAP: 平均精度均值
    """
    num_query = dist_matrix.shape[0]  # 每个样本作为查询样本
    all_cmc = torch.zeros(num_query, max_rank, dtype=torch.float32)  # 存储每个查询的CMC曲线
    all_ap = torch.zeros(num_query, dtype=torch.float32)  # 存储每个查询的AP

    for q_idx in range(num_query):
        # 1. 筛选合法的gallery样本（排除同摄像头的同一身份，避免自匹配）
        q_pid = pids[q_idx]
        q_camid = camids[q_idx]
        
        # 距离矩阵第q_idx行：当前查询样本与所有gallery样本的距离
        dists = dist_matrix[q_idx]
        
        # 过滤条件：gallery样本不能是"同摄像头+同身份"（自匹配）
        mask = (pids != q_pid) | (camids != q_camid)
        gallery_dists = dists[mask]
        gallery_pids = pids[mask]
        
        # 若没有合法gallery样本，跳过该查询
        if len(gallery_pids) == 0:
            all_cmc[q_idx] = 0
            all_ap[q_idx] = 0
            continue

        # 2. 对距离排序（从小到大，距离越小越相似）
        sorted_indices = torch.argsort(gallery_dists)
        sorted_pids = gallery_pids[sorted_indices]

        # 3. 计算CMC（Cumulative Matching Characteristics）
        # 标记排序后的样本是否为目标身份（1=匹配，0=不匹配）
        matches = (sorted_pids == q_pid).float()
        # CMC曲线：前k个样本中是否有匹配（cumprod确保只要前k个有1，后续都为1）
        cmc = matches.cumsum(dim=0)
        # 截取前max_rank个结果，不足则补0
        if len(cmc) > max_rank:
            cmc = cmc[:max_rank]  # 截断：超过max_rank的部分丢弃
        # 补全：若cmc长度不足max_rank，用0补到max_rank
        if len(cmc) < max_rank:
            cmc = torch.cat([
                cmc, 
                torch.zeros(max_rank - len(cmc), dtype=torch.float32, device=cmc.device)  # 新增device参数，避免设备不匹配
            ], dim=0)
        
        cmc[cmc > 1] = 1  # 确保CMC曲线最大值为1（避免多个匹配导致超过1）
        all_cmc[q_idx] = cmc

        # 4. 计算AP（Average Precision，平均精度）
        num_matches = matches.sum().int()
        if num_matches == 0:
            all_ap[q_idx] = 0
            continue
        
        # 计算每个匹配位置的精度
        precision = matches.cumsum(dim=0) / torch.arange(1, len(matches) + 1, dtype=torch.float32)
        # 对所有匹配位置的精度取平均（AP核心逻辑）
        ap = (precision * matches).sum() / num_matches
        all_ap[q_idx] = ap

    # 5. 计算最终指标（平均所有查询的结果）
    rank1 = all_cmc[:, 0].mean().item()  # Rank-1：取第0位（top-1）的平均准确率
    mAP = all_ap.mean().item()           # mAP：所有查询AP的平均值

    return rank1, mAP


# -------------------------- 2. 评估函数（复用原逻辑，调用自定义指标） --------------------------
def evaluate_reid(model, val_loader, device):
    """评估ReID模型在验证集上的性能（Rank-1和mAP）"""
    model.eval()
    all_features = []
    all_pids = []
    all_camids = []
    
     # 创建默认文本输入（空字符串的编码，需与训练时保持一致）
    default_input_ids = torch.zeros((1, 1), dtype=torch.long, device=device)  # 假设[CLS]对应0
    default_attention_mask = torch.ones((1, 1), dtype=torch.long, device=device)
    
    with torch.no_grad():
        for batch in val_loader:
            # 适配数据加载器的输出格式（dict或tuple）
            if isinstance(batch, dict):
                images = batch["image"]
                pids = batch["label"]
                camids = batch.get("camid", torch.zeros_like(pids, device=pids.device))
            else:
                images, pids, camids = batch[:3]
            
            # 数据移至目标设备
            images = images.to(device)
            pids = pids.to(device)
            camids = camids.to(device)
            
            image_inputs = {"pixel_values": images}
            batch_size = images.shape[0]
            text_inputs = {
                "input_ids": default_input_ids.repeat(batch_size, 1),
                "attention_mask": default_attention_mask.repeat(batch_size, 1)
            }
            outputs = model(image_inputs=image_inputs, text_inputs=text_inputs)
            # 提取模型特征（需匹配Blip-IAPM模型的eval模式输出）
            features = outputs[0]
            
            # 收集结果（统一移至CPU，避免GPU内存占用）
            all_features.append(features.cpu())
            all_pids.append(pids.cpu())
            all_camids.append(camids.cpu())
    
    # 拼接所有样本的特征和标签
    all_features = torch.cat(all_features, dim=0)
    all_pids = torch.cat(all_pids, dim=0)
    all_camids = torch.cat(all_camids, dim=0)
    
    # 调用自定义函数计算指标
    dist_matrix = compute_distance_matrix(all_features, all_features, metric="cosine")
    rank1, mAP = evaluate_rank(
        dist_matrix, all_pids, all_camids,
        max_rank=50, use_metric_cuhk03=False
    )
    
    # 评估后切回训练模式
    model.train()
    return rank1, mAP


# -------------------------- 3. 训练函数（保持原逻辑不变） --------------------------
def train():
    # 1. 初始化配置与设备
    cfg = Config()
    cfg.init_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_cfg = cfg.DATASETS[cfg.USE_DATASET]
    print(f"Using device: {device}")

    # 2. 加载数据（训练集+验证集）
    # 训练集（含伪标签）
    print("加载训练集数据...")
    train_data_loader = ReIDDataLoader(
        img_dir=dataset_cfg["train_img"],
        caption_path=dataset_cfg["caption"],
        pseudo_label_path=dataset_cfg["pseudo_label"],
        batch_size=cfg.batch_size,
        num_workers=8,
        is_train=True,
        # size=dataset_cfg["image_size"]
        size=[384, 128],  
        use_identity_sampler=False,
        num_ids_per_batch=16,
        num_instances_per_id=1
    )
    train_loader = train_data_loader.get_loader()
    with tqdm(total=len(train_data_loader.dataset), desc="训练集加载进度") as pbar:
        for _ in train_data_loader.dataset:
            pbar.update(1)
    num_train_ids = train_data_loader.num_ids
    
    # 验证集
    print("加载验证集数据...")
    val_data_loader = ReIDDataLoader(
        img_dir=dataset_cfg["test_img"],
        caption_path=dataset_cfg.get("test_caption", None),  # 验证集可能无文本，设为None
        batch_size=cfg.batch_size,
        num_workers=4,
        is_train=False,
        # size=dataset_cfg["image_size"]
        size=[384, 128],
        use_identity_sampler=False,
        num_ids_per_batch=16,
        num_instances_per_id=1
    )
    val_loader = val_data_loader.get_loader()
    with tqdm(total=len(val_data_loader.dataset), desc="验证集加载进度") as pbar:
        for _ in val_data_loader.dataset:
            pbar.update(1)
    
    print(f"训练集身份总数（num_train_ids）: {num_train_ids}")
    print(f"标签最小值: {min(train_data_loader.dataset.labels)}")
    print(f"标签最大值: {max(train_data_loader.dataset.labels)}")

    # 校验标签范围：若标签从1开始，转为0开始；若存在超界标签，抛出错误
    train_labels = train_data_loader.dataset.labels
    if min(train_labels) == 1 and max(train_labels) == num_train_ids:
        # 标签从1开始 → 转为0开始（模型分类头要求）
        train_data_loader.dataset.labels = [pid - 1 for pid in train_labels]
        print("标签已从1-based转为0-based")
    elif max(train_labels) >= num_train_ids or min(train_labels) < 0:
        # 存在超界标签 → 抛出错误并提示
        raise ValueError(
            f"标签范围异常！类别数={num_train_ids}，但标签范围=[{min(train_labels)}, {max(train_labels)}]，"
            "请检查 data_loader.py 的 _parse_img_labels 方法是否正确提取pid"
        )

    # 3. 初始化模型与损失函数
    model = BlipIAPMReID(
        blip_pretrained_path=cfg.blip_pretrained_path,
        num_parts=cfg.num_parts,
        part_feat_dim=cfg.IAPM_CONFIG["in_channels"],
        num_ids=num_train_ids
    ).to(device)
    
    sptl_loss_fn = SPTLLoss(alpha=cfg.SPTL_ALPHA, num_parts=cfg.num_parts)  # 部件损失
    center_loss_fn = torch.nn.MSELoss()  # 中心损失

    # 4. 优化器与学习率调度（分模块设置学习率）
    param_groups = [
        # BLIP视觉 backbone（学习率较低，避免破坏预训练权重）
        {
            "params": model.blip.vision_encoder.parameters(),
            "lr": cfg.INIT_LR["backbone"]
        },
        # IWM模块 + BLIP文本 encoder（学习率较高，重点更新）
        {
            "params": list(model.iwm_module.parameters()) + list(model.blip.text_encoder.parameters()),
            "lr": cfg.INIT_LR["iwm_em"]
        },
        # ID分类头 + PS伪标签模块（学习率较高，重点更新）
        {
            "params": list(model.id_head.parameters()) + list(model.ps_module.parameters()),
            "lr": cfg.INIT_LR["iwm_em"]
        }
    ]
    optimizer = optim.SGD(
        param_groups,
        momentum=cfg.OPTIMIZER["momentum"],  # 动量加速收敛
        weight_decay=cfg.OPTIMIZER["weight_decay"]  # L2正则化防过拟合
    )
    
    # -------------------------- 添加参数检查代码 --------------------------
    all_params = set(name for name, _ in model.named_parameters())
    optimized_params = set()
    for group in param_groups:
        for p in group["params"]:
            for name, param in model.named_parameters():
                if param is p:
                    optimized_params.add(name)
    print("未被优化的参数:", all_params - optimized_params)
    # ----------------------------------------------------------------------

    
    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=cfg.LR_SCHEDULER["milestones"],  # 学习率衰减节点
        gamma=cfg.LR_SCHEDULER["gamma"]  # 衰减系数
    )

    # 5. 初始化日志与最优指标（跟踪训练过程与最佳模型）
    logger = TensorboardLogger(log_dir=cfg.LOG_DIR)
    best_rank1 = 0.0  # 初始最优Rank-1准确率
    best_model_path = os.path.join(cfg.CHECKPOINT_DIR, f"best_model_{cfg.USE_DATASET}.pth")

    # 6. 模型状态初始化（类别中心与计数，用于中心损失计算）
    model.train()
    model.class_counts = torch.zeros(num_train_ids, device=device)  # 每个类别的样本计数
    if not hasattr(model, "class_centers"):
        # 注册类别中心（buffer不参与梯度更新，仅用于计算损失）
        model.register_buffer(
            "class_centers",
            torch.zeros(num_train_ids, cfg.IAPM_CONFIG["in_channels"], device=device)
        )

    # 7. 训练主循环（核心迭代逻辑）
    for epoch in range(cfg.MAX_EPOCHS):
        total_loss = 0.0
        epoch_losses = {
            "id_loss": 0.0,    # ID分类损失
            "sptl_loss": 0.0,  # 部件损失
            "ps_loss": 0.0,    # 伪标签损失
            "center_loss": 0.0 # 中心损失
        }
        
        # 进度条显示训练过程
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.MAX_EPOCHS}")
        for batch in pbar:
            # 数据预处理（提取batch数据并移至目标设备）
            image_inputs = {"pixel_values": batch["image"].to(device)}  # BLIP图像输入格式
            text_inputs = {
                "input_ids": batch["text_input_ids"].to(device),
                "attention_mask": batch["text_attention_mask"].to(device)
            }
            labels = batch["label"].to(device)  # 真实身份标签
            pseudo_labels = batch["pseudo_label"].to(device)  # PS模块伪标签

            # -------------------------- 伪标签检查代码（新增） ----------------------
            if pseudo_labels.min() < 0 or pseudo_labels.max() >= num_train_ids:
                raise ValueError(
                    f"伪标签超出范围: min={pseudo_labels.min()}, max={pseudo_labels.max()}, "
                    f"当前训练集类别总数num_train_ids={num_train_ids}"
                )
            # ----------------------------------------------------------------------
        
            # 前向传播（模型输出多维度结果）
            outputs = model(
                image_inputs=image_inputs,
                text_inputs=text_inputs,
                labels=labels,
                pseudo_labels=pseudo_labels
            )
            part_weights = outputs[1]  # 部件权重
            part_feats = outputs[2]  # 部件特征
            cross_feat = outputs[3]  # 跨模态融合特征
            losses = outputs[4]          # 模型内部计算的损失（ID+PS）

           # 更新类别中心（指数移动平均，避免单次批次波动）
            unique_labels = torch.unique(labels)  # 当前批次的唯一身份ID
            # 1. 复制原始计数和中心（避免原地修改）
            class_counts_copy = model.class_counts.detach()
            class_centers_copy = model.class_centers.detach()

            for lbl in unique_labels:
                mask = (labels == lbl)  # 筛选当前类别的样本
                class_feat = cross_feat[mask]  # 当前类别的融合特征
                mean_feat = torch.mean(class_feat, dim=0)  # 特征均值
                
                # 2. 更新计数副本（非原地）
                class_counts_copy[lbl] += 1
                # 3. 更新中心副本（非原地）
                momentum = 0.9 if class_counts_copy[lbl] > 1 else 1.0  # 基于副本的计数判断动量
                class_centers_copy[lbl.item()] = class_centers_copy[lbl.item()] * momentum + (1 - momentum) * mean_feat

            # 4. 赋值回原变量（覆盖原始值，但不破坏前向计算图的依赖）
            model.class_counts = class_counts_copy
            model.class_centers = class_centers_copy
            
            # 计算总损失（原论文式12：L = ID + PS + λ*SPTL + β*Center）
            sptl_loss = sptl_loss_fn(part_feats, part_weights, labels)  # 部件损失
            center_loss = center_loss_fn(
                cross_feat, 
                torch.index_select(getattr(model, "class_centers"), 0, labels)  # 索引当前批次的类别中心
            )  # 中心损失
            total_batch_loss = (
                losses["id_loss"] + 
                losses["ps_loss"] + 
                cfg.LOSS_COEFF["lambda_sptl"] * sptl_loss + 
                cfg.LOSS_COEFF["beta_center"] * center_loss
            )

            # 反向传播与参数更新
            optimizer.zero_grad()  # 清空梯度（避免累积）
            # 在反向传播前添加损失值检查
            print(f"ID Loss: {losses['id_loss'].item()}, SPTL Loss: {(sptl_loss.item() if isinstance(sptl_loss, torch.Tensor) else sptl_loss)}, PS Loss: {losses['ps_loss'].item()}, Center Loss: {center_loss.item()}")
            if torch.isnan(total_batch_loss) or torch.isinf(total_batch_loss):
                raise ValueError("Loss contains NaN/inf, cannot backward")

            total_batch_loss.backward()  # 原反向传播代码
            optimizer.step()  # 更新参数

            # 损失统计（按样本数加权，避免批次大小影响）
            batch_size = batch["image"].shape[0]
            total_loss += total_batch_loss.item() * batch_size
            
            # 分项损失统计
            epoch_losses["id_loss"] += losses["id_loss"].item() * batch_size
            epoch_losses["sptl_loss"] += (sptl_loss.item() if isinstance(sptl_loss, torch.Tensor) else sptl_loss) * batch_size
            epoch_losses["ps_loss"] += losses["ps_loss"].item() * batch_size
            epoch_losses["center_loss"] += center_loss.item() * batch_size

            # 进度条实时显示当前批次损失
            pbar.set_postfix({
                "Total Loss": f"{total_batch_loss.item():.4f}",
                "ID Loss": f"{losses['id_loss'].item():.4f}",
                "SPTL Loss": f"{(sptl_loss.item() if isinstance(sptl_loss, torch.Tensor) else sptl_loss):.4f}",
                "PS Loss": f"{losses['ps_loss'].item():.4f}",
                "Center Loss": f"{center_loss.item():.4f}"
            })

        # Epoch结束：计算平均损失并记录日志
        dataset_size = len(train_data_loader.dataset)
        avg_loss = total_loss / dataset_size
        for key in epoch_losses:
            epoch_losses[key] /= dataset_size  # 分项损失平均值

        # 记录TensorBoard日志（训练损失）
        logger.log_scalar("train/total_loss", avg_loss, epoch)
        logger.log_scalar("train/id_loss", epoch_losses["id_loss"], epoch)
        logger.log_scalar("train/sptl_loss", epoch_losses["sptl_loss"], epoch)
        logger.log_scalar("train/ps_loss", epoch_losses["ps_loss"], epoch)
        logger.log_scalar("train/center_loss", epoch_losses["center_loss"], epoch)

        # 学习率调度
        lr_scheduler.step()

        # 验证与模型保存
        current_rank1, current_map = evaluate_reid(model, val_loader, device)
        logger.log_scalar("val/rank1", current_rank1, epoch)
        logger.log_scalar("val/mAP", current_map, epoch)
        torch.cuda.empty_cache()  # 清理显存，避免碎片化
        print(f"Epoch [{epoch+1}/{cfg.MAX_EPOCHS}] "
              f"Val Rank-1: {current_rank1:.4f}, Val mAP: {current_map:.4f}")

        # 保存最优模型
        if current_rank1 > best_rank1:
            best_rank1 = current_rank1
            torch.save({
                "epoch": epoch + 1,
                "dataset": cfg.USE_DATASET,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "class_centers": model.class_centers,
                "class_counts": model.class_counts,
                "best_rank1": best_rank1,
                "avg_loss": avg_loss
            }, best_model_path)
            print(f"✅ New Best Model Saved! Best Rank-1: {best_rank1:.4f}")
        else:
            print(f"❌ No Improvement. Current Best Rank-1: {best_rank1:.4f}")

        # 定期保存检查点
        if (epoch + 1) % 10 == 0 or (epoch + 1) == cfg.MAX_EPOCHS:
            checkpoint_path = os.path.join(cfg.CHECKPOINT_DIR, 
                                         f"epoch_{epoch+1}_{cfg.USE_DATASET}.pth")
            torch.save({
                "epoch": epoch + 1,
                "dataset": cfg.USE_DATASET,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "class_centers": model.class_centers,
                "class_counts": model.class_counts,
                "avg_loss": avg_loss
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    print(f"Training finished for {cfg.USE_DATASET}! Best Rank-1: {best_rank1:.4f}")
    
    # 确保有可用的部件权重数据
    if 'part_weights' in locals():
        # 获取部件权重的维度数量（假设 shape 为 [batch_size, num_dims]）
        num_dims = part_weights.shape[1] if len(part_weights.shape) > 1 else len(part_weights)
        
        # 选择第一个样本的权重进行可视化（如果是批次数据）
        sample_weights = part_weights[0].cpu().detach().numpy() if len(part_weights.shape) > 1 else part_weights.cpu().detach().numpy()
        
        # 为每一维单独生成图片
        for dim_idx in range(num_dims):
            # 生成部件权重可视化（突出显示当前维度）
            plot_part_weights(
                img_path="./data/raw/Market-1501/query/0001_c1s1_001051_00.jpg",
                part_weights=sample_weights,  # 传入全部权重用于对比
                highlight_dim=dim_idx,        # 高亮当前维度（需要修改plot_part_weights支持）
                save_path=os.path.join("./output/results/visualizations", f"part_weight_dim_{dim_idx}.png")
            )
            print(f"维度 {dim_idx} 可视化完成，已保存7个部件的权重对比图")
            
    # 确保有可用的部件权重数据
    if 'part_weights' in locals():
        # 1. 解析部件权重的维度结构（关键：确保每个维度对应6个部件）
        # 假设 part_weights 形状为 [batch_size, num_dims, num_parts]
        # 其中 num_parts=6（符合函数要求的6个局部部件）
        batch_size, num_parts = part_weights.shape  # 提取维度信息
        assert num_parts == 6, f"部件数量必须为6，当前为{num_parts}，请检查模型输出的part_weights结构"

        # 选择第一个样本的权重（避免批次干扰）
        sample_weights = part_weights[0].cpu().detach().numpy()  # 形状：[num_dims, 6]（维度×6部件）
        
        for dim_idx in range(num_dims):
            # 核心：当前维度的iwm_weights取6个部件的权重（满足函数对6个局部部件的要求）
            current_dim_6parts_weights = sample_weights[dim_idx]  # 形状：[6]（6个部件权重）
            
            # -------------------------- 2. 生成权重与主观评分对比图（关键修复） --------------------------
            plot_weight_vs_subjective(
                query_idx=0,
                query_img_path="./data/raw/Market-1501/query/0001_c1s1_001051_00.jpg",
                iwm_weights=current_dim_6parts_weights,  # 传入6个部件权重（解决长度问题）
                subjective_scores=[0.3, 0.6, 0.8, 0.5, 0.4, 0.7],  # 对应6个部件的主观评分（需根据实际标注调整）
                save_path=os.path.join("./output/results/visualizations", f"dim_{dim_idx}_vs_subjective.png")
            )
            print(f"维度 {dim_idx} 可视化完成，已保存6个部件的权重对比图")
    else:
        print("No part_weights available for visualization")
        
if __name__ == "__main__":
    train()