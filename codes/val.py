import os
import sys
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

# 设置环境变量，避免tokenizers并行问题
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

# -------------------------- 1. 自定义 ReID 指标计算（无第三方依赖） --------------------------
def compute_distance_matrix(feat1: torch.Tensor, feat2: torch.Tensor, metric: str = "cosine") -> torch.Tensor:
    """
    计算特征距离矩阵（支持余弦距离和欧氏距离）
    Args:
        feat1: 特征矩阵1，shape=(N, D)，N为样本数，D为特征维度
        feat2: 特征矩阵2，shape=(M, D)
        metric: 距离度量方式，可选 "cosine"（余弦距离）或 "euclidean"（欧氏距离）
    Returns:
        dist_matrix: 距离矩阵，shape=(N, M)，dist_matrix[i][j] 表示 feat1[i] 与 feat2[j] 的距离
    """
    if metric not in ("cosine", "euclidean"):
        raise ValueError(f"不支持的距离度量方式: {metric}，可选 'cosine' 或 'euclidean'")
    
    # 特征归一化（余弦距离计算需先归一化）
    if metric == "cosine":
        feat1 = F.normalize(feat1, p=2, dim=1)
        feat2 = F.normalize(feat2, p=2, dim=1)
        sim_matrix = torch.matmul(feat1, feat2.T)  # 余弦相似度矩阵
        return 1 - sim_matrix  # 余弦距离（范围0~2，值越小越相似）
    else:  # euclidean
        return torch.cdist(feat1, feat2, p=2)  # 欧氏距离


def parse_gt_bbox(gt_bbox_dir: str) -> dict:
    """解析gt_bbox文件夹，返回样本类型字典：{图像索引: 类型}"""
    if not os.path.isdir(gt_bbox_dir):
        raise ValueError(f"gt_bbox目录不存在: {gt_bbox_dir}")
        
    gt_bbox = {}
    for filename in os.listdir(gt_bbox_dir):
        if filename.endswith('.jpg'):  # 这里可能是笔误，标签文件通常不是jpg格式
            file_path = os.path.join(gt_bbox_dir, filename)
            try:
                with open(file_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        # 假设格式：图像索引 类型（如 "0001_c1s1_001051_00.jpg"）
                        img_idx, label_type = line.split()
                        gt_bbox[int(img_idx)] = label_type
            except Exception as e:
                print(f"解析文件 {file_path} 出错: {e}")
    return gt_bbox


def parse_gt_query(gt_query_dir: str) -> dict:
    """解析gt_query文件夹，返回查询-有效样本映射：{查询索引: [有效样本索引列表]}(good/junk)"""
    if not os.path.isdir(gt_query_dir):
        raise ValueError(f"gt_query目录不存在: {gt_query_dir}")
        
    gt_query = {}
    for filename in os.listdir(gt_query_dir):
        if filename.endswith('.mat'):
            # 假设文件名对应查询索引（如 0001_c1s1_001051_00_good.mat 对应查询索引0001）
            try:
                query_idx = int(filename.split('_')[0])
                file_path = os.path.join(gt_query_dir, filename)
                with open(file_path, 'r') as f:
                    valid_indices = []
                    for line in f:
                        line = line.strip()
                        if line:
                            valid_indices.append(int(line))
                    gt_query[query_idx] = valid_indices
            except Exception as e:
                print(f"解析文件 {filename} 出错: {e}")
    return gt_query


def evaluate_rank(
    dist_matrix: torch.Tensor, 
    pids: torch.Tensor, 
    camids: torch.Tensor, 
    max_rank: int = 50, 
    use_metric_cuhk03: bool = False, 
    img_indices: list = None, 
    gt_query: dict = None
) -> tuple[float, float]:
    """
    计算 ReID 核心指标（Rank-1、mAP）
    Args:
        dist_matrix: 距离矩阵，shape=(N, N)，N为验证集样本总数
        pids: 身份ID列表，shape=(N,)
        camids: 摄像头ID列表，shape=(N,)
        max_rank: 评估的最大Rank值（默认50）
        use_metric_cuhk03: 是否使用CUHK03数据集的特殊评估方式
        img_indices: 图像索引列表
        gt_query: 查询-有效样本映射字典
    Returns:
        rank1: Rank-1准确率
        mAP: 平均精度均值
    """
    num_query = dist_matrix.shape[0]
    all_cmc = torch.zeros(num_query, max_rank, dtype=torch.float32, device=dist_matrix.device)
    all_ap = torch.zeros(num_query, dtype=torch.float32, device=dist_matrix.device)

    for q_idx in range(num_query):
        q_pid = pids[q_idx]
        q_camid = camids[q_idx]
        q_img_idx = img_indices[q_idx] if img_indices and q_idx < len(img_indices) else None
        
        # 1. 基础过滤：排除同摄像头同身份（自匹配）
        dists = dist_matrix[q_idx]
        mask = (pids != q_pid) | (camids != q_camid)
        gallery_dists = dists[mask]
        gallery_pids = pids[mask]
        
        # 2. 使用gt_query过滤，仅保留标注的有效样本
        if gt_query and q_img_idx in gt_query and img_indices:
            valid_gallery_indices = gt_query[q_img_idx]
            gallery_img_indices = [img_indices[i] for i in torch.where(mask)[0].cpu().numpy()]
            valid_mask = torch.tensor([idx in valid_gallery_indices for idx in gallery_img_indices], 
                                     device=gallery_dists.device)
            gallery_dists = gallery_dists[valid_mask]
            gallery_pids = gallery_pids[valid_mask]
        
        # 处理空gallery情况
        if len(gallery_pids) == 0:
            all_cmc[q_idx] = 0
            all_ap[q_idx] = 0
            continue
        
        # 排序并计算匹配结果
        sorted_indices = torch.argsort(gallery_dists)
        sorted_pids = gallery_pids[sorted_indices]
        matches = (sorted_pids == q_pid).float()
        
        # 计算CMC
        cmc = matches.cumsum(dim=0)
        cmc = cmc[:max_rank]  # 截断到max_rank
        if len(cmc) < max_rank:
            cmc = torch.cat([cmc, torch.zeros(max_rank - len(cmc), dtype=torch.float32, device=cmc.device)])
        cmc = torch.clamp(cmc, 0, 1)  # 确保不超过1
        all_cmc[q_idx] = cmc
        
        # 计算AP
        num_matches = matches.sum().int()
        if num_matches == 0:
            all_ap[q_idx] = 0
            continue
            
        precision = matches.cumsum(dim=0) / torch.arange(1, len(matches) + 1, dtype=torch.float32, device=matches.device)
        ap = (precision * matches).sum() / num_matches
        all_ap[q_idx] = ap
    
    return all_cmc[:, 0].mean().item(), all_ap.mean().item()


# -------------------------- 2. 评估函数（复用原逻辑，调用自定义指标） --------------------------
@torch.no_grad()  # 显式声明无梯度计算
def evaluate_reid(model: torch.nn.Module, val_loader, device: torch.device, 
                 gt_bbox: dict = None, gt_query: dict = None) -> tuple[float, float]:
    """评估ReID模型，加入gt_bbox和gt_query过滤"""
    model.eval()
    all_features = []
    all_pids = []
    all_camids = []
    all_img_indices = []
    
    # 准备默认文本输入
    default_input_ids = torch.zeros((1, 1), dtype=torch.long, device=device)
    default_attention_mask = torch.ones((1, 1), dtype=torch.long, device=device)
    
    for batch in tqdm(val_loader, desc="提取特征"):
        try:
            # 解析批次数据
            if isinstance(batch, dict):
                images = batch["image"]
                pids = batch["label"]
                camids = batch.get("camid", torch.zeros_like(pids, device=pids.device))
                img_indices = batch["img_index"]
            else:
                images, pids, camids, img_indices = batch[:4]  # 假设前四个元素是需要的
            
            # 移动到设备
            images = images.to(device, non_blocking=True)
            pids = pids.to(device, non_blocking=True)
            camids = camids.to(device, non_blocking=True)
            
            # 前向传播获取特征
            batch_size = images.shape[0]
            text_inputs = {
                "input_ids": default_input_ids.repeat(batch_size, 1),
                "attention_mask": default_attention_mask.repeat(batch_size, 1)
            }
            outputs = model(image_inputs={"pixel_values": images}, text_inputs=text_inputs)
            features = outputs[0]
            
            # 收集结果
            all_features.append(features.cpu())
            all_pids.append(pids.cpu())
            all_camids.append(camids.cpu())
            all_img_indices.extend(img_indices.cpu().numpy().tolist())
            
        except Exception as e:
            print(f"处理批次时出错: {e}")
            continue
    
    # 处理空数据情况
    if not all_features:
        print("警告: 未提取到任何特征")
        return 0.0, 0.0
    
    # 拼接所有样本
    all_features = torch.cat(all_features, dim=0)
    all_pids = torch.cat(all_pids, dim=0)
    all_camids = torch.cat(all_camids, dim=0)
    
    # 过滤非"good"样本（仅保留有效样本）
    if gt_bbox:
        good_mask = [gt_bbox.get(idx, "") == "good" for idx in all_img_indices]
        good_indices = torch.where(torch.tensor(good_mask))[0]
        
        if len(good_indices) == 0:
            print("警告: 未找到任何'good'样本")
            return 0.0, 0.0
            
        # 筛选有效样本
        good_features = all_features[good_indices]
        good_pids = all_pids[good_indices]
        good_camids = all_camids[good_indices]
        good_img_indices = [all_img_indices[i] for i in good_indices.numpy()]
    else:
        # 没有gt_bbox时使用全部样本
        good_features = all_features
        good_pids = all_pids
        good_camids = all_camids
        good_img_indices = all_img_indices
    
    # 计算距离矩阵
    dist_matrix = compute_distance_matrix(good_features, good_features, metric="cosine")
    
    # 评估
    rank1, mAP = evaluate_rank(
        dist_matrix, 
        good_pids, 
        good_camids, 
        max_rank=50, 
        use_metric_cuhk03=False,
        img_indices=good_img_indices,
        gt_query=gt_query
    )
    
    model.train()
    return rank1, mAP


def val():
    cfg = Config()
    cfg.init_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_cfg = cfg.DATASETS.get(cfg.USE_DATASET)
    
    if not dataset_cfg:
        raise ValueError(f"未找到数据集配置: {cfg.USE_DATASET}")
    
    print(f"使用设备: {device}")
    
    # 加载验证集数据
    print("加载验证集数据...")
    val_data_loader = ReIDDataLoader(
            img_dir=dataset_cfg["test_img"],
            caption_path=dataset_cfg.get("test_caption"),
            batch_size=cfg.batch_size,
            num_workers=4,
            is_train=False,
            size=[384, 128],  # 论文指定尺寸
            use_identity_sampler=False,
            num_ids_per_batch=16,
            num_instances_per_id=1
        )
    val_loader = val_data_loader.get_loader()
    
    with tqdm(total=len(val_data_loader.dataset), desc="验证集加载进度") as pbar:
        for _ in val_data_loader.dataset:
            pbar.update(1)
    num_val_ids = val_data_loader.num_ids
        
    # 加载gt_bbox和gt_query标注
    print("加载评估标注文件...")
    gt_bbox = None
    gt_query = None
    
    try:
        if "gt_bbox_dir" in dataset_cfg:
            gt_bbox_dir = dataset_cfg["gt_bbox_dir"]
            print(f"解析gt_bbox: {gt_bbox_dir}")
            gt_bbox = parse_gt_bbox(gt_bbox_dir)
        
        if "gt_query_dir" in dataset_cfg:
            gt_query_dir = dataset_cfg["gt_query_dir"]
            print(f"解析gt_query: {gt_query_dir}")
            gt_query = parse_gt_query(gt_query_dir)
    except Exception as e:
        print(f"加载标注文件出错: {e}")
        return

    # 校验标签范围：若标签从1开始，转为0开始；若存在超界标签，抛出错误
    val_labels = val_data_loader.dataset.labels
    if min(val_labels) == 1 and max(val_labels) == num_val_ids:
        # 标签从1开始 → 转为0开始（模型分类头要求）
        val_data_loader.dataset.labels = [pid - 1 for pid in val_labels]
        print("标签已从1-based转为0-based")
    elif max(val_labels) >= num_val_ids or min(val_labels) < 0:
        # 存在超界标签 → 抛出错误并提示
        raise ValueError(
            f"标签范围异常！类别数={num_val_ids}，但标签范围=[{min(val_labels)}, {max(val_labels)}]，"
            "请检查 data_loader.py 的 _parse_img_labels 方法是否正确提取pid"
        )
    # 初始化模型（原代码缺失，补充）
    model = BlipIAPMReID(
        blip_pretrained_path=cfg.blip_pretrained_path,
        num_parts=cfg.num_parts,
        part_feat_dim=cfg.IAPM_CONFIG["in_channels"],
        num_ids=num_val_ids
    ).to(device)
    
    # 初始化日志（原代码缺失，补充）
    logger = TensorboardLogger(log_dir=cfg.LOG_DIR)
    
    # 验证与模型保存
    try:
        current_rank1, current_map = evaluate_reid(model, val_loader, device, gt_bbox, gt_query)
        logger.log_scalar("val/rank1", current_rank1, 0)  # epoch假设为0，原代码未定义
        logger.log_scalar("val/mAP", current_map, 0)
        
        print(f"Val Rank-1: {current_rank1:.4f}, Val mAP: {current_map:.4f}")
        
        # 可视化（假设part_weights已定义）
        # plot_part_weights(
        #     img_path="./data/raw/Market-1501/query/0001_c1s1_001051_00.jpg",
        #     part_weights=part_weights,
        #     save_path=os.path.join(logger.log_dir, "part_weights.png")
        # )

        # plot_weight_vs_subjective(
        #     query_idx=0,
        #     query_img_path="./data/raw/Market-1501/query/0001_c1s1_001051_00.jpg",
        #     iwm_weights=part_weights,
        #     subjective_scores=[0.3, 0.5, 0.8],
        #     save_path=os.path.join(logger.log_dir, "weight_vs_subjective.png")
        # )
        
    except Exception as e:
        print(f"评估过程出错: {e}")
    finally:
        torch.cuda.empty_cache()  # 清理显存


if __name__ == "__main__":
    val()