import os
from typing import Dict, Tuple


class Config:
    """工程配置类（保留自定义实例配置）"""
    
    # ========================= 【类属性：默认配置】=========================
    # 基础路径配置
    ROOT_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 工程根目录
    DATA_DIR: str = os.path.join(ROOT_DIR, "data")
    MODEL_DIR: str = os.path.join(ROOT_DIR, "model")
    OUTPUT_DIR: str = os.path.join(ROOT_DIR, "output")
    PRETRAINED_DIR: str = os.path.join(ROOT_DIR, "pretrained")
    BLIP_MODEL_DIR: str = os.path.join(ROOT_DIR, "blip_model")  

    # 数据集默认配置
    from typing import Any
    DATASETS: Dict[str, Dict[str, Any]] = {
        "Market-1501": {
            "train_img": os.path.join(DATA_DIR, "raw/Market-1501/bounding_box_train"),
            "test_img": os.path.join(DATA_DIR, "raw/Market-1501/bounding_box_test"),
            "query_img": os.path.join(DATA_DIR, "raw/Market-1501/query"),
            "caption": os.path.join(DATA_DIR, "processed/Market-1501/captions.json"),
            "pseudo_label": os.path.join(DATA_DIR, "processed/Market-1501/pseudo_labels"),
            "gt_bbox": os.path.join(DATA_DIR, "raw/Market-1501/gt_bbox"),
            "gt_query": os.path.join(DATA_DIR, "raw/Market-1501/gt_query"),
            "num_train_ids": 751,  # 原论文Market-1501训练集身份数
            "image_size": (384, 128) 
        },
        "DukeMTMC-reID": {
            "train_img": os.path.join(DATA_DIR, "processed/DukeMTMC-reID/bounding_box_train"),
            "test_img": os.path.join(DATA_DIR, "processed/DukeMTMC-reID/bounding_box_test"),
            "query_img": os.path.join(DATA_DIR, "processed/DukeMTMC-reID/query"),
            "caption": os.path.join(DATA_DIR, "processed/DukeMTMC-reID/captions.json"),
            "pseudo_label": os.path.join(DATA_DIR, "processed/DukeMTMC-reID/pseudo_labels"),
            "num_train_ids": 702  # 原论文Duke训练集身份数
        },
        "CUHK03": {
            "train_img": os.path.join(DATA_DIR, "processed/CUHK03/labeled/bounding_box_train"),
            "test_img": os.path.join(DATA_DIR, "processed/CUHK03/labeled/bounding_box_test"),
            "query_img": os.path.join(DATA_DIR, "processed/CUHK03/labeled/query"),
            "caption": os.path.join(DATA_DIR, "processed/CUHK03/captions.json"),
            "pseudo_label": os.path.join(DATA_DIR, "processed/CUHK03/pseudo_labels"),
            "num_train_ids": 767  # 原论文CUHK03训练集身份数
        }
    }
    USE_DATASET: str = "Market-1501"  # 默认使用的数据集（三选一）
    IMG_SIZE: Tuple[int, int] = (384, 128)  # 输入尺寸（H, W）
    NUM_PARTS_DEFAULT: int = 7  # 原论文最优部件数（7个，作为实例属性的默认 fallback）

    # 模型默认配置
    BLIP_PRETRAINED_DEFAULT: str = os.path.abspath(r"./blip_model")
    IAPM_CONFIG: Dict[str, int] = {
        "in_channels": 768,  # BLIP-ViT输出通道数
        "num_classes_ps": 8,  # PS模块分割类别数（原论文8个类）
        "hidden_dim_iwm": 256  # IWM模块隐藏层维度（原论文未明确，取256适配）
    }

    # 训练默认配置（参数）
    BATCH_SIZE_DEFAULT: int = 128  
    MAX_EPOCHS: int = 1  # 默认训练轮数
    INIT_LR: Dict[str, float] = {
        "backbone": 0.0001,  # 骨干网络初始学习率
        "iwm_em": 0.0002     # IWM和EM层初始学习率
    }
    from typing import Any
    LR_SCHEDULER: Dict[str, Any] = {
        "milestones": [50, 80],  # 学习率衰减节点
        "gamma": 0.1             # 衰减系数（1/10）
    }
    OPTIMIZER: Dict[str, Any] = {
        "type": "SGD",
        "momentum": 0.9,         # 动量
        "weight_decay": 5e-4     # 权重衰减
    }
    LOSS_COEFF: Dict[str, float] = {
        "lambda_sptl": 1.0,  # SPTL损失系数（原论文1.0）
        "beta_center": 0.0005  # 中心损失系数（原论文0.0005）
    }
    SPTL_ALPHA: float = 1.2  # PTL损失α值（原论文1.2）

    # 输出默认配置
    LOG_DIR: str = os.path.join(OUTPUT_DIR, "logs")
    CHECKPOINT_DIR: str = os.path.join(OUTPUT_DIR, "checkpoints")
    RESULT_DIR_DEFAULT: str = os.path.join(OUTPUT_DIR, "results")  # 与实例属性result_dir对应
    VISUALIZE_DIR: str = os.path.join(RESULT_DIR_DEFAULT, "visualizations")
    BEST_CHECKPOINT_DEFAULT: str = os.path.join(CHECKPOINT_DIR, "epoch_100.pth")  # 与实例属性best_checkpoint_path对应

    # 测试默认配置
    METRIC: str = "euclidean"  # 原论文距离度量（欧氏距离）


    # ========================= 【自定义实例配置】=========================
    def __init__(self):
        # 用户自定义实例属性，优先级高于类属性
        self.batch_size = 8  # 自定义批次大小（覆盖类属性BATCH_SIZE_DEFAULT）
        self.num_workers = 24  # 数据加载线程数（自定义）
        self.num_parts = 7    # 自定义部件数（覆盖类属性NUM_PARTS_DEFAULT）
        self.num_train_ids = 1000  # 自定义训练身份数（若需覆盖数据集默认值）
        
        # 路径类实例属性：优先使用自定义路径，若为占位符则自动替换为类属性的默认路径
        self.blip_pretrained_path = "path/to/blip_pretrained.pth"  # 自定义BLIP权重路径
        self.best_checkpoint_path = "path/to/best_checkpoint.pth"  # 自定义最优模型路径
        self.result_dir = "results/"  # 自定义结果保存目录
        
        # 数据集相关自定义路径（优先使用，无则 fallback 到当前数据集的默认路径）
        self.test_img_dir = "path/to/test/images"  # 自定义测试图像目录
        self.test_caption_path = "path/to/test/captions.txt"  # 自定义测试文本描述路径
        self.query_img_dir = "path/to/query/images"  # 自定义查询图像目录
        self.gt_bbox_dir = "path/to/gt_bbox/images"  # 自定义GT标注图像目录
        self.gt_query_dir = "path/to/gt_query/images"  # 自定义查询图像的GT目录
        self.query_caption_path = "path/to/query/captions.txt"  # 自定义查询文本描述路径
        # Add more parameters as needed（用户可在此处继续添加自定义实例属性）

        # 初始化：自动修复占位符路径，避免硬编码无效路径
        self._fix_placeholder_paths()
        # 验证：确保关键配置有效（如数据集选择、路径存在性）
        self._validate_config()


    # ========================= 【辅助方法：保证配置有效性】=========================
    def _fix_placeholder_paths(self):
        """若实例属性为默认占位符，自动替换为类属性的默认有效路径"""
        # 1. BLIP预训练权重路径：占位符→类属性默认路径
        if self.blip_pretrained_path.startswith("path/to/"):
            self.blip_pretrained_path = self.BLIP_PRETRAINED_DEFAULT
            print(f"BLIP权重路径使用默认值：{self.blip_pretrained_path}")
        
        # 2. 最优模型路径：占位符→类属性默认路径
        if self.best_checkpoint_path.startswith("path/to/"):
            self.best_checkpoint_path = self.BEST_CHECKPOINT_DEFAULT
            print(f"最优模型路径使用默认值：{self.best_checkpoint_path}")
        
        # 3. 结果目录：相对路径→绝对路径（基于工程根目录）
        if not os.path.isabs(self.result_dir):
            self.result_dir = os.path.join(self.ROOT_DIR, self.result_dir)
            print(f"结果目录转换为绝对路径：{self.result_dir}")
        
        # 4. 数据集相关路径：占位符→当前使用数据集的默认路径
        current_dataset = self.DATASETS[self.USE_DATASET]
        # 测试图像目录
        if self.test_img_dir.startswith("path/to/"):
            self.test_img_dir = current_dataset["test_img"]
            print(f"测试图像目录使用默认值（{self.USE_DATASET}）：{self.test_img_dir}")
        # 测试文本描述路径
        if self.test_caption_path.startswith("path/to/"):
            self.test_caption_path = current_dataset["caption"]
            print(f"测试文本描述路径使用默认值（{self.USE_DATASET}）：{self.test_caption_path}")
        # 查询图像目录
        if self.query_img_dir.startswith("path/to/"):
            self.query_img_dir = current_dataset["query_img"]
            print(f"查询图像目录使用默认值（{self.USE_DATASET}）：{self.query_img_dir}")
        # GT标注图像目录
        if self.gt_bbox_dir.startswith("path/to/"):
            self.gt_bbox_dir = current_dataset["gt_bbox"]
            print(f"GT标注图像目录使用默认值（{self.USE_DATASET}）：{self.gt_bbox_dir}")
        # 查询图像的GT目录
        if self.gt_query_dir.startswith("path/to/"):
            self.gt_query_dir = current_dataset["gt_query"]
            print(f"查询图像的GT目录使用默认值（{self.USE_DATASET}）：{self.gt_query_dir}")
        # 查询文本描述路径
        if self.query_caption_path.startswith("path/to/"):
            self.query_caption_path = current_dataset["caption"]  # 若数据集caption通用则复用，否则需单独配置
            print(f"查询文本描述路径使用默认值（{self.USE_DATASET}）：{self.query_caption_path}")


    def _validate_config(self):
        """验证配置有效性"""
        # 1. 验证数据集选择是否有效
        if self.USE_DATASET not in self.DATASETS:
            raise ValueError(
                f"无效的数据集选择：{self.USE_DATASET}\n"
                f"可选数据集：{list(self.DATASETS.keys())}"
            )
        
        # 2. 验证关键路径是否存在（提前报错，避免后续训练/测试崩溃）
        critical_paths = [
            self.blip_pretrained_path,  # BLIP预训练权重
            self.test_img_dir,          # 测试图像目录
            self.query_img_dir,         # 查询图像目录
            self.gt_bbox_dir,           # GT标注图像目录
            self.gt_query_dir,          # 查询图像的GT目录
            self.test_caption_path      # 测试文本描述
        ]
        for path in critical_paths:
            if not os.path.exists(path):
                # 若为目录：自动创建；若为文件：报错提示
                if os.path.isdir(path) or (path.endswith("/") and not os.path.isfile(path)):
                    os.makedirs(path, exist_ok=True)
                    print(f"自动创建不存在的目录：{path}")
                else:
                    raise FileNotFoundError(
                        f"关键文件不存在：{path}\n"
                        f"请检查路径配置，或替换为有效路径"
                    )


    def get_current_dataset_info(self) -> Dict[str, str]:
        """获取当前使用数据集的完整信息（含默认路径、训练身份数）"""
        dataset_info = self.DATASETS[self.USE_DATASET].copy()
        # 覆盖为用户自定义的路径（若已设置）
        dataset_info["test_img"] = self.test_img_dir
        dataset_info["query_img"] = self.query_img_dir
        dataset_info["gt_bbox"] = self.gt_bbox_dir
        dataset_info["gt_query"] = self.gt_query_dir
        dataset_info["test_caption"] = self.test_caption_path
        dataset_info["query_caption"] = self.query_caption_path
        # 覆盖为用户自定义的训练身份数（若已设置）
        if self.num_train_ids != 1000:  # 1000为默认占位符，非占位符则认为是用户自定义值
            dataset_info["num_train_ids"] = self.num_train_ids
        return dataset_info


    @classmethod
    def init_dirs(cls) -> None:
        """初始化所有必要目录（日志、 checkpoint、可视化等）"""
        dirs = [cls.LOG_DIR, cls.CHECKPOINT_DIR, cls.VISUALIZE_DIR]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
        print(f"已初始化所有默认目录：{dirs}")