import os
import sys
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader


# 加入工程根目录到搜索路径
current_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(os.path.dirname(os.path.dirname(current_path)))
if root_path not in sys.path:
    sys.path.insert(0, root_path)
    
from codes.utils.samplers import RandomIdentitySampler
from model.blip.blip import BlipProcessor

class ReIDDataset(Dataset):
    """行人重识别数据集（适配原论文3个主流数据集+多模态文本标注）"""
    def __init__(self, img_dir, caption_path, pseudo_label_path=None, processor=None, is_train=True, size=None):
        """
        Args:
            img_dir: 图像文件夹路径（如bounding_box_train）
            caption_path: 文本标注JSON路径（image_path: caption）
            pseudo_label_path: PS模块伪标签文件夹路径（训练时需传入）
            processor: BlipProcessor（处理图像+文本）
            is_train: 是否为训练集（训练时返回伪标签，测试时不返回）
            size: 图像尺寸，格式为[height, width]
        """
        # 验证输入路径有效性
        if not os.path.isdir(img_dir):
            raise ValueError(f"图像目录不存在: {img_dir}")
        if caption_path and not os.path.isfile(caption_path):
            raise ValueError(f"文本标注文件不存在: {caption_path}")
        if pseudo_label_path and not os.path.isdir(pseudo_label_path):
            raise ValueError(f"伪标签目录不存在: {pseudo_label_path}")

        self.img_dir = img_dir
        self.caption_path = caption_path
        self.pseudo_label_path = pseudo_label_path
        self.processor = processor
        self.is_train = is_train
        self.size = size if size is not None else [384, 128]  # 默认尺寸
        
        # 1. 加载图像路径与身份标签
        self.img_paths = []
        self.labels = []
        self.original_labels = []  
        self.camids = []  # 摄像头ID
        self._parse_img_labels()
        
        # 处理标签映射（将原始PID映射为连续整数）
        self.label_mapping = self._create_label_mapping()
        self.labels = [self.label_mapping[pid] for pid in self.original_labels]
        
        # 2. 文本处理
        self.captions = self._load_captions()
        
        # 3. 加载伪标签（仅训练时）
        self.pseudo_labels = self._load_pseudo_labels() if self.is_train and self.pseudo_label_path else None

    def _parse_img_labels(self):
        """解析图像文件名获取身份标签和摄像头ID（适配 pid_cXsY_seq_frame.jpg 格式）"""
        valid_extensions = ('.jpg', '.png', '.jpeg')
        for img_name in os.listdir(self.img_dir):
            if not img_name.lower().endswith(valid_extensions):
                continue
                
            parts = img_name.split('_')
            if len(parts) < 4:
                print(f"警告：跳过格式异常的文件 {img_name}（不符合 'pid_cXsY_seq_frame.jpg' 格式）")
                continue
                
            try:
                # 1. 提取 pid（parts[0] 是纯数字，直接转整数）
                pid = int(parts[0])
                
                # 2. 提取 camid：从 parts[1]（如 'c5s3'）中提取 'c' 后的数字
                cam_part = parts[1]  # 示例：'c5s3'
                # 找到 'c' 的索引，取其后的数字（直到非数字字符停止）
                c_index = cam_part.find('c')
                if c_index == -1:
                    raise ValueError(f"未找到摄像头标识 'c'（{cam_part}）")
                # 从 'c' 后开始提取数字（如 'c5s3' → 从索引1开始取 '5'）
                camid_str = ''
                for char in cam_part[c_index+1:]:
                    if char.isdigit():
                        camid_str += char
                    else:
                        break  # 遇到非数字字符（如 's'）停止
                if not camid_str:
                    raise ValueError(f"无法从 {cam_part} 中提取摄像头数字")
                camid = int(camid_str)  # 转为整数（如 '5' → 5）
                
            except ValueError as e:
                print(f"警告：跳过无法提取pid/camid的文件 {img_name}，原因：{str(e)}")
                continue
                
            if pid == -1:  # 排除无效标签
                continue
                
            img_path = os.path.join(self.img_dir, img_name)
            self.img_paths.append(img_path)
            self.original_labels.append(pid)
            self.camids.append(camid)  # 保存提取的摄像头ID
        
    def _create_label_mapping(self):
        """将原始PID映射为连续整数"""
        unique_pids = sorted(list(set(self.original_labels)))
        return {pid: idx for idx, pid in enumerate(unique_pids)}
    
    def _load_captions(self):
        """加载文本标注"""
        if not self.caption_path or not os.path.exists(self.caption_path):
            return {}
            
        try:
            with open(self.caption_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"警告：无法解析JSON文件 {self.caption_path}，使用空标注")
            return {}
    
    def _load_pseudo_labels(self):
        """加载PS模块的伪标签"""
        pseudo_labels = {}
        if not self.pseudo_label_path:
            return pseudo_labels
            
        for img_name in os.listdir(self.pseudo_label_path):
            if not img_name.lower().endswith(('.png', '.npy')):
                continue
                
            img_base = os.path.splitext(img_name)[0]
            label_path = os.path.join(self.pseudo_label_path, img_name)
            
            try:
                # 读取伪标签并调整尺寸
                label = Image.open(label_path).convert('L')
                label = label.resize((16, 48), Image.Resampling.NEAREST)  # (W=16, H=48)
                pseudo_labels[img_base] = torch.tensor(np.array(label), dtype=torch.long)
            except Exception as e:
                print(f"警告：无法加载伪标签 {label_path}，错误: {str(e)}")
                
        if pseudo_labels:  # 检查字典非空
            total_non_zero = sum(torch.sum(label != 0) for label in pseudo_labels.values())
            print(f"伪标签非零值数量: {total_non_zero}")
        return pseudo_labels

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        # 步骤1：处理张量类型索引（如采样器返回torch.Tensor）
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()  # 转为列表
        
        # 步骤2：处理嵌套列表（支持任意深度的嵌套，如[[[0,1]], [2,3]] → [0,1,2,3]）
        def flatten_list(lst):
            flat = []
            for item in lst:
                if isinstance(item, list):
                    flat.extend(flatten_list(item))  # 递归扁平化
                else:
                    flat.append(item)
            return flat
        
        # 步骤3：统一将索引转为“单层整数列表”
        if isinstance(idx, list):
            flat_idx = flatten_list(idx)  # 彻底扁平化嵌套列表
        else:  # 单个整数索引
            flat_idx = [idx]
        
        # 步骤4：生成字典列表（确保返回的是单层列表，无嵌套）
        result = [self._get_single_item(i) for i in flat_idx]
        
        # 步骤5：若原始索引是单个整数，返回单个字典（保持单样本调用的兼容性）
        if not isinstance(idx, list) and not isinstance(idx, torch.Tensor):
            return result[0]
        
        # 批量索引返回单层字典列表
        return result
    
    def _get_single_item(self, idx):
        """处理单个整数索引，返回单样本字典"""
        if not isinstance(idx, int):
            raise TypeError(f"单样本索引必须是整数，当前为 {type(idx)}（请检查采样器输出格式）")
        if idx < 0 or idx >= len(self):
            raise IndexError(f"索引超出数据集范围！有效范围: 0 ~ {len(self)-1}，当前: {idx}")
            
        img_path = self.img_paths[idx]
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        label = self.labels[idx]
        
        # 1. 处理图像
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"无法打开图像 {img_path}: {str(e)}")
            
        if self.processor is None:
            raise ValueError("BlipProcessor (self.processor) 未初始化，请提供有效的processor实例")

        # 调整图像尺寸
        image = image.resize((self.size[1], self.size[0]), Image.Resampling.BILINEAR)
        
        # 使用processor处理
        image_inputs = self.processor(
            images=image,
            do_resize=True,
            size={"height": self.size[0], "width": self.size[1]},
            return_tensors="pt"
        )['pixel_values'].squeeze(0)

        # 验证尺寸
        if image_inputs.shape[1:] != (self.size[0], self.size[1]):
            raise ValueError(f"图像尺寸不符合要求，期望{self.size}，实际{image_inputs.shape[1:]}")

        # 2. 处理文本
        caption = self.captions.get(img_path, "A person in the image")
        text_inputs = self.processor(
            text=caption,
            return_tensors="pt",
            padding="max_length",
            max_length=50,
            truncation=True  # 新增截断过长文本
        )
        text_inputs = {k: v.squeeze(0) for k, v in text_inputs.items()}

        # 3. 处理伪标签
        pseudo_label = None
        if self.is_train and self.pseudo_labels is not None:
            pseudo_label = self.pseudo_labels.get(
                img_name, 
                torch.zeros((48, 16), dtype=torch.long)
        )
                
            return {
                    "image": image_inputs,
                    "text_input_ids": text_inputs["input_ids"],
                    "text_attention_mask": text_inputs["attention_mask"],
                    "label": torch.tensor(label, dtype=torch.long),
                    "pseudo_label": pseudo_label,
                    "img_path": img_path,
                    "camid": torch.tensor(self.camids[idx], dtype=torch.long) 
                }
        
        else:
            return {
                "image": image_inputs,
                "text_input_ids": text_inputs["input_ids"],
                "text_attention_mask": text_inputs["attention_mask"],
                "label": torch.tensor(label, dtype=torch.long),
                "img_path": img_path,
                "camid": torch.tensor(self.camids[idx], dtype=torch.long) 
            }

class ReIDDataLoader:
    """数据加载器封装（适配训练/测试流程）"""
    def __init__(self, img_dir, caption_path, pseudo_label_path=None, 
                 batch_size=None, num_workers=4, is_train=True, 
                 size=[384, 128], use_identity_sampler=True,
                 num_ids_per_batch=16, num_instances_per_id=4):
        """
        Args:
            img_dir: 图像文件夹路径
            caption_path: 文本标注JSON路径
            pseudo_label_path: PS模块伪标签文件夹路径
            batch_size: 批次大小
            num_workers: 数据加载线程数
            is_train: 是否为训练集
            size: 图像尺寸，格式为[height, width]
            use_identity_sampler: 是否使用身份采样器
            num_ids_per_batch: 每个批次中的身份数量（仅当使用身份采样器时有效）
            num_instances_per_id: 每个身份的样本数量（仅当使用身份采样器时有效）
        """
        # 初始化BLIP处理器
        self.processor = self._init_processor()
        
        # 初始化数据集
        self.dataset = ReIDDataset(
            img_dir=img_dir,
            caption_path=caption_path,
            pseudo_label_path=pseudo_label_path,
            processor=self.processor,
            is_train=is_train,
            size=size
        )
        
        # 配置参数
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.is_train = is_train
        self.use_identity_sampler = use_identity_sampler
        self.num_ids_per_batch = num_ids_per_batch
        self.num_instances_per_id = num_instances_per_id
        
        # 如果使用身份采样器，调整批次大小
        if self.use_identity_sampler:
            self.batch_size = num_ids_per_batch * num_instances_per_id

    def _init_processor(self):
        """初始化BLIP处理器"""
        try:
            blip_processor_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                r"A:\ReID_DC\BLIP-IAPM-ReID\blip_model"  # 简化路径，避免绝对路径依赖
                # "/Volumes/SC/ReID_DC/BLIP-IAPM-ReID/blip_model"
            )
            blip_processor_dir = os.path.abspath(blip_processor_dir)
            
            if not os.path.isdir(blip_processor_dir):
                raise FileNotFoundError(f"BLIP处理器目录不存在: {blip_processor_dir}")
                
            return BlipProcessor.from_pretrained(
                blip_processor_dir,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
        except Exception as e:
            raise RuntimeError(f"初始化BLIP处理器失败: {str(e)}")

    def get_loader(self):
        """返回DataLoader（训练时可使用身份采样器）"""
        sampler = None
        if self.is_train and self.use_identity_sampler:
            sampler = RandomIdentitySampler(
                dataset=self.dataset,
                num_ids=self.num_ids_per_batch,
                num_instances=self.num_instances_per_id
            )
        
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=self.is_train and (sampler is None),  # 采样器存在时不打乱
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),  # 自动根据是否有GPU决定
            drop_last=self.is_train,
            collate_fn=self._collate_fn  # 自定义collate函数处理可能的不规则数据
        )

    def _collate_fn(self, batch):
        """自定义collate函数处理批次数据"""
        def flatten_batch(batch):
            flat_batch = []
            for item in batch:
                if isinstance(item, list):  # 若元素是列表，递归展开
                    flat_batch.extend(flatten_batch(item))
                else:  # 元素是字典，直接加入
                    flat_batch.append(item)
            return flat_batch
    
        # 执行扁平化（关键步骤：消除任何残留的嵌套结构）
        batch = flatten_batch(batch)
    
        # 验证：确保扁平化后所有元素都是字典
        if not batch or not isinstance(batch[0], dict):
            raise TypeError(f"batch扁平化后格式异常！期望元素是dict，实际第一个元素类型是 {type(batch[0])}")
        
        # 处理可能的None值（如伪标签）
        keys = batch[0].keys()
        collated = {k: [] for k in keys}
        
        for item in batch:
            for k, v in item.items():
                collated[k].append(v)
        
        tensor_collated = {}
        for k in collated:
        # 1. 路径字段保持字符串列表（无需stack）
            if k == "img_path":
                tensor_collated[k] = collated[k]
            # 2. 仅对Tensor类型字段执行stack（过滤None或非Tensor类型）
            elif isinstance(collated[k][0], torch.Tensor):
                # 额外检查：确保列表中没有None（避免极端情况）
                if any(v is None for v in collated[k]):
                    raise ValueError(f"字段 {k} 中存在 None 值，请检查数据集生成逻辑")
                tensor_collated[k] = torch.stack(collated[k])
            # 3. 其他非Tensor字段（如未来可能新增的标量）直接保留列表
            else:
                tensor_collated[k] = collated[k]
        
        return tensor_collated

    @property
    def num_ids(self):
        """返回数据集中的身份总数"""
        return len(self.dataset.label_mapping)

# # 示例用法
# if __name__ == "__main__":
#     # 训练集配置
#     train_loader = ReIDDataLoader(
#         img_dir="./data/train/images",
#         caption_path="./data/train/captions.json",
#         pseudo_label_path="./data/train/pseudo_labels",
#         batch_size=64,
#         num_workers=4,
#         is_train=True,
#         use_identity_sampler=True,
#         num_ids_per_batch=16,
#         num_instances_per_id=4
#     )
    
#     # 获取数据加载器
#     train_data_loader = train_loader.get_loader()
#     print(f"训练集身份数量: {train_loader.num_ids}")
#     print(f"训练集批次大小: {train_loader.batch_size}")
    
#     # 测试集配置
#     test_loader = ReIDDataLoader(
#         img_dir="./data/test/images",
#         caption_path="./data/test/captions.json",
#         is_train=False,
#         batch_size=64,
#         num_workers=4
#     )
    
#     test_data_loader = test_loader.get_loader()
#     print(f"测试集身份数量: {test_loader.num_ids}")