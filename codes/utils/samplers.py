import random
import numpy as np
from torch.utils.data.sampler import Sampler

class RandomIdentitySampler(Sampler):
    """
    行人重识别专用采样器
    确保每个批次中:
    - 包含 num_instances 个不同身份
    - 每个身份有 num_instances 个样本
    """
    def __init__(self, dataset, num_ids, num_instances):
        """
        参数:
            dataset: 数据集，需要有dataset.idx_to_class属性，映射索引到身份ID
            num_ids: 每个批次中的身份数量
            num_instances: 每个身份在批次中的样本数量
        """
        self.dataset = dataset
        self.num_ids = num_ids  # 每个批次的身份数
        self.num_instances = num_instances  # 每个身份的样本数
        self.batch_size = num_ids * num_instances  # 批次大小
        
        # 构建身份到样本索引的映射
        self.id_to_indices = {}
        for idx, data in enumerate(dataset):
            pid = data["label"]
            camid = data["camid"]
            if pid not in self.id_to_indices:
                self.id_to_indices[pid] = []
            self.id_to_indices[pid].append(idx)
            print(f"Index: {idx}, PID: {pid}, CamID: {camid}")  # 调试信息

        # 过滤样本数不足的身份
        self.valid_ids = [pid for pid, indices in self.id_to_indices.items() if len(indices) >= num_instances]
        self.num_valid_ids = len(self.valid_ids)
        
        if self.num_valid_ids == 0:
            raise ValueError("没有足够的身份满足每个身份至少有{}个样本".format(num_instances))
        
        # 为每个身份准备足够的样本（重复采样）
        self.id_buckets = []
        for pid in self.valid_ids:
            indices = self.id_to_indices[pid]
            # 如果样本不足，循环采样补齐
            if len(indices) < self.num_instances:
                repeated = np.random.choice(indices, size=self.num_instances, replace=True)
            else:
                repeated = np.random.choice(indices, size=self.num_instances, replace=False)
            self.id_buckets.append(repeated)

    def __iter__(self):
        # 打乱身份顺序
        np.random.shuffle(self.valid_ids)
        
        batch = []
        for i in range(0, self.num_valid_ids, self.num_ids):
            # 选择当前批次的身份
            current_ids = self.valid_ids[i:i+self.num_ids]
            if len(current_ids) < self.num_ids:
                break  # 剩余身份不足，跳过
            
            # 为每个身份选择样本
            for pid in current_ids:
                # 找到该身份的索引
                idx = self.valid_ids.index(pid)
                # 随机选择num_instances个样本
                selected = np.random.choice(self.id_buckets[idx], size=self.num_instances, replace=False)
                batch.extend(selected.tolist())
                
                # 当批次满了就返回
                if len(batch) >= self.batch_size:
                    yield batch[:self.batch_size]
                    batch = batch[self.batch_size:]
        
        # 处理剩余样本
        if batch:
            yield batch

    def __len__(self):
        # 计算总批次数量
        return (self.num_valid_ids // self.num_ids) * self.batch_size // self.batch_size
