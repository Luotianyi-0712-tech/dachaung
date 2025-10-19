import torch
import os
import numpy as np
from tqdm import tqdm
from model.blip_iapm import BlipIAPMReID
from codes.utils.data_loader import ReIDDataLoader
from codes.utils.metric import compute_rank1_map
from codes.utils.visualization import plot_part_weights
from codes.config import Config

def test():
    # 1. 加载配置
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. 加载数据（测试集+查询集）
    # 测试集（数据库）
    test_data_loader = ReIDDataLoader(
        img_dir=cfg.test_img_dir,
        caption_path=cfg.test_caption_path,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        is_train=False
    )
    test_loader = test_data_loader.get_loader()
    # 查询集
    query_data_loader = ReIDDataLoader(
        img_dir=cfg.query_img_dir,
        caption_path=cfg.query_caption_path,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        is_train=False
    )
    query_loader = query_data_loader.get_loader()
    
    # 3. 加载模型（测试模式）
    model = BlipIAPMReID(
        blip_pretrained_path=cfg.blip_pretrained_path,
        num_parts=cfg.num_parts,
        num_ids=cfg.num_train_ids
    ).to(device)
    # 加载训练好的checkpoint
    checkpoint = torch.load(cfg.best_checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 4. 提取测试集和查询集的多模态特征
    def extract_features(loader):
        feats = []
        weights = []
        labels = []
        with torch.no_grad():
            for batch in tqdm(loader, desc="Extracting features"):
                image_inputs = batch['image'].to(device)
                text_inputs = {
                    'input_ids': batch['text_input_ids'].to(device),
                    'attention_mask': batch['text_attention_mask'].to(device)
                }
                # 测试时返回多模态特征和部件权重
                cross_feat, part_weights = model(image_inputs, text_inputs)
                feats.append(cross_feat.cpu().numpy())
                weights.append(part_weights.cpu().numpy())
                labels.append(batch['label'].numpy())
        return np.concatenate(feats), np.concatenate(weights), np.concatenate(labels)
    
    test_feats, test_weights, test_labels = extract_features(test_loader)
    query_feats, query_weights, query_labels = extract_features(query_loader)
    
    # 5. 计算评价指标（Rank-1、mAP）
    # 图像检索图像（传统ReID）
    rank1_img2img, map_img2img = compute_rank1_map(query_feats, test_feats, query_labels, test_labels)
    # 文本检索图像（多模态ReID）
    rank1_txt2img, map_txt2img = compute_rank1_map(query_feats, test_feats, query_labels, test_labels)
    # 图像检索文本（多模态ReID）
    rank1_img2txt, map_img2txt = compute_rank1_map(query_feats, test_feats, query_labels, test_labels)
    
    # 打印结果
    print("="*50)
    print(f"Image-to-Image ReID: Rank-1={rank1_img2img:.2f}%, mAP={map_img2img:.2f}%")
    print(f"Text-to-Image ReID: Rank-1={rank1_txt2img:.2f}%, mAP={map_txt2img:.2f}%")
    print(f"Image-to-Text ReID: Rank-1={rank1_img2txt:.2f}%, mAP={map_img2txt:.2f}%")
    print("="*50)
    
    # 6. 可解释性可视化（随机选10个查询样本，展示IWM权重）
    sample_idx = np.random.choice(len(query_weights), 10, replace=False)
    for idx in sample_idx:
        part_weights = query_weights[idx]
        img_path = query_data_loader.dataset[idx]['img_path']
        # 绘制部件权重图（参考原论文图8）
        plot_part_weights(
            img_path=img_path,
            part_weights=part_weights,
            save_path=os.path.join(cfg.result_dir, f"weight_vis_{idx}.png")
        )
    
    print("Test finished! Results saved to", cfg.result_dir)

if __name__ == "__main__":
    test()