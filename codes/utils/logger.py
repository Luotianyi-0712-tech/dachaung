import os
import time
import logging
from tensorboardX import SummaryWriter
from codes.config import Config

class TensorboardLogger:
    """日志工具（记录训练/测试过程，适配原论文实验记录需求）"""
    def __init__(self, log_dir=None):
        # 初始化日志目录（基于配置）
        cfg = Config()
        self.log_dir = log_dir if log_dir is not None else cfg.LOG_DIR
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 1. TensorBoard日志（记录损失、指标曲线）
        self.tb_writer = SummaryWriter(log_dir=os.path.join(self.log_dir, "tensorboard"))
        
        # 2. 文本日志（记录详细参数、迭代信息）
        self.text_logger = logging.getLogger("IAPM-ReID")
        self.text_logger.setLevel(logging.INFO)
        # 避免重复添加handler
        if not self.text_logger.handlers:
            # 文件handler（按时间命名）
            log_file = os.path.join(self.log_dir, f"log_{time.strftime('%Y%m%d_%H%M%S')}.txt")
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            # 控制台handler
            console_handler = logging.StreamHandler()
            # 日志格式（包含时间、模块、级别、内容）
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            self.text_logger.addHandler(file_handler)
            self.text_logger.addHandler(console_handler)
        
        # 记录配置信息（原论文实验参数，确保可复现）
        self.log_config(cfg)

    def log_config(self, cfg):
        """记录实验配置（原论文数据集、模型、训练参数等）"""
        self.text_logger.info("="*50 + " Experiment Config " + "="*50)
        self.text_logger.info(f"Dataset: {cfg.USE_DATASET}")
        self.text_logger.info(f"Image Size: {cfg.IMG_SIZE} (H×W)")
        self.text_logger.info(f"Number of Parts: {cfg.num_parts} (aligned with paper: 7)")
        self.text_logger.info(f"Batch Size: {cfg.batch_size} (paper: 128)")
        self.text_logger.info(f"Max Epochs: {cfg.MAX_EPOCHS} (paper: 100)")
        self.text_logger.info(f"Initial LR (Backbone): {cfg.INIT_LR['backbone']} (paper: 0.0001)")
        self.text_logger.info(f"Initial LR (IWM/EM): {cfg.INIT_LR['iwm_em']} (paper: 0.0002)")
        self.text_logger.info(f"SPTL Alpha: {cfg.SPTL_ALPHA} (paper: 1.2)")
        self.text_logger.info(f"LOSS Coefficients (λ, β): {cfg.LOSS_COEFF['lambda_sptl']}, {cfg.LOSS_COEFF['beta_center']} (paper: 1.0, 0.0005)")
        self.text_logger.info("="*108)

    def log_scalar(self, tag, value, step):
        """记录标量（损失、指标等，对应原论文实验曲线）"""
        # TensorBoard记录
        self.tb_writer.add_scalar(tag, value, step)
        # 文本日志记录（每10步记录一次，避免冗余）
        if step % 10 == 0:
            self.text_logger.info(f"Step {step} - {tag}: {value:.6f}")

    def log_figure(self, tag, figure, step):
        """记录图像（可解释性权重图、伪标签图等，对应原论文图8、图10）"""
        self.tb_writer.add_figure(tag, figure, step)
        self.text_logger.info(f"Step {step} - Figure saved: {tag}")

    def log_test_metrics(self, metrics, dataset_name):
        """记录测试指标（Rank-1、mAP，对应原论文3.3节对比实验）"""
        self.text_logger.info("="*50 + f" Test Metrics ({dataset_name}) " + "="*50)
        self.text_logger.info(f"Image-to-Image ReID - Rank-1: {metrics['img2img_rank1']:.2f}%, mAP: {metrics['img2img_map']:.2f}%")
        self.text_logger.info(f"Text-to-Image ReID - Rank-1: {metrics['txt2img_rank1']:.2f}%, mAP: {metrics['txt2img_map']:.2f}%")
        self.text_logger.info(f"Image-to-Text ReID - Rank-1: {metrics['img2txt_rank1']:.2f}%, mAP: {metrics['img2txt_map']:.2f}%")
        self.text_logger.info("="*108)

    def close(self):
        """关闭日志器"""
        self.tb_writer.close()
        self.text_logger.info("Experiment finished! Logger closed.")