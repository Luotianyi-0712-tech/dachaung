# Download the pre-trained keypoint-RCNN model for COCO dataset
import os
import urllib.request
import argparse
from tqdm import tqdm

def download_file(url, save_path):
    """
    下载文件并显示进度条
    :param url: 下载链接
    :param save_path: 保存路径
    """
    try:
        # 发送请求获取文件信息
        with urllib.request.urlopen(url) as response:
            file_size = int(response.headers.get('Content-Length', 0))
            block_size = 1024  # 1KB
            progress_bar = tqdm(total=file_size, unit='iB', unit_scale=True)
            
            with open(save_path, 'wb') as file:
                while True:
                    block = response.read(block_size)
                    if not block:
                        break
                    file.write(block)
                    progress_bar.update(len(block))
            progress_bar.close()
            
            # 验证文件大小（如果服务器提供了Content-Length）
            if file_size != 0 and os.path.getsize(save_path) != file_size:
                raise Exception("文件下载不完整，大小不匹配")
                
        return True
    except Exception as e:
        print(f"下载失败: {str(e)}")
        # 清理不完整的文件
        if os.path.exists(save_path):
            os.remove(save_path)
        return False

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='下载预训练的keypoint-RCNN模型')
    parser.add_argument('--save_dir', type=str, default='/Volumes/SC/ReID_DC/BLIP-IAPM-ReID/keypointrcnn_resnet50_fpn_coco', help='模型保存目录')
    args = parser.parse_args()
    
    # 模型下载链接和文件名
    url = "https://download.pytorch.org/models/keypointrcnn_resnet50_fpn_coco-9f466800.pth"
    filename = "keypointrcnn_resnet50_fpn_coco-9f466800.pth"
    save_path = os.path.join(args.save_dir, filename)
    
    # 创建保存目录（如果不存在）
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 检查文件是否已存在
    if os.path.exists(save_path):
        print(f'预训练的keypoint-RCNN模型已存在于: {save_path}')
        return
    
    # 下载文件
    print(f'开始下载预训练的keypoint-RCNN模型...')
    if download_file(url, save_path):
        print(f'模型已成功下载至: {save_path}')

if __name__ == "__main__":
    main()
