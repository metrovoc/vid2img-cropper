#!/usr/bin/env python3
"""
下载YuNet模型
"""
import os
import sys

def download_yunet_model():
    """下载YuNet模型"""
    # 创建models目录
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # 模型路径
    model_path = os.path.join(models_dir, "face_detection_yunet_2023mar.onnx")
    
    # 检查模型文件是否存在
    if os.path.exists(model_path):
        print(f"模型文件已存在: {model_path}")
        return
    
    # 下载模型
    model_url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    try:
        import urllib.request
        print(f"正在下载YuNet模型: {model_url}")
        urllib.request.urlretrieve(model_url, model_path)
        print(f"模型已下载到: {model_path}")
    except Exception as e:
        print(f"模型下载失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download_yunet_model()
