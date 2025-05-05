#!/usr/bin/env python3
"""
路径工具函数
"""
import os
from pathlib import Path


def get_app_dir():
    """
    获取应用程序数据目录
    
    Returns:
        应用程序数据目录路径
    """
    home = Path.home()
    app_dir = os.path.join(home, ".vid2img-cropper")
    os.makedirs(app_dir, exist_ok=True)
    return app_dir


def get_models_dir():
    """
    获取模型目录路径
    
    Returns:
        模型目录路径
    """
    # 使用main.py所在目录下的models目录
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    models_dir = os.path.join(root_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    return models_dir


def get_insightface_models_dir():
    """
    获取InsightFace模型目录路径
    
    Returns:
        InsightFace模型目录路径
    """
    models_dir = get_models_dir()
    insightface_dir = os.path.join(models_dir, "insightface")
    os.makedirs(insightface_dir, exist_ok=True)
    return insightface_dir
