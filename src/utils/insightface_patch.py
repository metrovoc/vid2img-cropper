#!/usr/bin/env python3
"""
InsightFace库的补丁，用于覆盖默认的模型下载路径
"""
import os
import logging
import importlib.util
from pathlib import Path

logger = logging.getLogger(__name__)

def patch_insightface():
    """
    修补InsightFace库，使其使用我们指定的模型目录
    
    这个函数会尝试修改InsightFace库的模型下载路径，使其使用我们指定的目录
    """
    try:
        # 检查InsightFace是否已安装
        insightface_spec = importlib.util.find_spec("insightface")
        if insightface_spec is None:
            logger.warning("未安装InsightFace库，无法应用补丁")
            return False
        
        # 获取当前INSIGHTFACE_HOME环境变量
        insightface_home = os.environ.get('INSIGHTFACE_HOME', None)
        if not insightface_home:
            logger.warning("INSIGHTFACE_HOME环境变量未设置，无法应用补丁")
            return False
        
        logger.info(f"尝试修补InsightFace库，使用模型目录: {insightface_home}")
        
        # 导入InsightFace的utils模块
        import insightface.utils.filesystem as fs
        import insightface.utils.storage as storage
        
        # 保存原始函数
        original_get_model_dir = fs.get_model_dir
        original_download = storage.download
        original_download_onnx = storage.download_onnx
        
        # 修改get_model_dir函数
        def patched_get_model_dir(name, root=None):
            """修补后的get_model_dir函数，使用INSIGHTFACE_HOME环境变量"""
            if root is None:
                root = os.environ.get('INSIGHTFACE_HOME', '~/.insightface')
            return original_get_model_dir(name, root)
        
        # 修改download函数
        def patched_download(sub_dir, name, force=False, root=None):
            """修补后的download函数，使用INSIGHTFACE_HOME环境变量"""
            if root is None:
                root = os.environ.get('INSIGHTFACE_HOME', '~/.insightface')
            return original_download(sub_dir, name, force, root)
        
        # 修改download_onnx函数
        def patched_download_onnx(sub_dir, model_file, force=False, root=None, download_zip=False):
            """修补后的download_onnx函数，使用INSIGHTFACE_HOME环境变量"""
            if root is None:
                root = os.environ.get('INSIGHTFACE_HOME', '~/.insightface')
            return original_download_onnx(sub_dir, model_file, force, root, download_zip)
        
        # 应用补丁
        fs.get_model_dir = patched_get_model_dir
        storage.download = patched_download
        storage.download_onnx = patched_download_onnx
        
        logger.info("成功修补InsightFace库")
        return True
    except Exception as e:
        logger.error(f"修补InsightFace库失败: {e}")
        return False
