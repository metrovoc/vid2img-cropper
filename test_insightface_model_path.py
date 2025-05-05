#!/usr/bin/env python3
"""
测试InsightFace模型路径
"""
import os
import sys
import logging
from src.utils.paths import get_insightface_models_dir
from src.core.face_recognizer import create_face_recognizer

# 配置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_insightface_model_path():
    """测试InsightFace模型路径"""
    # 获取InsightFace模型目录
    model_name = "buffalo_l"
    model_dir = os.path.join(get_insightface_models_dir(), model_name)
    
    logger.info(f"InsightFace模型目录: {model_dir}")
    
    # 检查目录是否存在
    if not os.path.exists(model_dir):
        logger.info(f"创建模型目录: {model_dir}")
        os.makedirs(model_dir, exist_ok=True)
    
    # 检查模型文件是否存在
    model_files = [
        "w600k_mbf.onnx",  # 旧版本
        "w600k_r50.onnx",  # 新版本
        "recognition.onnx"  # 可能的通用名称
    ]
    
    found_models = []
    for model_file in model_files:
        model_path = os.path.join(model_dir, model_file)
        if os.path.exists(model_path):
            found_models.append(model_file)
            logger.info(f"找到模型文件: {model_path}")
    
    if not found_models:
        logger.info("未找到任何模型文件，尝试创建人脸识别器以触发下载")
        
        # 创建人脸识别器
        recognizer = create_face_recognizer(
            recognizer_type="insightface",
            model_name=model_name
        )
        
        # 再次检查模型文件
        for model_file in model_files:
            model_path = os.path.join(model_dir, model_file)
            if os.path.exists(model_path):
                found_models.append(model_file)
                logger.info(f"下载后找到模型文件: {model_path}")
        
        if not found_models:
            logger.error("下载后仍未找到任何模型文件")
            return False
    
    # 列出目录中的所有文件
    logger.info(f"模型目录 {model_dir} 中的所有文件:")
    for root, dirs, files in os.walk(model_dir):
        for file in files:
            logger.info(f"  - {file}")
    
    return True

if __name__ in {"__main__", "__mp_main__"}:
    success = test_insightface_model_path()
    if success:
        logger.info("InsightFace模型路径测试成功")
        sys.exit(0)
    else:
        logger.error("InsightFace模型路径测试失败")
        sys.exit(1)
