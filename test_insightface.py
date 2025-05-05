#!/usr/bin/env python3
"""
测试InsightFace模型下载和使用
"""
import os
import sys
import logging
import cv2
import numpy as np
from src.core.face_recognizer import create_face_recognizer

# 配置日志
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_insightface():
    """测试InsightFace人脸识别器"""
    logger.info("测试InsightFace人脸识别器...")
    
    # 创建InsightFace人脸识别器
    recognizer = create_face_recognizer(
        recognizer_type="insightface",
        model_name="buffalo_l"  # 可以尝试 buffalo_s 或 buffalo_m
    )
    
    if recognizer is None:
        logger.error("创建InsightFace人脸识别器失败")
        return False
    
    # 创建一个测试图像
    logger.info("创建测试图像...")
    test_image = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    
    # 提取特征
    logger.info("提取人脸特征...")
    feature = recognizer.extract_feature(test_image)
    
    if feature is None:
        logger.error("特征提取失败")
        return False
    
    logger.info(f"特征提取成功，特征向量维度: {len(feature)}")
    return True

if __name__ in {"__main__", "__mp_main__"}:
    success = test_insightface()
    if success:
        logger.info("InsightFace测试成功")
        sys.exit(0)
    else:
        logger.error("InsightFace测试失败")
        sys.exit(1)
