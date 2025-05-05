#!/usr/bin/env python3
"""
测试OpenCV和InsightFace人脸识别器的功能
"""
import os
import cv2
import numpy as np
import logging
from src.core.face_recognizer import create_face_recognizer, OpenCVFaceRecognizer, InsightFaceRecognizer

# 配置日志
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_recognizers():
    """测试两种人脸识别器的功能"""
    # 测试OpenCV人脸识别器
    logger.info("测试OpenCV人脸识别器...")
    opencv_recognizer = create_face_recognizer(recognizer_type="opencv")
    
    # 测试InsightFace人脸识别器
    logger.info("测试InsightFace人脸识别器...")
    insightface_recognizer = create_face_recognizer(
        recognizer_type="insightface",
        model_name="buffalo_s"
    )
    
    # 加载测试图像
    test_image_path = "test_face.jpg"
    if not os.path.exists(test_image_path):
        logger.warning(f"测试图像 {test_image_path} 不存在，尝试使用随机图像")
        # 创建一个随机图像用于测试
        random_image = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        cv2.imwrite(test_image_path, random_image)
    
    # 读取测试图像
    test_image = cv2.imread(test_image_path)
    if test_image is None:
        logger.error(f"无法读取测试图像: {test_image_path}")
        return
    
    # 测试OpenCV特征提取
    if opencv_recognizer is not None:
        logger.info("使用OpenCV提取特征...")
        try:
            opencv_feature = opencv_recognizer.extract_feature(test_image)
            if opencv_feature is not None:
                logger.info(f"OpenCV特征向量长度: {len(opencv_feature)}")
                logger.info(f"OpenCV特征向量前5个值: {opencv_feature[:5]}")
            else:
                logger.error("OpenCV特征提取失败")
        except Exception as e:
            logger.error(f"OpenCV特征提取出错: {e}")
    else:
        logger.error("OpenCV人脸识别器创建失败")
    
    # 测试InsightFace特征提取
    if insightface_recognizer is not None:
        logger.info("使用InsightFace提取特征...")
        try:
            insightface_feature = insightface_recognizer.extract_feature(test_image)
            if insightface_feature is not None:
                logger.info(f"InsightFace特征向量长度: {len(insightface_feature)}")
                logger.info(f"InsightFace特征向量前5个值: {insightface_feature[:5]}")
            else:
                logger.error("InsightFace特征提取失败")
        except Exception as e:
            logger.error(f"InsightFace特征提取出错: {e}")
    else:
        logger.error("InsightFace人脸识别器创建失败")
    
    # 如果两个特征向量都提取成功，测试相似度计算
    if opencv_recognizer is not None and insightface_recognizer is not None:
        opencv_feature = opencv_recognizer.extract_feature(test_image)
        insightface_feature = insightface_recognizer.extract_feature(test_image)
        
        if opencv_feature is not None and insightface_feature is not None:
            # 测试OpenCV的相似度计算
            logger.info("测试OpenCV相似度计算...")
            try:
                # 与自身比较应该接近1.0
                similarity = opencv_recognizer.compute_similarity(opencv_feature, opencv_feature)
                logger.info(f"OpenCV自身相似度: {similarity}")
            except Exception as e:
                logger.error(f"OpenCV相似度计算出错: {e}")
            
            # 测试InsightFace的相似度计算
            logger.info("测试InsightFace相似度计算...")
            try:
                # 与自身比较应该接近1.0
                similarity = insightface_recognizer.compute_similarity(insightface_feature, insightface_feature)
                logger.info(f"InsightFace自身相似度: {similarity}")
            except Exception as e:
                logger.error(f"InsightFace相似度计算出错: {e}")

if __name__ == "__main__":
    test_recognizers()
