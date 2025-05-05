"""
测试相似度增强效果
"""
import os
import sys
import logging
import numpy as np
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.config import Config
from src.core.face_recognizer import create_face_recognizer, InsightFaceRecognizer

def test_similarity_enhancement():
    """测试相似度增强效果"""
    logger.info("测试相似度增强效果...")
    
    # 创建InsightFace人脸识别器
    recognizer = create_face_recognizer(
        recognizer_type="insightface",
        model_name="buffalo_l"
    )
    
    if recognizer is None:
        logger.error("创建InsightFace人脸识别器失败")
        return False
    
    # 创建一些测试特征向量
    # 完全相同的特征向量
    feature1 = np.random.rand(512).tolist()
    feature1_same = feature1.copy()
    
    # 稍微不同的特征向量 (95%相似)
    feature1_similar = feature1.copy()
    indices = np.random.choice(len(feature1), int(len(feature1) * 0.05), replace=False)
    for i in indices:
        feature1_similar[i] = np.random.rand()
    
    # 较大不同的特征向量 (80%相似)
    feature1_different = feature1.copy()
    indices = np.random.choice(len(feature1), int(len(feature1) * 0.2), replace=False)
    for i in indices:
        feature1_different[i] = np.random.rand()
    
    # 完全不同的特征向量
    feature2 = np.random.rand(512).tolist()
    
    # 测试相似度计算
    logger.info("测试相似度计算...")
    
    # 完全相同的特征向量
    similarity_same = recognizer.compute_similarity(feature1, feature1_same)
    logger.info(f"完全相同的特征向量相似度: {similarity_same:.6f}")
    
    # 稍微不同的特征向量
    similarity_similar = recognizer.compute_similarity(feature1, feature1_similar)
    logger.info(f"稍微不同的特征向量相似度 (95%相似): {similarity_similar:.6f}")
    
    # 较大不同的特征向量
    similarity_different = recognizer.compute_similarity(feature1, feature1_different)
    logger.info(f"较大不同的特征向量相似度 (80%相似): {similarity_different:.6f}")
    
    # 完全不同的特征向量
    similarity_random = recognizer.compute_similarity(feature1, feature2)
    logger.info(f"完全不同的特征向量相似度: {similarity_random:.6f}")
    
    # 测试不同阈值的分类结果
    thresholds = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    
    logger.info("\n测试不同阈值的分类结果:")
    for threshold in thresholds:
        logger.info(f"阈值 {threshold}:")
        logger.info(f"  - 完全相同: {'同一人' if similarity_same >= threshold else '不同人'}")
        logger.info(f"  - 稍微不同 (95%相似): {'同一人' if similarity_similar >= threshold else '不同人'}")
        logger.info(f"  - 较大不同 (80%相似): {'同一人' if similarity_different >= threshold else '不同人'}")
        logger.info(f"  - 完全不同: {'同一人' if similarity_random >= threshold else '不同人'}")
    
    return True

if __name__ == "__main__":
    test_similarity_enhancement()
