"""
测试人脸相似度阈值对聚类结果的影响
"""
import os
import sys
import logging
import numpy as np
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.config import Config
from src.core.face_recognizer import create_face_recognizer, InsightFaceRecognizer

def test_face_similarity_thresholds():
    """测试不同相似度阈值对人脸聚类的影响"""
    logger.info("测试不同相似度阈值对人脸聚类的影响...")
    
    # 创建InsightFace人脸识别器
    recognizer = create_face_recognizer(
        recognizer_type="insightface",
        model_name="buffalo_l"
    )
    
    if recognizer is None:
        logger.error("创建InsightFace人脸识别器失败")
        return False
    
    # 测试自身相似度
    logger.info("测试自身相似度...")
    test_image = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    feature = recognizer.extract_feature(test_image)
    
    if feature is None:
        logger.error("特征提取失败")
        return False
    
    # 自身相似度应该接近1.0
    self_similarity = recognizer.compute_similarity(feature, feature)
    logger.info(f"自身相似度: {self_similarity}")
    
    # 测试不同相似度阈值
    thresholds = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]
    
    # 创建两个随机特征向量，模拟不同人脸
    feature1 = np.random.rand(512).tolist()  # InsightFace特征向量通常是512维
    feature2 = np.random.rand(512).tolist()
    
    # 计算两个随机特征向量的相似度
    random_similarity = recognizer.compute_similarity(feature1, feature2)
    logger.info(f"随机特征向量相似度: {random_similarity}")
    
    # 创建一个稍微相似的特征向量（feature1的变体）
    feature1_variant = feature1.copy()
    # 修改20%的元素
    indices = np.random.choice(len(feature1), int(len(feature1) * 0.2), replace=False)
    for i in indices:
        feature1_variant[i] = np.random.rand()
    
    # 计算变体相似度
    variant_similarity = recognizer.compute_similarity(feature1, feature1_variant)
    logger.info(f"变体特征向量相似度: {variant_similarity}")
    
    # 测试不同阈值的分类结果
    logger.info("\n测试不同阈值的分类结果:")
    for threshold in thresholds:
        # 随机向量是否被归为同一类
        random_same_class = random_similarity >= threshold
        # 变体向量是否被归为同一类
        variant_same_class = variant_similarity >= threshold
        
        logger.info(f"阈值 {threshold}:")
        logger.info(f"  - 随机向量归为同一类: {random_same_class}")
        logger.info(f"  - 变体向量归为同一类: {variant_same_class}")
    
    return True

if __name__ == "__main__":
    test_face_similarity_thresholds()
