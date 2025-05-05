"""
使用真实人脸图像测试InsightFace聚类效果
"""
import os
import sys
import logging
import numpy as np
import cv2
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.config import Config
from src.core.face_recognizer import create_face_recognizer, InsightFaceRecognizer

def test_with_real_faces():
    """使用真实人脸图像测试InsightFace聚类效果"""
    logger.info("使用真实人脸图像测试InsightFace聚类效果...")
    
    # 创建InsightFace人脸识别器
    recognizer = create_face_recognizer(
        recognizer_type="insightface",
        model_name="buffalo_l"
    )
    
    if recognizer is None:
        logger.error("创建InsightFace人脸识别器失败")
        return False
    
    # 测试目录
    test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_real_faces")
    os.makedirs(test_dir, exist_ok=True)
    
    # 检查是否有测试图像
    test_images = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    if not test_images:
        logger.warning(f"测试目录 {test_dir} 中没有图像，请添加一些人脸图像用于测试")
        logger.info("您可以将一些人脸图像放入此目录，然后重新运行测试")
        return False
    
    # 提取所有图像的特征
    features = {}
    for image_file in test_images:
        image_path = os.path.join(test_dir, image_file)
        logger.info(f"处理图像: {image_file}")
        
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            logger.warning(f"无法读取图像: {image_path}")
            continue
        
        # 检测人脸
        faces = recognizer.model.get(image)
        
        if not faces:
            logger.warning(f"在图像 {image_file} 中未检测到人脸")
            continue
        
        # 使用最大的人脸
        face = max(faces, key=lambda x: x.bbox[2] * x.bbox[3])
        
        # 获取特征向量
        feature = face.normed_embedding
        
        # 保存特征
        features[image_file] = feature
    
    if len(features) < 2:
        logger.warning("提取的特征数量不足，至少需要2个人脸图像")
        return False
    
    # 计算所有图像之间的相似度
    logger.info("\n计算所有图像之间的相似度:")
    
    # 测试不同阈值
    thresholds = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    
    # 计算所有图像对之间的相似度
    for i, (name1, feature1) in enumerate(features.items()):
        for name2, feature2 in list(features.items())[i+1:]:
            similarity = recognizer.compute_similarity(feature1, feature2)
            logger.info(f"{name1} vs {name2}: 相似度 = {similarity:.4f}")
            
            # 显示不同阈值下的分类结果
            results = []
            for threshold in thresholds:
                if similarity >= threshold:
                    results.append(f"{threshold}: 同一人")
                else:
                    results.append(f"{threshold}: 不同人")
            
            logger.info("  阈值判断结果: " + ", ".join(results))
    
    # 模拟聚类过程
    logger.info("\n模拟聚类过程:")
    
    # 测试不同阈值下的聚类结果
    for threshold in thresholds:
        logger.info(f"\n阈值 {threshold} 的聚类结果:")
        
        # 初始化分组
        groups = []
        
        # 对每个特征进行分组
        for name, feature in features.items():
            # 计算与现有分组的相似度
            max_similarity = 0
            best_group = -1
            
            for i, group in enumerate(groups):
                for group_name, group_feature in group:
                    similarity = recognizer.compute_similarity(feature, group_feature)
                    if similarity > max_similarity:
                        max_similarity = similarity
                        best_group = i
            
            # 如果最大相似度超过阈值，分配到现有分组
            if max_similarity >= threshold and best_group >= 0:
                groups[best_group].append((name, feature))
                logger.info(f"  {name} 分配到分组 {best_group+1}，相似度: {max_similarity:.4f}")
            else:
                # 否则创建新分组
                groups.append([(name, feature)])
                logger.info(f"  {name} 创建新分组 {len(groups)}，最高相似度: {max_similarity:.4f}")
        
        # 显示最终分组结果
        logger.info(f"  共形成 {len(groups)} 个分组:")
        for i, group in enumerate(groups):
            group_members = [name for name, _ in group]
            logger.info(f"  分组 {i+1}: {', '.join(group_members)}")
    
    return True

if __name__ == "__main__":
    test_with_real_faces()
