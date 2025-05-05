"""
测试InsightFace人脸聚类效果
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

def test_insightface_clustering():
    """测试InsightFace人脸聚类效果"""
    logger.info("测试InsightFace人脸聚类效果...")

    # 创建InsightFace人脸识别器
    recognizer = create_face_recognizer(
        recognizer_type="insightface",
        model_name="buffalo_l"
    )

    if recognizer is None:
        logger.error("创建InsightFace人脸识别器失败")
        return False

    # 创建测试目录
    test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_faces")
    os.makedirs(test_dir, exist_ok=True)

    # 生成测试图像
    face_size = 112  # InsightFace模型输入大小

    # 生成两个完全不同的"人脸"图像
    face1 = np.random.randint(0, 255, (face_size, face_size, 3), dtype=np.uint8)
    face2 = np.random.randint(0, 255, (face_size, face_size, 3), dtype=np.uint8)

    # 添加一些结构化特征
    # 人脸1: 添加眼睛和嘴巴的模拟
    cv2.circle(face1, (40, 40), 10, (255, 255, 255), -1)  # 左眼
    cv2.circle(face1, (70, 40), 10, (255, 255, 255), -1)  # 右眼
    cv2.ellipse(face1, (55, 70), (20, 10), 0, 0, 180, (255, 255, 255), -1)  # 嘴巴

    # 人脸2: 不同位置的眼睛和嘴巴
    cv2.circle(face2, (35, 45), 12, (255, 255, 255), -1)  # 左眼
    cv2.circle(face2, (75, 45), 12, (255, 255, 255), -1)  # 右眼
    cv2.rectangle(face2, (45, 75), (65, 85), (255, 255, 255), -1)  # 嘴巴

    # 生成face1的变体 (保持眼睛和嘴巴位置，但改变背景和细节)
    face1_variant = face1.copy()

    # 改变部分背景，但保留面部特征
    mask = np.zeros((face_size, face_size), dtype=np.uint8)
    cv2.circle(mask, (40, 40), 12, 255, -1)  # 左眼区域
    cv2.circle(mask, (70, 40), 12, 255, -1)  # 右眼区域
    cv2.ellipse(mask, (55, 70), (22, 12), 0, 0, 180, 255, -1)  # 嘴巴区域

    # 在保留面部特征的同时修改背景
    random_bg = np.random.randint(0, 255, (face_size, face_size, 3), dtype=np.uint8)
    face1_variant = np.where(mask[:, :, np.newaxis] > 0, face1, random_bg)

    # 保存测试图像
    cv2.imwrite(os.path.join(test_dir, "face1.jpg"), face1)
    cv2.imwrite(os.path.join(test_dir, "face2.jpg"), face2)
    cv2.imwrite(os.path.join(test_dir, "face1_variant.jpg"), face1_variant)

    # 提取特征
    feature1 = recognizer.extract_feature(face1)
    feature2 = recognizer.extract_feature(face2)
    feature1_variant = recognizer.extract_feature(face1_variant)

    if feature1 is None or feature2 is None or feature1_variant is None:
        logger.error("特征提取失败")
        return False

    # 计算相似度
    similarity_1_2 = recognizer.compute_similarity(feature1, feature2)
    similarity_1_variant = recognizer.compute_similarity(feature1, feature1_variant)

    logger.info(f"不同人脸相似度 (face1 vs face2): {similarity_1_2}")
    logger.info(f"相似人脸相似度 (face1 vs face1_variant): {similarity_1_variant}")

    # 测试不同阈值的分类结果
    thresholds = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    logger.info("\n测试不同阈值的分类结果:")
    for threshold in thresholds:
        # 不同人脸是否被归为同一类
        diff_same_class = similarity_1_2 >= threshold
        # 相似人脸是否被归为同一类
        similar_same_class = similarity_1_variant >= threshold

        logger.info(f"阈值 {threshold}:")
        logger.info(f"  - 不同人脸归为同一类: {diff_same_class}")
        logger.info(f"  - 相似人脸归为同一类: {similar_same_class}")

    # 测试真实场景
    logger.info("\n测试真实场景下的阈值效果:")

    # 模拟一个人脸分组列表
    face_groups = [
        {"id": 1, "feature_vector": feature1},
        {"id": 2, "feature_vector": feature2}
    ]

    # 测试新人脸的分配
    new_face = feature1_variant

    for threshold in thresholds:
        # 计算与现有分组的相似度
        similarities = []
        for group in face_groups:
            similarity = recognizer.compute_similarity(new_face, group["feature_vector"])
            similarities.append((group["id"], similarity))

        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)

        # 获取最高相似度
        best_group_id, max_similarity = similarities[0]

        # 判断分配结果
        if max_similarity >= threshold:
            result = f"分配到现有分组 {best_group_id}，相似度: {max_similarity:.4f}"
        else:
            result = f"创建新分组，最高相似度 {max_similarity:.4f} 低于阈值 {threshold}"

        logger.info(f"阈值 {threshold}: {result}")

    return True

if __name__ == "__main__":
    test_insightface_clustering()
