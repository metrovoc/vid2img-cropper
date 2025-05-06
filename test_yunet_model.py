#!/usr/bin/env python3
"""
测试YuNet模型加载
"""
import os
import cv2
import logging
from src.utils.paths import get_models_dir
from src.core.face_detector import YuNetFaceDetector

# 配置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_yunet_model():
    """测试YuNet模型加载"""
    # 获取模型目录
    models_dir = get_models_dir()
    model_path = os.path.join(models_dir, "face_detection_yunet_2023mar.onnx")

    # 检查模型文件是否存在
    if os.path.exists(model_path):
        logger.info(f"模型文件存在: {model_path}")
    else:
        logger.error(f"模型文件不存在: {model_path}")
        logger.info("尝试下载模型...")

        # 创建目录
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # 下载模型
        model_url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
        try:
            import urllib.request
            urllib.request.urlretrieve(model_url, model_path)
            logger.info(f"模型已下载到: {model_path}")
        except Exception as e:
            logger.error(f"模型下载失败: {e}")
            return

    # 尝试加载模型
    logger.info("尝试加载YuNet模型...")
    detector = YuNetFaceDetector(model_path=model_path)

    if detector.detector is None:
        logger.error("YuNet模型加载失败")
    else:
        logger.info("YuNet模型加载成功")

        # 尝试检测一个简单的图像
        logger.info("创建测试图像...")
        test_image = cv2.imread("test_image.jpg") if os.path.exists("test_image.jpg") else None

        if test_image is None:
            # 创建一个简单的测试图像
            test_image = cv2.imread(os.path.join(os.path.dirname(__file__), "assets", "test_image.jpg")) if os.path.exists(os.path.join(os.path.dirname(__file__), "assets", "test_image.jpg")) else None

        if test_image is None:
            # 如果没有测试图像，创建一个空白图像
            logger.info("创建空白测试图像...")
            test_image = cv2.imread(os.path.join(os.path.dirname(__file__), "assets", "logo.png")) if os.path.exists(os.path.join(os.path.dirname(__file__), "assets", "logo.png")) else None

        if test_image is None:
            # 如果仍然没有图像，创建一个空白图像
            logger.info("创建空白测试图像...")
            test_image = cv2.imread(os.path.join(os.path.dirname(__file__), "assets", "icon.png")) if os.path.exists(os.path.join(os.path.dirname(__file__), "assets", "icon.png")) else None

        if test_image is None:
            # 如果仍然没有图像，创建一个空白图像
            logger.info("创建空白测试图像...")
            test_image = cv2.imread(os.path.join(os.path.dirname(__file__), "icon.png")) if os.path.exists(os.path.join(os.path.dirname(__file__), "icon.png")) else None

        if test_image is None:
            # 如果仍然没有图像，创建一个空白图像
            logger.info("创建空白测试图像...")
            test_image = cv2.imread(os.path.join(os.path.dirname(__file__), "logo.png")) if os.path.exists(os.path.join(os.path.dirname(__file__), "logo.png")) else None

        if test_image is None:
            # 如果仍然没有图像，创建一个空白图像
            logger.info("创建空白测试图像...")
            test_image = cv2.imread(os.path.join(os.path.dirname(__file__), "src", "assets", "logo.png")) if os.path.exists(os.path.join(os.path.dirname(__file__), "src", "assets", "logo.png")) else None

        if test_image is None:
            # 如果仍然没有图像，创建一个空白图像
            logger.info("创建空白测试图像...")
            test_image = cv2.imread(os.path.join(os.path.dirname(__file__), "src", "assets", "icon.png")) if os.path.exists(os.path.join(os.path.dirname(__file__), "src", "assets", "icon.png")) else None

        if test_image is None:
            # 如果仍然没有图像，创建一个空白图像
            logger.info("创建空白测试图像...")
            import numpy as np
            test_image = np.zeros((320, 320, 3), dtype=np.uint8)

        # 检测人脸
        logger.info("尝试检测人脸...")
        faces = detector.detect(test_image)
        logger.info(f"检测到 {len(faces)} 个人脸")

if __name__ == "__main__":
    test_yunet_model()
