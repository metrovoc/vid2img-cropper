#!/usr/bin/env python3
"""
测试人脸识别器功能
"""
import os
import sys
import logging
import cv2
import numpy as np

# 配置日志
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 导入人脸识别器
from src.core.face_recognizer import create_face_recognizer

def test_face_recognizer():
    """测试人脸识别器功能"""
    print("测试OpenCV人脸识别器...")
    opencv_recognizer = create_face_recognizer(recognizer_type="opencv")
    if opencv_recognizer is None:
        print("OpenCV人脸识别器创建失败")
    else:
        print("OpenCV人脸识别器创建成功")
    
    print("\n测试InsightFace人脸识别器...")
    insightface_recognizer = create_face_recognizer(recognizer_type="insightface")
    if insightface_recognizer is None:
        print("InsightFace人脸识别器创建失败")
    else:
        print("InsightFace人脸识别器创建成功")
    
    # 测试带有model_name参数的创建
    print("\n测试带有model_name参数的InsightFace人脸识别器...")
    insightface_recognizer_with_model = create_face_recognizer(
        recognizer_type="insightface", 
        model_name="buffalo_s"
    )
    if insightface_recognizer_with_model is None:
        print("带有model_name参数的InsightFace人脸识别器创建失败")
    else:
        print("带有model_name参数的InsightFace人脸识别器创建成功")
    
    # 测试带有model_name参数的OpenCV人脸识别器（应该忽略model_name参数）
    print("\n测试带有model_name参数的OpenCV人脸识别器...")
    opencv_recognizer_with_model = create_face_recognizer(
        recognizer_type="opencv", 
        model_name="some_model"  # 这个参数应该被忽略
    )
    if opencv_recognizer_with_model is None:
        print("带有model_name参数的OpenCV人脸识别器创建失败")
    else:
        print("带有model_name参数的OpenCV人脸识别器创建成功")

if __name__ == "__main__":
    test_face_recognizer()
