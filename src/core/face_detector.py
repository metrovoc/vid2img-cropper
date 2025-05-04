"""
人脸检测模块，负责检测图像中的人脸
"""
import os
import cv2
import numpy as np
from pathlib import Path


class FaceDetector:
    """人脸检测器基类"""
    
    def __init__(self, confidence_threshold=0.6):
        """
        初始化人脸检测器
        
        Args:
            confidence_threshold: 置信度阈值
        """
        self.confidence_threshold = confidence_threshold
    
    def detect(self, image):
        """
        检测图像中的人脸
        
        Args:
            image: 输入图像 (OpenCV格式)
        
        Returns:
            检测结果列表，每个结果为 [x, y, width, height, confidence]
        """
        raise NotImplementedError("子类必须实现detect方法")


class YuNetFaceDetector(FaceDetector):
    """使用YuNet模型的人脸检测器"""
    
    def __init__(self, model_path=None, confidence_threshold=0.6, input_size=(320, 320)):
        """
        初始化YuNet人脸检测器
        
        Args:
            model_path: 模型文件路径，如果为None则尝试下载
            confidence_threshold: 置信度阈值
            input_size: 模型输入大小
        """
        super().__init__(confidence_threshold)
        self.input_size = input_size
        self.model_path = self._get_model_path(model_path)
        self.detector = self._load_model()
    
    def _get_model_path(self, model_path):
        """获取模型路径，如果未指定则使用默认路径"""
        if model_path:
            return model_path
        
        # 默认模型路径
        default_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                    "models", "face_detection_yunet_2023mar.onnx")
        
        # 如果模型不存在，尝试下载
        if not os.path.exists(default_path):
            self._download_model(default_path)
        
        return default_path
    
    def _download_model(self, save_path):
        """下载YuNet模型"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        print("正在下载YuNet人脸检测模型...")
        model_url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
        
        try:
            import urllib.request
            urllib.request.urlretrieve(model_url, save_path)
            print(f"模型已下载到: {save_path}")
        except Exception as e:
            print(f"模型下载失败: {e}")
            print("请手动下载模型并放置到models目录")
            raise
    
    def _load_model(self):
        """加载YuNet模型"""
        detector = cv2.FaceDetectorYN.create(
            model=self.model_path,
            config="",
            input_size=self.input_size,
            score_threshold=self.confidence_threshold,
            backend_id=cv2.dnn.DNN_BACKEND_DEFAULT,
            target_id=cv2.dnn.DNN_TARGET_CPU
        )
        return detector
    
    def detect(self, image):
        """
        检测图像中的人脸
        
        Args:
            image: 输入图像 (OpenCV格式)
        
        Returns:
            检测结果列表，每个结果为 [x, y, width, height, confidence]
        """
        # 调整图像大小以适应模型输入
        height, width, _ = image.shape
        self.detector.setInputSize((width, height))
        
        # 执行检测
        _, faces = self.detector.detect(image)
        
        # 如果没有检测到人脸，返回空列表
        if faces is None:
            return []
        
        # 转换结果格式
        results = []
        for face in faces:
            x, y, w, h, confidence = face[0], face[1], face[2], face[3], face[-1]
            if confidence >= self.confidence_threshold:
                results.append([int(x), int(y), int(w), int(h), float(confidence)])
        
        return results


class AnimeFaceDetector(FaceDetector):
    """使用级联分类器的动漫人脸检测器"""
    
    def __init__(self, model_path=None, confidence_threshold=0.6):
        """
        初始化动漫人脸检测器
        
        Args:
            model_path: 级联分类器XML文件路径，如果为None则尝试下载
            confidence_threshold: 置信度阈值
        """
        super().__init__(confidence_threshold)
        self.model_path = self._get_model_path(model_path)
        self.detector = self._load_model()
    
    def _get_model_path(self, model_path):
        """获取模型路径，如果未指定则使用默认路径"""
        if model_path:
            return model_path
        
        # 默认模型路径
        default_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                    "models", "lbpcascade_animeface.xml")
        
        # 如果模型不存在，尝试下载
        if not os.path.exists(default_path):
            self._download_model(default_path)
        
        return default_path
    
    def _download_model(self, save_path):
        """下载动漫人脸检测模型"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        print("正在下载动漫人脸检测模型...")
        model_url = "https://raw.githubusercontent.com/nagadomi/lbpcascade_animeface/master/lbpcascade_animeface.xml"
        
        try:
            import urllib.request
            urllib.request.urlretrieve(model_url, save_path)
            print(f"模型已下载到: {save_path}")
        except Exception as e:
            print(f"模型下载失败: {e}")
            print("请手动下载模型并放置到models目录")
            raise
    
    def _load_model(self):
        """加载级联分类器模型"""
        detector = cv2.CascadeClassifier(self.model_path)
        return detector
    
    def detect(self, image):
        """
        检测图像中的动漫人脸
        
        Args:
            image: 输入图像 (OpenCV格式)
        
        Returns:
            检测结果列表，每个结果为 [x, y, width, height, confidence]
        """
        # 转换为灰度图像
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 执行检测
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # 如果没有检测到人脸，返回空列表
        if len(faces) == 0:
            return []
        
        # 转换结果格式，由于级联分类器没有置信度，我们使用1.0作为默认值
        results = []
        for (x, y, w, h) in faces:
            results.append([int(x), int(y), int(w), int(h), 1.0])
        
        return results


def create_detector(detector_type="yunet", **kwargs):
    """
    创建人脸检测器
    
    Args:
        detector_type: 检测器类型，"yunet"或"anime"
        **kwargs: 传递给检测器的参数
    
    Returns:
        人脸检测器实例
    """
    if detector_type.lower() == "yunet":
        return YuNetFaceDetector(**kwargs)
    elif detector_type.lower() == "anime":
        return AnimeFaceDetector(**kwargs)
    else:
        raise ValueError(f"不支持的检测器类型: {detector_type}")
