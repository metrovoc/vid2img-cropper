"""
人脸识别模块，提供多种人脸特征提取和识别方法
"""
import os
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

class FaceRecognizer:
    """人脸识别器基类"""
    
    def __init__(self, confidence_threshold=0.6):
        """
        初始化人脸识别器
        
        Args:
            confidence_threshold: 置信度阈值
        """
        self.confidence_threshold = confidence_threshold
        
    def extract_feature(self, face_img):
        """
        提取人脸特征向量
        
        Args:
            face_img: 人脸图像
            
        Returns:
            特征向量
        """
        raise NotImplementedError("子类必须实现此方法")
    
    def compute_similarity(self, feature1, feature2, method="cosine"):
        """
        计算两个特征向量的相似度
        
        Args:
            feature1: 特征向量1
            feature2: 特征向量2
            method: 相似度计算方法，"cosine"或"euclidean"
            
        Returns:
            相似度 (0-1)
        """
        try:
            # 转换为numpy数组
            f1 = np.array(feature1).reshape(1, -1)
            f2 = np.array(feature2).reshape(1, -1)
            
            if method == "cosine":
                # 计算余弦相似度
                similarity = cosine_similarity(f1, f2)[0][0]
                return float(similarity)
            elif method == "euclidean":
                # 计算欧氏距离并转换为相似度
                distance = np.linalg.norm(f1 - f2)
                # 将距离转换为相似度 (0-1)
                similarity = 1.0 / (1.0 + distance)
                return float(similarity)
            else:
                logger.warning(f"不支持的相似度计算方法: {method}，使用余弦相似度")
                similarity = cosine_similarity(f1, f2)[0][0]
                return float(similarity)
        except Exception as e:
            logger.error(f"计算人脸特征相似度失败: {e}")
            return 0.0


class OpenCVFaceRecognizer(FaceRecognizer):
    """使用OpenCV的人脸识别器"""
    
    def __init__(self, model_path=None, confidence_threshold=0.6):
        """
        初始化OpenCV人脸识别器
        
        Args:
            model_path: 模型文件路径，如果为None则使用默认路径
            confidence_threshold: 置信度阈值
        """
        super().__init__(confidence_threshold)
        self.model_path = self._get_model_path(model_path)
        self.recognizer = self._load_model()
        
    def _get_model_path(self, model_path):
        """获取模型路径，如果未指定则使用默认路径"""
        if model_path:
            return model_path
            
        # 默认模型路径
        default_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                    "models", "face_recognition_sface_2021dec.onnx")
                                    
        # 如果模型不存在，尝试下载
        if not os.path.exists(default_path):
            self._download_model(default_path)
            
        return default_path
        
    def _download_model(self, save_path):
        """下载OpenCV人脸识别模型"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        logger.info("正在下载OpenCV人脸识别模型...")
        model_url = "https://github.com/opencv/opencv_zoo/raw/master/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
        
        try:
            import urllib.request
            urllib.request.urlretrieve(model_url, save_path)
            logger.info(f"模型已下载到: {save_path}")
        except Exception as e:
            logger.error(f"模型下载失败: {e}")
            logger.info("请手动下载模型并放置到models目录")
            raise
            
    def _load_model(self):
        """加载OpenCV人脸识别模型"""
        try:
            recognizer = cv2.FaceRecognizerSF.create(
                model=self.model_path,
                config="",
                backend_id=cv2.dnn.DNN_BACKEND_DEFAULT,
                target_id=cv2.dnn.DNN_TARGET_CPU
            )
            return recognizer
        except Exception as e:
            logger.error(f"加载OpenCV人脸识别模型失败: {e}")
            return None
            
    def extract_feature(self, face_img):
        """
        提取人脸特征向量
        
        Args:
            face_img: 人脸图像
            
        Returns:
            特征向量
        """
        try:
            if self.recognizer is None:
                logger.error("人脸识别模型未加载")
                return None
                
            # 调整大小为模型所需的输入尺寸
            face_img = cv2.resize(face_img, (112, 112))
            
            # 提取特征向量
            feature = self.recognizer.feature(face_img).flatten().tolist()
            return feature
        except Exception as e:
            logger.error(f"提取人脸特征失败: {e}")
            return None


class InsightFaceRecognizer(FaceRecognizer):
    """使用InsightFace的人脸识别器"""
    
    def __init__(self, model_name="buffalo_l", confidence_threshold=0.6, det_size=(640, 640)):
        """
        初始化InsightFace人脸识别器
        
        Args:
            model_name: 模型名称，可选"buffalo_l", "buffalo_m", "buffalo_s"
            confidence_threshold: 置信度阈值
            det_size: 检测尺寸
        """
        super().__init__(confidence_threshold)
        self.model_name = model_name
        self.det_size = det_size
        self.analyzer = self._load_model()
        
    def _load_model(self):
        """加载InsightFace模型"""
        try:
            import insightface
            from insightface.app import FaceAnalysis
            
            # 初始化FaceAnalysis
            analyzer = FaceAnalysis(name=self.model_name, providers=['CPUExecutionProvider'])
            analyzer.prepare(ctx_id=0, det_size=self.det_size)
            
            logger.info(f"InsightFace {self.model_name} 模型加载成功")
            return analyzer
        except ImportError:
            logger.error("未安装InsightFace库，请使用pip install insightface安装")
            return None
        except Exception as e:
            logger.error(f"加载InsightFace模型失败: {e}")
            return None
            
    def extract_feature(self, face_img):
        """
        提取人脸特征向量
        
        Args:
            face_img: 人脸图像
            
        Returns:
            特征向量
        """
        try:
            if self.analyzer is None:
                logger.error("InsightFace模型未加载")
                return None
                
            # 检测人脸并提取特征
            faces = self.analyzer.get(face_img)
            
            if len(faces) == 0:
                logger.warning("未检测到人脸")
                return None
                
            # 使用最大的人脸
            face = max(faces, key=lambda x: x.bbox[2] * x.bbox[3])
            
            # 返回特征向量
            return face.normed_embedding
        except Exception as e:
            logger.error(f"提取人脸特征失败: {e}")
            return None


def create_face_recognizer(recognizer_type="insightface", **kwargs):
    """
    创建人脸识别器
    
    Args:
        recognizer_type: 识别器类型，"insightface"或"opencv"
        **kwargs: 传递给识别器的参数
        
    Returns:
        人脸识别器实例
    """
    if recognizer_type.lower() == "insightface":
        return InsightFaceRecognizer(**kwargs)
    elif recognizer_type.lower() == "opencv":
        return OpenCVFaceRecognizer(**kwargs)
    else:
        logger.warning(f"不支持的人脸识别器类型: {recognizer_type}，使用InsightFace")
        return InsightFaceRecognizer(**kwargs)
