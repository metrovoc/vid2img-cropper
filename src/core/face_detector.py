"""
人脸检测模块，负责检测图像中的人脸
"""
import os
import cv2
import numpy as np
from pathlib import Path
import urllib.request
import shutil
import importlib.util
import logging

from src.utils.paths import get_models_dir

# 配置日志
logger = logging.getLogger(__name__)


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

    def __init__(self, model_path=None, confidence_threshold=0.6, input_size=(320, 320), use_gpu=False):
        """
        初始化YuNet人脸检测器

        Args:
            model_path: 模型文件路径，如果为None则尝试下载
            confidence_threshold: 置信度阈值
            input_size: 模型输入大小
            use_gpu: 是否使用GPU加速
        """
        super().__init__(confidence_threshold)
        self.input_size = input_size
        self.use_gpu = use_gpu
        self.model_path = self._get_model_path(model_path)
        self.detector = self._load_model()

    def _get_model_path(self, model_path):
        """获取模型路径，如果未指定则使用默认路径"""
        if model_path:
            return model_path

        # 默认模型路径
        models_dir = get_models_dir()
        default_path = os.path.join(models_dir, "face_detection_yunet_2023mar.onnx")

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
        # 根据是否使用GPU选择后端和目标
        backend_id = cv2.dnn.DNN_BACKEND_DEFAULT
        target_id = cv2.dnn.DNN_TARGET_CPU

        if self.use_gpu:
            # 检查CUDA是否可用
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                try:
                    # 尝试使用CUDA后端
                    backend_id = cv2.dnn.DNN_BACKEND_CUDA
                    target_id = cv2.dnn.DNN_TARGET_CUDA
                    logger.info("YuNet检测器使用CUDA加速")
                except Exception as e:
                    logger.warning(f"无法使用CUDA加速YuNet检测器: {e}")
                    logger.info("回退到CPU模式")
            else:
                logger.warning("未检测到CUDA设备，YuNet检测器使用CPU模式")

        try:
            detector = cv2.FaceDetectorYN.create(
                model=self.model_path,
                config="",
                input_size=self.input_size,
                score_threshold=self.confidence_threshold,
                backend_id=backend_id,
                target_id=target_id
            )
            return detector
        except Exception as e:
            logger.error(f"加载YuNet模型失败: {e}")
            # 如果使用GPU失败，尝试回退到CPU
            if self.use_gpu:
                logger.info("尝试使用CPU模式加载YuNet模型")
                try:
                    detector = cv2.FaceDetectorYN.create(
                        model=self.model_path,
                        config="",
                        input_size=self.input_size,
                        score_threshold=self.confidence_threshold,
                        backend_id=cv2.dnn.DNN_BACKEND_DEFAULT,
                        target_id=cv2.dnn.DNN_TARGET_CPU
                    )
                    return detector
                except Exception as e2:
                    logger.error(f"使用CPU模式加载YuNet模型也失败: {e2}")
                    return None
            return None

    def detect(self, image):
        """
        检测图像中的人脸

        Args:
            image: 输入图像 (OpenCV格式)

        Returns:
            检测结果列表，每个结果为 [x, y, width, height, confidence]
        """
        # 检查检测器是否成功加载
        if self.detector is None:
            logger.error("YuNet检测器未成功加载，无法执行人脸检测")
            return []

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

    def __init__(self, model_path=None, confidence_threshold=0.6, use_gpu=False):
        """
        初始化动漫人脸检测器

        Args:
            model_path: 级联分类器XML文件路径，如果为None则尝试下载
            confidence_threshold: 置信度阈值
            use_gpu: 是否使用GPU加速（级联分类器不支持GPU加速，此参数仅为统一接口）
        """
        super().__init__(confidence_threshold)
        self.use_gpu = use_gpu
        self.model_path = self._get_model_path(model_path)
        self.detector = self._load_model()

        if self.use_gpu:
            logger.warning("级联分类器不支持GPU加速，将使用CPU模式")

    def _get_model_path(self, model_path):
        """获取模型路径，如果未指定则使用默认路径"""
        if model_path:
            return model_path

        # 默认模型路径
        models_dir = get_models_dir()
        default_path = os.path.join(models_dir, "lbpcascade_animeface.xml")

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


class YOLOv8FaceDetector(FaceDetector):
    """使用YOLOv8模型的人脸检测器"""

    def __init__(self, model_path=None, confidence_threshold=0.6, input_size=(640, 640), use_gpu=False):
        """
        初始化YOLOv8人脸检测器

        Args:
            model_path: 模型文件路径，如果为None则尝试下载
            confidence_threshold: 置信度阈值
            input_size: 模型输入大小
            use_gpu: 是否使用GPU加速
        """
        super().__init__(confidence_threshold)
        self.input_size = input_size
        self.use_gpu = use_gpu
        self.model_path = self._get_model_path(model_path)
        self.detector = self._load_model()

    def _get_model_path(self, model_path):
        """获取模型路径，如果未指定则使用默认路径"""
        if model_path:
            return model_path

        # 默认模型路径
        models_dir = get_models_dir()
        default_path = os.path.join(models_dir, "yolov8n-face.pt")

        # 如果模型不存在，尝试下载
        if not os.path.exists(default_path):
            self._download_model(default_path)

        return default_path

    def _download_model(self, save_path):
        """下载YOLOv8模型"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        print("正在下载YOLOv8人脸检测模型...")
        model_url = "https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt"

        try:
            urllib.request.urlretrieve(model_url, save_path)
            print(f"模型已下载到: {save_path}")
        except Exception as e:
            print(f"模型下载失败: {e}")
            print("请手动下载模型并放置到models目录")
            raise

    def _load_model(self):
        """加载YOLOv8模型"""
        try:
            # 检查是否安装了ultralytics
            if importlib.util.find_spec("ultralytics") is None:
                logger.error("未安装ultralytics库，请使用pip install ultralytics安装")
                return None

            from ultralytics import YOLO

            # 设置设备
            device = 'cpu'
            if self.use_gpu:
                try:
                    import torch
                    if torch.cuda.is_available():
                        device = 0  # 使用第一个GPU
                        logger.info("YOLOv8检测器使用CUDA加速")
                    else:
                        logger.warning("未检测到CUDA设备，YOLOv8检测器使用CPU模式")
                except ImportError:
                    logger.warning("未安装PyTorch或CUDA不可用，YOLOv8检测器使用CPU模式")

            # 加载模型
            model = YOLO(self.model_path)

            # 设置设备
            if device != 'cpu':
                try:
                    model.to(device)
                except Exception as e:
                    logger.warning(f"将YOLOv8模型移至GPU失败: {e}，使用CPU模式")

            return model
        except Exception as e:
            logger.error(f"加载YOLOv8模型失败: {e}")
            return None

    def detect(self, image):
        """
        检测图像中的人脸

        Args:
            image: 输入图像 (OpenCV格式)

        Returns:
            检测结果列表，每个结果为 [x, y, width, height, confidence]
        """
        if self.detector is None:
            return []

        # 执行检测
        results = self.detector(image, conf=self.confidence_threshold)

        # 转换结果格式
        face_results = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # 获取边界框坐标
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                # 计算宽度和高度
                w = x2 - x1
                h = y2 - y1
                # 获取置信度
                conf = box.conf[0].cpu().numpy()

                face_results.append([int(x1), int(y1), int(w), int(h), float(conf)])

        return face_results


class SCRFDFaceDetector(FaceDetector):
    """使用SCRFD模型的人脸检测器"""

    def __init__(self, model_path=None, confidence_threshold=0.6, input_size=(640, 640), use_gpu=False):
        """
        初始化SCRFD人脸检测器

        Args:
            model_path: 模型文件路径，如果为None则尝试下载
            confidence_threshold: 置信度阈值
            input_size: 模型输入大小
            use_gpu: 是否使用GPU加速
        """
        super().__init__(confidence_threshold)
        self.input_size = input_size
        self.use_gpu = use_gpu
        self.model_path = self._get_model_path(model_path)
        self.detector = self._load_model()

    def _get_model_path(self, model_path):
        """获取模型路径，如果未指定则使用默认路径"""
        if model_path:
            return model_path

        # 默认模型路径
        models_dir = get_models_dir()
        default_path = os.path.join(models_dir, "scrfd_10g_bnkps.onnx")

        # 如果模型不存在，尝试下载
        if not os.path.exists(default_path):
            self._download_model(default_path)

        return default_path

    def _download_model(self, save_path):
        """下载SCRFD模型"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        print("正在下载SCRFD人脸检测模型...")
        model_url = "https://huggingface.co/MonsterMMORPG/tools/resolve/main/scrfd_10g_bnkps.onnx"

        try:
            urllib.request.urlretrieve(model_url, save_path)
            print(f"模型已下载到: {save_path}")
        except Exception as e:
            print(f"模型下载失败: {e}")
            print("请手动下载模型并放置到models目录")
            raise

    def _load_model(self):
        """加载SCRFD模型"""
        try:
            # 检查是否安装了onnxruntime
            if importlib.util.find_spec("onnxruntime") is None:
                logger.error("未安装onnxruntime库，请使用pip install onnxruntime安装")
                return None

            import onnxruntime

            # 根据是否使用GPU选择提供程序
            providers = ['CPUExecutionProvider']

            if self.use_gpu:
                # 检查是否有可用的GPU
                try:
                    if 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
                        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                        logger.info("SCRFD检测器使用CUDA加速")
                    else:
                        logger.warning("未检测到CUDA提供程序，SCRFD检测器使用CPU模式")
                except Exception as e:
                    logger.warning(f"检查CUDA提供程序时出错: {e}，SCRFD检测器使用CPU模式")

            # 创建会话
            try:
                session = onnxruntime.InferenceSession(self.model_path, providers=providers)
                return session
            except Exception as e:
                logger.error(f"使用提供程序 {providers} 加载SCRFD模型失败: {e}")

                # 如果使用GPU失败，尝试回退到CPU
                if self.use_gpu and 'CUDAExecutionProvider' in providers:
                    logger.info("尝试使用CPU模式加载SCRFD模型")
                    try:
                        session = onnxruntime.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
                        return session
                    except Exception as e2:
                        logger.error(f"使用CPU模式加载SCRFD模型也失败: {e2}")

                return None
        except Exception as e:
            logger.error(f"加载SCRFD模型失败: {e}")
            return None

    def detect(self, image):
        """
        检测图像中的人脸

        Args:
            image: 输入图像 (OpenCV格式)

        Returns:
            检测结果列表，每个结果为 [x, y, width, height, confidence]
        """
        if self.detector is None:
            return []

        # 预处理图像
        height, width, _ = image.shape
        input_height, input_width = self.input_size

        # 调整图像大小
        resized_img = cv2.resize(image, (input_width, input_height))

        # 转换为RGB并归一化
        img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        img_norm = img_rgb.astype(np.float32) / 255.0

        # 转换为NCHW格式
        img_input = np.transpose(img_norm, (2, 0, 1))
        img_input = np.expand_dims(img_input, axis=0)

        # 执行推理
        input_name = self.detector.get_inputs()[0].name
        outputs = self.detector.run(None, {input_name: img_input})

        # 解析结果
        # 注意：SCRFD模型的输出格式可能需要根据具体模型调整
        # 这里假设输出包含边界框、置信度和关键点
        boxes = outputs[0][0]
        scores = outputs[1][0]

        # 筛选高置信度的检测结果
        face_results = []
        for i, score in enumerate(scores):
            if score >= self.confidence_threshold:
                # 获取边界框坐标
                box = boxes[i]
                x1, y1, x2, y2 = box

                # 映射回原始图像尺寸
                x1 = int(x1 * width / input_width)
                y1 = int(y1 * height / input_height)
                x2 = int(x2 * width / input_width)
                y2 = int(y2 * height / input_height)

                # 计算宽度和高度
                w = x2 - x1
                h = y2 - y1

                face_results.append([int(x1), int(y1), int(w), int(h), float(score)])

        return face_results


def create_detector(detector_type="yunet", use_gpu=False, **kwargs):
    """
    创建人脸检测器

    Args:
        detector_type: 检测器类型，"yunet"、"anime"、"yolov8"或"scrfd"
        use_gpu: 是否使用GPU加速
        **kwargs: 传递给检测器的参数

    Returns:
        人脸检测器实例
    """
    # 确保use_gpu参数传递给检测器
    kwargs['use_gpu'] = use_gpu

    if detector_type.lower() == "yunet":
        return YuNetFaceDetector(**kwargs)
    elif detector_type.lower() == "anime":
        return AnimeFaceDetector(**kwargs)
    elif detector_type.lower() == "yolov8":
        return YOLOv8FaceDetector(**kwargs)
    elif detector_type.lower() == "scrfd":
        return SCRFDFaceDetector(**kwargs)
    else:
        raise ValueError(f"不支持的检测器类型: {detector_type}")
