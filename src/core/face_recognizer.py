"""
人脸识别模块，提供多种人脸特征提取和识别方法
"""
import os
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging
import urllib.request
import importlib.util
import tempfile
import zipfile
import shutil

from src.utils.paths import get_models_dir, get_insightface_models_dir

logger = logging.getLogger(__name__)

class FaceRecognizer:
    """人脸识别器基类"""

    def __init__(self, model_path=None, model_name=None, confidence_threshold=0.6, **kwargs):
        """
        初始化人脸识别器

        Args:
            model_path: 模型文件路径，如果为None则使用默认路径
            model_name: 模型名称，用于某些特定的识别器
            confidence_threshold: 置信度阈值
            **kwargs: 其他参数
        """
        self.model_path = model_path
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.model = None

    def _get_model_path(self, default_filename, default_url):
        """
        获取模型路径，如果未指定则使用默认路径

        Args:
            default_filename: 默认模型文件名
            default_url: 默认模型下载URL

        Returns:
            模型路径
        """
        if self.model_path:
            return self.model_path

        # 默认模型路径
        models_dir = get_models_dir()
        default_path = os.path.join(models_dir, default_filename)

        # 如果模型不存在，尝试下载
        if not os.path.exists(default_path):
            self._download_model(default_path, default_url)

        return default_path

    def _download_model(self, save_path, url):
        """
        下载模型

        Args:
            save_path: 保存路径
            url: 下载URL
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        logger.info(f"正在下载模型: {url}")

        try:
            urllib.request.urlretrieve(url, save_path)
            logger.info(f"模型已下载到: {save_path}")
        except Exception as e:
            logger.error(f"模型下载失败: {e}")
            logger.info("请手动下载模型并放置到models目录")
            raise

    def _load_model(self):
        """加载模型，子类必须实现此方法"""
        raise NotImplementedError("子类必须实现此方法")

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
            f1 = np.array(feature1, dtype=np.float32)
            f2 = np.array(feature2, dtype=np.float32)

            # 检查维度是否匹配，如果不匹配则无法比较
            if f1.size != f2.size:
                logger.warning(f"特征向量维度不匹配: {f1.size} vs {f2.size}，无法计算相似度")
                return 0.0

            # 确保是二维数组，便于使用sklearn的相似度计算
            f1 = f1.reshape(1, -1)
            f2 = f2.reshape(1, -1)

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

    def __init__(self, model_path=None, model_name=None, confidence_threshold=0.6, **kwargs):
        """
        初始化OpenCV人脸识别器

        Args:
            model_path: 模型文件路径，如果为None则使用默认路径
            model_name: 模型名称（OpenCV识别器不使用此参数，但保留以统一接口）
            confidence_threshold: 置信度阈值
            **kwargs: 其他参数
        """
        super().__init__(model_path, model_name, confidence_threshold, **kwargs)

        # 默认模型文件名和URL
        default_filename = "face_recognition_sface_2021dec.onnx"
        default_url = "https://github.com/opencv/opencv_zoo/raw/master/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

        # 获取模型路径
        self.model_path = self._get_model_path(default_filename, default_url)

        # 加载模型
        self.model = self._load_model()

    def _load_model(self):
        """加载OpenCV人脸识别模型"""
        try:
            recognizer = cv2.FaceRecognizerSF.create(
                model=self.model_path,
                config="",
                backend_id=cv2.dnn.DNN_BACKEND_DEFAULT,
                target_id=cv2.dnn.DNN_TARGET_CPU
            )
            logger.info("OpenCV人脸识别模型加载成功")
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
            if self.model is None:
                logger.error("OpenCV人脸识别模型未加载")
                return None

            # 调整大小为模型所需的输入尺寸
            face_img = cv2.resize(face_img, (112, 112))

            # 提取特征向量
            feature = self.model.feature(face_img).flatten().tolist()
            # 确保返回Python原生列表，而不是NumPy数组
            if isinstance(feature, np.ndarray):
                feature = feature.tolist()
            return feature
        except Exception as e:
            logger.error(f"提取人脸特征失败: {e}")
            return None


class InsightFaceRecognizer(FaceRecognizer):
    """使用InsightFace的人脸识别器"""

    def __init__(self, model_path=None, model_name="buffalo_l", confidence_threshold=0.6, det_size=(640, 640), **kwargs):
        """
        初始化InsightFace人脸识别器

        Args:
            model_path: 模型文件路径（InsightFace不使用此参数，但保留以统一接口）
            model_name: 模型名称，可选"buffalo_l", "buffalo_m", "buffalo_s"
            confidence_threshold: 置信度阈值
            det_size: 检测尺寸
            **kwargs: 其他参数
        """
        super().__init__(model_path, model_name, confidence_threshold, **kwargs)
        self.det_size = det_size
        self.model = self._load_model()

    def compute_similarity(self, feature1, feature2, method="cosine"):
        """
        计算两个特征向量的相似度，针对InsightFace进行优化

        Args:
            feature1: 特征向量1
            feature2: 特征向量2
            method: 相似度计算方法，"cosine"或"euclidean"

        Returns:
            相似度 (0-1)
        """
        try:
            # 调用父类方法计算原始相似度
            raw_similarity = super().compute_similarity(feature1, feature2, method)

            # InsightFace的余弦相似度通常很高，即使是不同的人脸
            # 使用非线性变换来增强差异
            if method == "cosine":
                # 对于余弦相似度，使用指数变换来增强差异
                # 这会使得高相似度(>0.9)的差异更加明显
                # 例如：0.95 -> 0.90, 0.98 -> 0.96, 0.99 -> 0.98
                enhanced_similarity = 1.0 - np.sqrt(1.0 - raw_similarity)

                # 记录原始相似度和增强后的相似度
                logger.debug(f"InsightFace原始相似度: {raw_similarity:.4f}, 增强后: {enhanced_similarity:.4f}")

                return float(enhanced_similarity)
            else:
                # 对于其他方法，保持原样
                return raw_similarity
        except Exception as e:
            logger.error(f"计算InsightFace特征相似度失败: {e}")
            return super().compute_similarity(feature1, feature2, method)

    def _load_model(self):
        """加载InsightFace模型"""
        try:
            # 动态导入InsightFace，避免在未使用时强制要求安装
            import importlib
            insightface_spec = importlib.util.find_spec("insightface")
            if insightface_spec is None:
                logger.error("未安装InsightFace库，请使用pip install insightface安装")
                return None

            import insightface
            from insightface.app import FaceAnalysis

            # 检查并确保模型目录存在
            model_dir = os.path.join(get_insightface_models_dir(), self.model_name)
            os.makedirs(model_dir, exist_ok=True)

            # 检查是否有任何模型文件
            has_models = False
            if os.path.exists(model_dir):
                onnx_files = [f for f in os.listdir(model_dir) if f.endswith('.onnx')]
                has_models = len(onnx_files) > 0

            # 如果没有模型文件，尝试下载
            if not has_models:
                logger.info(f"模型目录中没有找到ONNX文件，尝试下载模型")
                self._download_insightface_model(self.model_name)

            # 初始化FaceAnalysis
            try:
                # 尝试使用CPU执行提供程序
                analyzer = FaceAnalysis(name=self.model_name, providers=['CPUExecutionProvider'])
                analyzer.prepare(ctx_id=0, det_size=self.det_size)
                logger.info(f"InsightFace {self.model_name} 模型加载成功")
                return analyzer
            except Exception as e:
                logger.warning(f"使用CPUExecutionProvider加载模型失败: {e}")

                # 尝试不指定提供程序
                try:
                    analyzer = FaceAnalysis(name=self.model_name)
                    analyzer.prepare(ctx_id=0, det_size=self.det_size)
                    logger.info(f"InsightFace {self.model_name} 模型加载成功(默认提供程序)")
                    return analyzer
                except Exception as e2:
                    logger.error(f"使用默认提供程序加载模型也失败: {e2}")
                    raise
        except ImportError:
            logger.error("未安装InsightFace库，请使用pip install insightface安装")
            return None
        except Exception as e:
            logger.error(f"加载InsightFace模型失败: {e}")
            return None

    def _download_insightface_model(self, model_name):
        """
        下载InsightFace模型

        Args:
            model_name: 模型名称，如"buffalo_l", "buffalo_m", "buffalo_s"
        """
        try:
            logger.info(f"尝试下载InsightFace {model_name}模型...")

            # 获取模型目录
            model_dir = os.path.join(get_insightface_models_dir(), model_name)
            os.makedirs(model_dir, exist_ok=True)

            # 模型文件路径
            model_file = os.path.join(model_dir, 'w600k_mbf.onnx')

            # 尝试使用InsightFace的内置下载功能
            try:
                import insightface
                from insightface.model_zoo import get_model
                from insightface.utils import download_onnx

                logger.info("尝试使用InsightFace内置功能下载模型...")
                download_onnx(model_name, model_dir)
                if os.path.exists(model_file):
                    logger.info(f"模型下载成功: {model_file}")
                    return
            except Exception as e:
                logger.warning(f"使用InsightFace内置下载功能失败: {e}")

            # 尝试直接下载模型文件
            try:
                # 根据模型名称选择URL
                if model_name == "buffalo_l":
                    url = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
                elif model_name == "buffalo_m":
                    url = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_m.zip"
                elif model_name == "buffalo_s":
                    url = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip"
                else:
                    raise ValueError(f"不支持的模型名称: {model_name}")

                logger.info(f"尝试从 {url} 下载模型...")

                # 下载并解压模型
                import tempfile
                import zipfile
                import urllib.request

                # 创建临时文件
                with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_file:
                    temp_path = temp_file.name

                try:
                    # 下载模型
                    urllib.request.urlretrieve(url, temp_path)

                    # 解压模型
                    logger.info(f"正在解压模型到 {model_dir}...")
                    with zipfile.ZipFile(temp_path, 'r') as zip_ref:
                        # 创建临时目录用于解压
                        temp_extract_dir = tempfile.mkdtemp()
                        try:
                            # 先解压到临时目录
                            zip_ref.extractall(temp_extract_dir)

                            # 查找解压后的模型文件
                            for root, _, files in os.walk(temp_extract_dir):
                                for file in files:
                                    if file.endswith('.onnx'):
                                        # 找到onnx文件，复制到目标目录
                                        src_file = os.path.join(root, file)
                                        dst_file = os.path.join(model_dir, file)
                                        logger.info(f"复制模型文件: {src_file} -> {dst_file}")
                                        shutil.copy2(src_file, dst_file)
                        finally:
                            # 清理临时目录
                            shutil.rmtree(temp_extract_dir, ignore_errors=True)

                    if os.path.exists(model_file):
                        logger.info(f"模型下载并解压成功: {model_file}")
                        return
                    else:
                        logger.warning(f"解压后未找到模型文件: {model_file}")
                finally:
                    # 删除临时文件
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
            except Exception as e:
                logger.warning(f"直接下载模型失败: {e}")

            # 如果所有自动下载方法都失败，提供手动下载指南
            manual_guide = f"""
            ======================================================
            自动下载InsightFace模型失败，请按照以下步骤手动下载：

            1. 访问 https://github.com/deepinsight/insightface/releases/tag/v0.7
            2. 下载 {model_name}.zip 文件
            3. 解压文件到 {os.path.dirname(model_dir)} 目录
               确保 {model_file} 文件存在

            或者，您可以尝试使用pip安装最新版本的InsightFace:
            pip install --upgrade insightface

            然后运行以下Python代码来下载模型:
            ```python
            import insightface
            from insightface.app import FaceAnalysis
            app = FaceAnalysis(name='{model_name}')
            app.prepare(ctx_id=0)
            ```
            ======================================================
            """
            logger.error(manual_guide)
            raise RuntimeError(f"无法自动下载InsightFace模型，请参考上述指南手动下载")

        except Exception as e:
            logger.error(f"下载InsightFace模型失败: {e}")
            raise

    def extract_feature(self, face_img):
        """
        提取人脸特征向量

        Args:
            face_img: 人脸图像

        Returns:
            特征向量
        """
        try:
            if self.model is None:
                logger.error("InsightFace模型未加载")
                return None

            # 检测人脸并提取特征
            faces = self.model.get(face_img)

            if len(faces) == 0:
                logger.warning("未检测到人脸，尝试直接使用人脸对齐和特征提取")
                try:
                    # 直接使用InsightFace的特征提取模型
                    # 调整大小为模型所需的输入尺寸并确保数据类型正确
                    face_img = cv2.resize(face_img, (112, 112))
                    # 确保图像是float32类型，并进行归一化
                    face_img = face_img.astype(np.float32)
                    # 归一化到[-1,1]范围
                    face_img = (face_img - 127.5) / 127.5
                    # 添加批次维度 [B,C,H,W]
                    face_img = np.transpose(face_img, (2, 0, 1))  # [C,H,W]
                    face_img = np.expand_dims(face_img, axis=0)  # [1,C,H,W]

                    # 获取特征提取模型
                    import insightface
                    from insightface.model_zoo import get_model as ins_get_model

                    # 检查是否已有特征提取模型
                    if not hasattr(self, 'feature_model'):
                        # 尝试加载特征提取模型
                        # 首先尝试w600k_mbf.onnx (旧版本)
                        model_path = os.path.join(get_insightface_models_dir(),
                                                 self.model_name, 'w600k_mbf.onnx')
                        # 如果不存在，尝试w600k_r50.onnx (新版本)
                        if not os.path.exists(model_path):
                            alt_model_path = os.path.join(get_insightface_models_dir(),
                                                        self.model_name, 'w600k_r50.onnx')
                            if os.path.exists(alt_model_path):
                                model_path = alt_model_path
                                logger.info(f"使用替代模型: {model_path}")

                        if os.path.exists(model_path):
                            try:
                                self.feature_model = ins_get_model(model_path)
                                self.feature_model.prepare(ctx_id=0)
                                logger.info(f"加载InsightFace特征提取模型: {model_path}")
                            except Exception as e:
                                logger.error(f"加载特征提取模型失败: {e}")
                                return None
                        else:
                            logger.warning(f"特征提取模型不存在: {model_path}，尝试下载")
                            try:
                                # 尝试下载模型
                                self._download_insightface_model(self.model_name)

                                # 检查模型目录是否存在任何ONNX文件
                                model_dir = os.path.join(get_insightface_models_dir(), self.model_name)

                                # 尝试查找可用的模型文件
                                model_found = False

                                # 首先尝试特定的已知模型文件
                                known_models = [
                                    'w600k_mbf.onnx',  # 旧版本
                                    'w600k_r50.onnx',  # 新版本
                                    'recognition.onnx',  # 可能的通用名称
                                ]

                                for model_filename in known_models:
                                    model_path = os.path.join(model_dir, model_filename)
                                    if os.path.exists(model_path):
                                        logger.info(f"找到模型文件: {model_path}")
                                        try:
                                            self.feature_model = ins_get_model(model_path)
                                            self.feature_model.prepare(ctx_id=0)
                                            logger.info(f"加载InsightFace特征提取模型: {model_path}")
                                            model_found = True
                                            break
                                        except Exception as e:
                                            logger.warning(f"加载模型 {model_filename} 失败: {e}")

                                # 如果没有找到已知模型，尝试目录中的任何ONNX文件
                                if not model_found and os.path.exists(model_dir):
                                    onnx_files = [f for f in os.listdir(model_dir) if f.endswith('.onnx')]
                                    for onnx_file in onnx_files:
                                        if onnx_file not in known_models:  # 避免重复尝试
                                            model_path = os.path.join(model_dir, onnx_file)
                                            logger.info(f"尝试加载未知模型文件: {model_path}")
                                            try:
                                                self.feature_model = ins_get_model(model_path)
                                                self.feature_model.prepare(ctx_id=0)
                                                logger.info(f"成功加载模型: {model_path}")
                                                model_found = True
                                                break
                                            except Exception as e:
                                                logger.warning(f"加载模型 {onnx_file} 失败: {e}")

                                # 如果仍然没有找到可用模型，提供手动下载指南
                                if not model_found:
                                    if os.path.exists(model_dir):
                                        onnx_files = [f for f in os.listdir(model_dir) if f.endswith('.onnx')]
                                        if onnx_files:
                                            logger.warning(f"目录中存在以下ONNX文件，但无法加载: {onnx_files}")

                                    # 提供手动下载指南
                                    manual_guide = f"""
                                    ======================================================
                                    无法找到或加载InsightFace特征提取模型

                                    请按照以下步骤手动下载并安装模型:

                                    1. 访问 https://github.com/deepinsight/insightface/releases/tag/v0.7
                                    2. 下载 {self.model_name}.zip 文件
                                    3. 解压文件到 {os.path.dirname(model_dir)} 目录

                                    或者，您可以尝试使用pip安装最新版本的InsightFace:
                                    pip install --upgrade insightface

                                    然后运行以下Python代码来下载模型:
                                    ```python
                                    import insightface
                                    from insightface.app import FaceAnalysis
                                    app = FaceAnalysis(name='{self.model_name}')
                                    app.prepare(ctx_id=0)
                                    ```

                                    或者，您可以切换到OpenCV人脸识别模型:
                                    在应用程序设置中，将"人脸识别模型"从"InsightFace"切换为"OpenCV"
                                    ======================================================
                                    """
                                    logger.error(manual_guide)
                                    return None
                            except Exception as e:
                                logger.error(f"下载特征提取模型失败: {e}")
                                return None

                    # 提取特征 - ArcFaceONNX模型使用forward方法
                    if hasattr(self.feature_model, 'get_embedding'):
                        feature = self.feature_model.get_embedding(face_img)
                    elif hasattr(self.feature_model, 'forward'):
                        feature = self.feature_model.forward(face_img)
                    elif hasattr(self.feature_model, 'get_feat'):
                        feature = self.feature_model.get_feat(face_img)
                    else:
                        logger.error("无法找到特征提取方法")
                        return None

                    # 如果特征是多维数组，将其展平为一维向量
                    if isinstance(feature, np.ndarray) and feature.ndim > 1:
                        feature = feature.flatten()

                    # 确保返回Python原生列表，而不是NumPy数组，以便JSON序列化
                    if isinstance(feature, np.ndarray):
                        feature = feature.tolist()

                    return feature
                except Exception as e:
                    logger.error(f"直接特征提取失败: {e}")
                    return None

            # 使用最大的人脸
            face = max(faces, key=lambda x: x.bbox[2] * x.bbox[3])

            # 获取特征向量
            feature = face.normed_embedding

            # 确保返回Python原生列表，而不是NumPy数组
            if isinstance(feature, np.ndarray):
                feature = feature.tolist()

            return feature
        except Exception as e:
            logger.error(f"提取人脸特征失败: {e}")
            return None


def create_face_recognizer(recognizer_type="insightface", **kwargs):
    """
    创建人脸识别器

    Args:
        recognizer_type: 识别器类型，"insightface"或"opencv"
        **kwargs: 传递给识别器的参数，可包含：
            - model_path: 模型文件路径
            - model_name: 模型名称（主要用于InsightFace）
            - confidence_threshold: 置信度阈值
            - det_size: 检测尺寸（仅用于InsightFace）

    Returns:
        人脸识别器实例
    """
    # 统一参数处理，确保所有必要参数都存在
    recognizer_type = recognizer_type.lower()

    try:
        if recognizer_type == "insightface":
            return InsightFaceRecognizer(**kwargs)
        elif recognizer_type == "opencv":
            return OpenCVFaceRecognizer(**kwargs)
        else:
            logger.warning(f"不支持的人脸识别器类型: {recognizer_type}，使用InsightFace")
            return InsightFaceRecognizer(**kwargs)
    except Exception as e:
        logger.error(f"创建人脸识别器失败: {e}")
        # 如果创建失败，尝试使用另一个识别器
        try:
            if recognizer_type == "insightface":
                logger.info("尝试使用OpenCV人脸识别器作为备选")
                return OpenCVFaceRecognizer(**kwargs)
            else:
                logger.info("尝试使用InsightFace人脸识别器作为备选")
                return InsightFaceRecognizer(**kwargs)
        except Exception as e2:
            logger.error(f"创建备选人脸识别器也失败: {e2}")
            return None
