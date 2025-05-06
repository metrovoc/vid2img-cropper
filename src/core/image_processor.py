"""
图片处理模块，负责处理图片文件中的人脸检测和裁剪
"""
import os
import cv2
import time
import logging
import numpy as np
from datetime import timedelta
from PIL import Image
import imagehash
from pathlib import Path

from src.core.face_detector import create_detector
from src.utils.paths import get_models_dir

logger = logging.getLogger(__name__)


class ImageProcessor:
    """图片处理类"""

    def __init__(self, config):
        """
        初始化图片处理器

        Args:
            config: 配置对象
        """
        self.config = config
        self.detector = None
        self.running = False

        # 从配置中读取处理参数
        self.detection_width = config.get("processing", "detection_width", 640)
        self.detector_type = config.get("processing", "detector_type", "yunet")
        self.confidence_threshold = config.get("processing", "confidence_threshold", 0.6)
        self.crop_padding = config.get("processing", "crop_padding", 0.2)
        self.crop_aspect_ratio = config.get("processing", "crop_aspect_ratio", 1.0)
        self.min_face_size = config.get("processing", "min_face_size", 40)
        self.use_gpu = config.get("processing", "use_gpu", False)

        # 输出相关配置
        self.output_format = config.get("output", "format", "jpg")
        self.output_quality = config.get("output", "quality", 95)
        self.output_dir = config.get_image_output_dir()

    def _init_detector(self):
        """初始化人脸检测器"""
        if self.detector is None:
            # 获取模型路径
            models_dir = get_models_dir()

            # 根据检测器类型选择模型文件
            model_path = None
            if self.detector_type.lower() == "yunet":
                model_path = os.path.join(models_dir, "face_detection_yunet_2023mar.onnx")
            elif self.detector_type.lower() == "anime":
                model_path = os.path.join(models_dir, "lbpcascade_animeface.xml")
            elif self.detector_type.lower() == "yolov8":
                model_path = os.path.join(models_dir, "yolov8n-face.pt")
            elif self.detector_type.lower() == "scrfd":
                model_path = os.path.join(models_dir, "scrfd_10g_bnkps.onnx")

            # 创建检测器
            self.detector = create_detector(
                detector_type=self.detector_type,
                model_path=model_path,
                confidence_threshold=self.confidence_threshold,
                use_gpu=self.use_gpu
            )

    def process_image(self, image_path, output_dir=None, progress_callback=None, status_callback=None):
        """
        处理单个图片文件

        Args:
            image_path: 图片文件路径
            output_dir: 输出目录，如果为None则使用默认输出目录
            progress_callback: 进度回调函数
            status_callback: 状态回调函数

        Returns:
            处理结果字典
        """
        self._init_detector()

        self.running = True
        self.processed_images = 0
        self.detected_faces = 0

        # 创建输出目录
        if output_dir is None:
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            output_dir = os.path.join(self.output_dir, image_name)

        os.makedirs(output_dir, exist_ok=True)

        # 读取图片
        try:
            image = cv2.imread(image_path)
            if image is None:
                if status_callback:
                    status_callback(f"无法读取图片文件: {image_path}")
                return {"success": False, "error": "无法读取图片文件"}
        except Exception as e:
            if status_callback:
                status_callback(f"读取图片文件失败: {e}")
            return {"success": False, "error": f"读取图片文件失败: {e}"}

        start_time = time.time()

        # 处理图片
        self._process_image(image, image_path, output_dir)

        self.processed_images += 1

        # 更新进度
        if progress_callback:
            progress_callback(1, 1, 1.0, time.time() - start_time, 0, self.processed_images, self.detected_faces)

        end_time = time.time()
        processing_time = end_time - start_time

        if status_callback:
            status_callback(f"处理完成: 检测到 {self.detected_faces} 个人脸")
            status_callback(f"处理时间: {timedelta(seconds=processing_time)}")

        self.running = False

        return {
            "success": True,
            "image_path": image_path,
            "detected_faces": self.detected_faces,
            "processing_time": processing_time
        }

    def process_images(self, image_paths, output_dir=None, progress_callback=None, status_callback=None):
        """
        批量处理图片文件

        Args:
            image_paths: 图片文件路径列表
            output_dir: 输出目录，如果为None则使用默认输出目录
            progress_callback: 进度回调函数
            status_callback: 状态回调函数

        Returns:
            处理结果字典列表
        """
        self._init_detector()

        self.running = True
        self.processed_images = 0
        self.detected_faces = 0

        # 创建输出目录
        if output_dir is None:
            output_dir = os.path.join(self.output_dir, "batch_" + time.strftime("%Y%m%d_%H%M%S"))

        os.makedirs(output_dir, exist_ok=True)

        results = []
        total_images = len(image_paths)
        start_time = time.time()

        for i, image_path in enumerate(image_paths):
            if not self.running:
                break

            # 更新状态
            if status_callback:
                status_callback(f"处理图片 {i+1}/{total_images}: {os.path.basename(image_path)}")

            # 读取图片
            try:
                image = cv2.imread(image_path)
                if image is None:
                    results.append({
                        "success": False,
                        "image_path": image_path,
                        "error": "无法读取图片文件"
                    })
                    continue
            except Exception as e:
                results.append({
                    "success": False,
                    "image_path": image_path,
                    "error": f"读取图片文件失败: {e}"
                })
                continue

            # 处理图片
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            image_output_dir = os.path.join(output_dir, image_name)
            os.makedirs(image_output_dir, exist_ok=True)

            faces_before = self.detected_faces
            self._process_image(image, image_path, image_output_dir)
            faces_detected = self.detected_faces - faces_before

            self.processed_images += 1

            # 添加结果
            results.append({
                "success": True,
                "image_path": image_path,
                "detected_faces": faces_detected
            })

            # 更新进度
            if progress_callback:
                progress = (i + 1) / total_images
                elapsed = time.time() - start_time
                remaining = (elapsed / (i + 1)) * (total_images - i - 1) if i > 0 else 0

                progress_callback(
                    i + 1, total_images, progress,
                    elapsed, remaining, self.processed_images, self.detected_faces
                )

        end_time = time.time()
        total_time = end_time - start_time

        if status_callback:
            status_callback(f"批处理完成: 处理了 {self.processed_images} 张图片，检测到 {self.detected_faces} 个人脸")
            status_callback(f"总处理时间: {timedelta(seconds=total_time)}")

        self.running = False

        return results

    def _process_image(self, image, image_path, output_dir):
        """
        处理单个图片

        Args:
            image: OpenCV格式的图片
            image_path: 图片文件路径
            output_dir: 输出目录
        """
        # 检查检测器是否已初始化
        if self.detector is None:
            logger.error("人脸检测器未成功初始化，无法处理图片")
            return

        # 调整大小用于检测
        height, width = image.shape[:2]
        scale = self.detection_width / width
        detection_height = int(height * scale)
        detection_image = cv2.resize(image, (self.detection_width, detection_height))

        # 检测人脸
        faces = self.detector.detect(detection_image)

        # 将检测结果映射回原始分辨率
        scale_back = width / self.detection_width

        for i, face in enumerate(faces):
            x, y, w, h, confidence = face

            # 映射回原始分辨率
            x = int(x * scale_back)
            y = int(y * scale_back)
            w = int(w * scale_back)
            h = int(h * scale_back)

            # 检查人脸尺寸是否过小
            if w < self.min_face_size or h < self.min_face_size:
                continue  # 跳过过小的人脸

            # 添加padding
            padding_x = int(w * self.crop_padding)
            padding_y = int(h * self.crop_padding)

            # 计算裁剪区域
            if self.crop_aspect_ratio == 1.0:
                # 正方形裁剪
                crop_size = max(w + 2 * padding_x, h + 2 * padding_y)
                x_center = x + w // 2
                y_center = y + h // 2
                x1 = max(0, x_center - crop_size // 2)
                y1 = max(0, y_center - crop_size // 2)
                x2 = min(width, x_center + crop_size // 2)
                y2 = min(height, y_center + crop_size // 2)
            else:
                # 按指定宽高比裁剪
                crop_width = w + 2 * padding_x
                crop_height = int(crop_width / self.crop_aspect_ratio)

                # 如果计算出的高度小于人脸高度加padding，则重新计算
                if crop_height < h + 2 * padding_y:
                    crop_height = h + 2 * padding_y
                    crop_width = int(crop_height * self.crop_aspect_ratio)

                x_center = x + w // 2
                y_center = y + h // 2
                x1 = max(0, x_center - crop_width // 2)
                y1 = max(0, y_center - crop_height // 2)
                x2 = min(width, x_center + crop_width // 2)
                y2 = min(height, y_center + crop_height // 2)

            # 裁剪图像
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue  # 跳过无效裁剪

            # 保存裁剪图像
            filename = f"face_{i}.{self.output_format}"
            crop_path = os.path.join(output_dir, filename)

            self._save_image(crop, crop_path)
            self.detected_faces += 1

    def _save_image(self, image, path):
        """
        保存图像

        Args:
            image: OpenCV格式的图像
            path: 保存路径

        Returns:
            图像哈希值
        """
        # 计算图像哈希
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        phash = str(imagehash.phash(pil_image))

        # 根据输出格式保存图像
        if self.output_format.lower() in ['jpg', 'jpeg']:
            cv2.imwrite(path, image, [cv2.IMWRITE_JPEG_QUALITY, self.output_quality])
        elif self.output_format.lower() == 'png':
            cv2.imwrite(path, image, [cv2.IMWRITE_PNG_COMPRESSION, min(9, max(0, 10 - self.output_quality // 10))])
        elif self.output_format.lower() == 'webp':
            cv2.imwrite(path, image, [cv2.IMWRITE_WEBP_QUALITY, self.output_quality])
        else:
            # 默认使用JPEG
            cv2.imwrite(path, image, [cv2.IMWRITE_JPEG_QUALITY, self.output_quality])

        return phash

    def stop(self):
        """停止处理"""
        self.running = False
