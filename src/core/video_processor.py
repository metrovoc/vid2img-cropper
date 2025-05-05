"""
视频处理模块，负责视频的加载、处理和人脸裁剪
"""
import os
import cv2
import numpy as np
import time
import imagehash
from PIL import Image
from pathlib import Path
from datetime import timedelta
import io
import math
import logging
from sklearn.metrics.pairwise import cosine_similarity
from skimage.metrics import structural_similarity as ssim

from src.core.face_detector import create_detector
from src.core.face_recognizer import create_face_recognizer

logger = logging.getLogger(__name__)


class VideoProcessor:
    """视频处理器类"""

    def __init__(self, config, database):
        """
        初始化视频处理器

        Args:
            config: 配置对象
            database: 数据库对象
        """
        self.config = config
        self.database = database
        self.detector = None
        self.face_recognizer = None
        self.running = False
        self.last_frame_hash = None
        self.processed_frames = 0
        self.detected_faces = 0

        # 加载配置
        self.detection_width = config.get("processing", "detection_width", 640)
        self.frames_per_second = config.get("processing", "frames_per_second", 5)
        self.confidence_threshold = config.get("processing", "confidence_threshold", 0.6)
        self.skip_similar_frames = config.get("processing", "skip_similar_frames", True)
        self.similarity_threshold = config.get("processing", "similarity_threshold", 0.9)
        self.crop_padding = config.get("processing", "crop_padding", 0.2)
        self.crop_aspect_ratio = config.get("processing", "crop_aspect_ratio", 1.0)
        self.min_face_size = config.get("processing", "min_face_size", 40)
        self.auto_face_grouping = config.get("processing", "auto_face_grouping", True)
        self.face_similarity_threshold = config.get("processing", "face_similarity_threshold", 0.6)

        # 人脸去重相关配置
        self.skip_similar_faces = config.get("processing", "skip_similar_faces", True)
        self.face_iou_threshold = config.get("processing", "face_iou_threshold", 0.7)
        self.face_appearance_threshold = config.get("processing", "face_appearance_threshold", 0.8)

        # 存储上一帧检测到的人脸信息
        self.last_frame_faces = []

        self.output_format = config.get("output", "format", "jpg")
        self.output_quality = config.get("output", "quality", 95)
        self.output_dir = config.get_output_dir()

    def _init_detector(self):
        """初始化人脸检测器"""
        if self.detector is None:
            detector_type = self.config.get("processing", "detector_type", "yunet")
            self.detector = create_detector(
                detector_type=detector_type,
                confidence_threshold=self.confidence_threshold
            )

    def _init_face_recognizer(self):
        """初始化人脸特征提取器"""
        if self.face_recognizer is None:
            # 获取配置的人脸识别模型类型
            recognizer_type = self.config.get("processing", "face_recognition_model", "insightface")
            model_name = self.config.get("processing", "face_recognition_model_name", "buffalo_l")

            # 准备参数
            params = {
                "confidence_threshold": self.face_similarity_threshold
            }

            # 根据识别器类型添加特定参数
            if recognizer_type.lower() == "insightface":
                params["model_name"] = model_name
                # InsightFace特有参数
                det_size = (640, 640)  # 默认检测尺寸
                params["det_size"] = det_size
            elif recognizer_type.lower() == "opencv":
                # OpenCV特有参数，如果有的话
                pass

            try:
                # 创建人脸识别器
                self.face_recognizer = create_face_recognizer(
                    recognizer_type=recognizer_type,
                    **params
                )
                logger.info(f"使用 {recognizer_type} 人脸识别器初始化成功")
            except Exception as e:
                logger.error(f"初始化人脸特征提取器失败: {e}")
                self.face_recognizer = None
                self.auto_face_grouping = False

    def _download_face_recognizer_model(self, save_path):
        """下载人脸特征提取模型"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        print("正在下载人脸特征提取模型...")
        model_url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

        try:
            import urllib.request
            urllib.request.urlretrieve(model_url, save_path)
            print(f"模型已下载到: {save_path}")
        except Exception as e:
            print(f"模型下载失败: {e}")
            print("请手动下载模型并放置到models目录")

    def process_video(self, video_path, progress_callback=None, status_callback=None):
        """
        处理视频文件

        Args:
            video_path: 视频文件路径
            progress_callback: 进度回调函数，接收参数 (current, total, percentage)
            status_callback: 状态回调函数，接收参数 (status_message)

        Returns:
            处理结果字典
        """
        self._init_detector()

        # 如果启用了自动人脸分组，初始化人脸特征提取器
        if self.auto_face_grouping:
            self._init_face_recognizer()

        self.running = True
        self.processed_frames = 0
        self.detected_faces = 0
        self.last_frame_hash = None
        self.last_frame_faces = []  # 重置人脸信息

        # 创建输出目录
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_subdir = os.path.join(self.output_dir, video_name)
        os.makedirs(output_subdir, exist_ok=True)

        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            if status_callback:
                status_callback(f"无法打开视频文件: {video_path}")
            return {"success": False, "error": "无法打开视频文件"}

        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if status_callback:
            status_callback(f"开始处理视频: {video_path}")
            status_callback(f"视频信息: {width}x{height}, {fps} FPS, {timedelta(seconds=duration)}")

        # 计算帧间隔
        if self.frames_per_second >= fps:
            frame_interval = 1  # 处理每一帧
        else:
            frame_interval = int(fps / self.frames_per_second)

        frame_index = 0
        start_time = time.time()

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            # 按指定间隔处理帧
            if frame_index % frame_interval == 0:
                timestamp_ms = int((frame_index / fps) * 1000)

                # 处理当前帧
                self._process_frame(
                    frame, video_path, timestamp_ms, output_subdir,
                    frame_width=width, frame_height=height
                )

                self.processed_frames += 1

                # 更新进度
                if progress_callback and frame_count > 0:
                    progress = frame_index / frame_count
                    elapsed = time.time() - start_time
                    remaining = (elapsed / max(1, frame_index)) * (frame_count - frame_index) if frame_index > 0 else 0

                    progress_callback(
                        frame_index, frame_count, progress,
                        elapsed, remaining, self.processed_frames, self.detected_faces
                    )

            frame_index += 1

        # 关闭视频
        cap.release()

        end_time = time.time()
        processing_time = end_time - start_time

        if status_callback:
            status_callback(f"处理完成: 处理了 {self.processed_frames} 帧，检测到 {self.detected_faces} 个人脸")
            status_callback(f"处理时间: {timedelta(seconds=processing_time)}")

        self.running = False

        return {
            "success": True,
            "video_path": video_path,
            "processed_frames": self.processed_frames,
            "detected_faces": self.detected_faces,
            "processing_time": processing_time
        }

    def _process_frame(self, frame, video_path, timestamp_ms, output_dir, frame_width=None, frame_height=None):
        """
        处理单个视频帧

        Args:
            frame: 视频帧 (OpenCV格式)
            video_path: 视频文件路径
            timestamp_ms: 时间戳（毫秒）
            output_dir: 输出目录
            frame_width: 原始帧宽度
            frame_height: 原始帧高度
        """
        # 检查是否需要跳过相似帧
        if self.skip_similar_frames and self._is_similar_to_previous(frame):
            return

        # 调整大小用于检测
        height, width = frame.shape[:2]
        scale = self.detection_width / width
        detection_height = int(height * scale)
        detection_frame = cv2.resize(frame, (self.detection_width, detection_height))

        # 检测人脸
        faces = self.detector.detect(detection_frame)

        # 将检测结果映射回原始分辨率
        scale_back = width / self.detection_width

        # 存储当前帧的人脸信息，用于下一帧的相似度判断
        current_frame_faces = []

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

            # 检查是否与上一帧的人脸太相似（位置和外观）
            if self.skip_similar_faces and self._is_similar_face(frame, [x, y, w, h]):
                continue  # 跳过相似人脸

            # 添加padding
            padding_x = int(w * self.crop_padding)
            padding_y = int(h * self.crop_padding)

            # 计算裁剪区域，考虑宽高比
            if self.crop_aspect_ratio > 1.0:  # 宽大于高
                crop_w = w + 2 * padding_x
                crop_h = int(crop_w / self.crop_aspect_ratio)
                extra_h = crop_h - (h + 2 * padding_y)
                x1 = max(0, x - padding_x)
                y1 = max(0, y - padding_y - extra_h // 2)
                x2 = min(width, x + w + padding_x)
                y2 = min(height, y + h + padding_y + (extra_h - extra_h // 2))
            elif self.crop_aspect_ratio < 1.0:  # 高大于宽
                crop_h = h + 2 * padding_y
                crop_w = int(crop_h * self.crop_aspect_ratio)
                extra_w = crop_w - (w + 2 * padding_x)
                x1 = max(0, x - padding_x - extra_w // 2)
                y1 = max(0, y - padding_y)
                x2 = min(width, x + w + padding_x + (extra_w - extra_w // 2))
                y2 = min(height, y + h + padding_y)
            else:  # 正方形
                # 确保裁剪区域是正方形
                crop_size = max(w + 2 * padding_x, h + 2 * padding_y)
                x_center = x + w // 2
                y_center = y + h // 2
                x1 = max(0, x_center - crop_size // 2)
                y1 = max(0, y_center - crop_size // 2)
                x2 = min(width, x_center + crop_size // 2)
                y2 = min(height, y_center + crop_size // 2)

            # 裁剪图像
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue  # 跳过无效裁剪

            # 保存裁剪图像
            timestamp_str = self._format_timestamp(timestamp_ms)
            filename = f"{timestamp_str}_{i}.{self.output_format}"
            crop_path = os.path.join(output_dir, filename)

            # 计算图像哈希并检查是否已存在相同图片
            phash, md5_hash = self._save_image(crop, crop_path)

            # 检查是否已存在相同图片
            existing_crop = self.database.get_crop_by_image_hash(md5_hash)
            if existing_crop:
                # 如果已存在相同图片，删除刚保存的图片并使用已存在的图片
                try:
                    os.remove(crop_path)
                    print(f"检测到重复图片，使用已存在的图片: {existing_crop['crop_image_path']}")
                    # 跳过后续处理
                    continue
                except Exception as e:
                    print(f"删除重复图片失败: {e}")

            # 提取人脸特征向量
            feature_vector = None
            group_id = None

            if self.auto_face_grouping and self.face_recognizer is not None:
                try:
                    # 提取人脸区域用于特征提取
                    face_roi = frame[y:y+h, x:x+w]
                    if face_roi.size > 0:
                        # 提取特征向量
                        feature_vector = self.face_recognizer.extract_feature(face_roi)
                        # 尝试分配到现有分组或创建新分组
                        group_id = self._assign_to_face_group(feature_vector, crop_path)
                except Exception as e:
                    logger.error(f"提取人脸特征失败: {e}")

            # 记录到数据库
            bounding_box = [int(x), int(y), int(w), int(h)]
            self.database.add_crop(
                video_path=video_path,
                timestamp_ms=timestamp_ms,
                crop_image_path=crop_path,
                bounding_box=bounding_box,
                confidence=float(confidence),
                frame_width=frame_width,
                frame_height=frame_height,
                group_id=group_id,
                feature_vector=feature_vector,
                image_hash=md5_hash
            )

            self.detected_faces += 1

    def _assign_to_face_group(self, feature_vector, crop_image_path):
        """
        将人脸分配到现有分组或创建新分组

        Args:
            feature_vector: 人脸特征向量
            crop_image_path: 裁剪图像路径

        Returns:
            分组ID
        """
        if feature_vector is None:
            return None

        # 获取所有现有分组
        face_groups = self.database.get_face_groups()

        # 如果没有分组，创建新分组
        if not face_groups:
            group_name = f"人物 1"
            return self.database.add_face_group(
                name=group_name,
                feature_vector=feature_vector,
                sample_image_path=crop_image_path
            )

        # 获取相似度计算方法
        similarity_method = self.config.get("processing", "face_clustering_method", "cosine")

        # 计算与现有分组的相似度
        similarities = []
        for group in face_groups:
            if group['feature_vector'] is None:
                continue

            # 使用人脸识别器计算相似度
            if hasattr(self.face_recognizer, 'compute_similarity'):
                similarity = self.face_recognizer.compute_similarity(
                    feature_vector,
                    group['feature_vector'],
                    method=similarity_method
                )
            else:
                # 回退到内部方法
                similarity = self._compute_face_similarity(feature_vector, group['feature_vector'])

            similarities.append((group['id'], similarity))

        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)

        # 获取最高相似度
        if similarities:
            best_group_id, max_similarity = similarities[0]

            # 如果最大相似度超过阈值，分配到现有分组
            if max_similarity >= self.face_similarity_threshold:
                logger.debug(f"人脸分配到现有分组 {best_group_id}，相似度: {max_similarity:.4f}")
                return best_group_id
            else:
                logger.debug(f"最高相似度 {max_similarity:.4f} 低于阈值 {self.face_similarity_threshold}，创建新分组")

        # 否则创建新分组
        group_name = f"人物 {len(face_groups) + 1}"
        new_group_id = self.database.add_face_group(
            name=group_name,
            feature_vector=feature_vector,
            sample_image_path=crop_image_path
        )
        logger.debug(f"创建新分组 {group_name}，ID: {new_group_id}")
        return new_group_id

    def _compute_face_similarity(self, feature1, feature2):
        """
        计算两个人脸特征向量的相似度

        Args:
            feature1: 特征向量1
            feature2: 特征向量2

        Returns:
            相似度 (0-1)
        """
        try:
            # 转换为numpy数组
            f1 = np.array(feature1).reshape(1, -1)
            f2 = np.array(feature2).reshape(1, -1)

            # 计算余弦相似度
            similarity = cosine_similarity(f1, f2)[0][0]
            return float(similarity)
        except Exception as e:
            print(f"计算人脸特征相似度失败: {e}")
            return 0.0

    def _is_similar_face(self, frame, bbox):
        """
        判断当前人脸是否与上一帧中的某个人脸相似

        Args:
            frame: 当前帧
            bbox: 当前人脸边界框 [x, y, w, h]

        Returns:
            是否相似
        """
        if not self.last_frame_faces:
            # 如果没有上一帧的人脸信息，保存当前人脸并返回False
            self.last_frame_faces.append({
                'bbox': bbox,
                'appearance': self._compute_face_appearance_hash(frame, bbox)
            })
            return False

        # 计算当前人脸与上一帧所有人脸的IoU和外观相似度
        x1, y1, w1, h1 = bbox
        current_appearance = self._compute_face_appearance_hash(frame, bbox)

        for face_info in self.last_frame_faces:
            prev_bbox = face_info['bbox']
            prev_appearance = face_info['appearance']

            # 计算IoU
            iou = self._compute_iou(bbox, prev_bbox)

            # 如果IoU足够高，检查外观相似度
            if iou >= self.face_iou_threshold:
                # 计算外观相似度
                appearance_similarity = 1 - (current_appearance - prev_appearance) / (8 * 8)  # 假设hash_size=8

                # 如果外观也足够相似，认为是同一个人脸
                if appearance_similarity >= self.face_appearance_threshold:
                    # 更新人脸信息
                    face_info['bbox'] = bbox
                    face_info['appearance'] = current_appearance
                    return True

        # 如果没有找到相似的人脸，添加到列表中
        self.last_frame_faces.append({
            'bbox': bbox,
            'appearance': current_appearance
        })

        # 限制列表大小，避免内存占用过大
        if len(self.last_frame_faces) > 10:
            self.last_frame_faces = self.last_frame_faces[-10:]

        return False

    def _compute_face_appearance_hash(self, frame, bbox, hash_size=8):
        """
        计算人脸区域的感知哈希

        Args:
            frame: 视频帧
            bbox: 人脸边界框 [x, y, w, h]
            hash_size: 哈希大小

        Returns:
            感知哈希
        """
        try:
            x, y, w, h = bbox
            # 提取人脸区域
            face_roi = frame[y:y+h, x:x+w]
            if face_roi.size == 0:
                return imagehash.ImageHash(np.zeros((hash_size, hash_size), dtype=np.bool_))

            # 转换为PIL图像
            face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(face_rgb)

            # 计算感知哈希
            return imagehash.phash(pil_image, hash_size=hash_size)
        except Exception as e:
            print(f"计算人脸外观哈希失败: {e}")
            return imagehash.ImageHash(np.zeros((hash_size, hash_size), dtype=np.bool_))

    def _compute_iou(self, bbox1, bbox2):
        """
        计算两个边界框的IoU (Intersection over Union)

        Args:
            bbox1: 边界框1 [x, y, w, h]
            bbox2: 边界框2 [x, y, w, h]

        Returns:
            IoU值 (0-1)
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # 计算边界框的坐标
        x1_1, y1_1, x2_1, y2_1 = x1, y1, x1 + w1, y1 + h1
        x1_2, y1_2, x2_2, y2_2 = x2, y2, x2 + w2, y2 + h2

        # 计算交集区域
        xx1 = max(x1_1, x1_2)
        yy1 = max(y1_1, y1_2)
        xx2 = min(x2_1, x2_2)
        yy2 = min(y2_1, y2_2)

        # 计算交集面积
        w = max(0, xx2 - xx1)
        h = max(0, yy2 - yy1)
        intersection = w * h

        # 计算并集面积
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection

        # 计算IoU
        iou = intersection / union if union > 0 else 0
        return iou

    def _is_similar_to_previous(self, frame, hash_size=8):
        """
        检查当前帧是否与上一帧相似

        Args:
            frame: 当前帧
            hash_size: 感知哈希大小

        Returns:
            是否相似
        """
        # 获取相似度判断方法
        similarity_method = self.config.get("processing", "similarity_method", "phash")

        if similarity_method == "ssim":
            return self._is_similar_ssim(frame)
        else:  # 默认使用感知哈希
            return self._is_similar_phash(frame, hash_size)

    def _is_similar_phash(self, frame, hash_size=8):
        """
        使用感知哈希判断帧相似度

        Args:
            frame: 当前帧
            hash_size: 感知哈希大小

        Returns:
            是否相似
        """
        if self.last_frame_hash is None:
            # 第一帧，计算哈希并返回False
            self.last_frame_hash = self._compute_frame_hash(frame, hash_size)
            return False

        # 计算当前帧的哈希
        current_hash = self._compute_frame_hash(frame, hash_size)

        # 计算哈希相似度
        similarity = 1 - (current_hash - self.last_frame_hash) / (hash_size * hash_size)

        # 更新上一帧哈希
        self.last_frame_hash = current_hash

        # 如果相似度高于阈值，认为是相似帧
        return similarity >= self.similarity_threshold

    def _is_similar_ssim(self, frame):
        """
        使用结构相似性(SSIM)判断帧相似度

        Args:
            frame: 当前帧

        Returns:
            是否相似
        """
        if not hasattr(self, 'last_frame'):
            # 第一帧，保存并返回False
            self.last_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return False

        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 计算SSIM
        similarity_index, _ = ssim(self.last_frame, gray, full=True)

        # 更新上一帧
        self.last_frame = gray

        # 如果相似度高于阈值，认为是相似帧
        return similarity_index >= self.similarity_threshold

    def _compute_frame_hash(self, frame, hash_size=8):
        """
        计算帧的感知哈希

        Args:
            frame: 视频帧
            hash_size: 哈希大小

        Returns:
            感知哈希
        """
        # 转换为PIL图像
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        # 计算感知哈希
        return imagehash.phash(pil_image, hash_size=hash_size)

    def _format_timestamp(self, timestamp_ms):
        """
        格式化时间戳

        Args:
            timestamp_ms: 时间戳（毫秒）

        Returns:
            格式化的时间戳字符串
        """
        total_seconds = timestamp_ms / 1000
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        milliseconds = int(timestamp_ms % 1000)

        return f"{hours:02d}_{minutes:02d}_{seconds:02d}_{milliseconds:03d}"

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

        # 计算文件MD5哈希
        md5_hash = self.database.compute_image_hash(path)

        return phash, md5_hash

    def stop(self):
        """停止处理"""
        self.running = False
