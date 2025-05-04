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

from src.core.face_detector import create_detector


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
        
        self.output_format = config.get("output", "format", "jpg")
        self.output_quality = config.get("output", "quality", 95)
        self.output_dir = config.get_output_dir()
    
    def _init_detector(self):
        """初始化人脸检测器"""
        if self.detector is None:
            self.detector = create_detector(
                detector_type="yunet",
                confidence_threshold=self.confidence_threshold
            )
    
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
        self.running = True
        self.processed_frames = 0
        self.detected_faces = 0
        self.last_frame_hash = None
        
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
        for i, face in enumerate(faces):
            x, y, w, h, confidence = face
            
            # 映射回原始分辨率
            x = int(x * scale_back)
            y = int(y * scale_back)
            w = int(w * scale_back)
            h = int(h * scale_back)
            
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
            
            self._save_image(crop, crop_path)
            
            # 记录到数据库
            bounding_box = [int(x), int(y), int(w), int(h)]
            self.database.add_crop(
                video_path=video_path,
                timestamp_ms=timestamp_ms,
                crop_image_path=crop_path,
                bounding_box=bounding_box,
                confidence=float(confidence),
                frame_width=frame_width,
                frame_height=frame_height
            )
            
            self.detected_faces += 1
    
    def _is_similar_to_previous(self, frame, hash_size=8):
        """
        检查当前帧是否与上一帧相似
        
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
        """
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
    
    def stop(self):
        """停止处理"""
        self.running = False
