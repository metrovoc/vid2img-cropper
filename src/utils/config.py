"""
配置管理模块，负责应用程序的配置参数
"""
import os
import json
from pathlib import Path


class Config:
    """配置管理类"""

    DEFAULT_CONFIG = {
        # 视频处理配置
        "processing": {
            "detection_width": 640,  # 检测用的缩放宽度
            "frames_per_second": 5,  # 每秒处理的帧数
            "detector_type": "yunet",  # 人脸检测器类型: yunet, anime, yolov8, scrfd
            "confidence_threshold": 0.6,  # 人脸检测置信度阈值
            "skip_similar_frames": True,  # 是否跳过相似帧
            "similarity_method": "phash",  # 帧相似度判断方法: phash, ssim
            "similarity_threshold": 0.9,  # 相似帧阈值
            "crop_padding": 0.2,  # 裁剪时额外添加的边距（相对于人脸大小的比例）
            "crop_aspect_ratio": 1.0,  # 裁剪的宽高比
            "min_face_size": 40,  # 最小人脸尺寸（像素）
            "auto_face_grouping": True,  # 是否自动对人脸进行分组
            "face_recognition_model": "insightface",  # 人脸识别模型类型: insightface, opencv
            "face_recognition_model_name": "buffalo_l",  # InsightFace模型名称: buffalo_l, buffalo_m, buffalo_s
            "face_similarity_threshold": 0.85,  # 人脸特征向量相似度阈值
            "face_clustering_method": "cosine",  # 人脸聚类方法: cosine, euclidean
            "skip_similar_faces": True,  # 是否跳过相似人脸
            "face_iou_threshold": 0.7,  # 人脸IoU阈值
            "face_appearance_threshold": 0.8,  # 人脸外观相似度阈值
            "use_gpu": False,  # 是否使用GPU加速
            "gpu_memory_limit": 0,  # GPU内存限制（MB），0表示不限制
        },
        # 输出配置
        "output": {
            "format": "jpg",  # 输出格式：jpg, png, webp
            "quality": 95,  # 输出质量 (1-100)
            "output_dir": "",  # 输出目录，空字符串表示使用默认位置
        },
        # 数据库配置
        "database": {
            "path": "",  # 数据库路径，空字符串表示使用默认位置
        },
        # UI配置
        "ui": {
            "theme": "system",  # 主题：light, dark, system
            "thumbnail_size": 150,  # 缩略图大小
            "default_player": "",  # 默认视频播放器路径
            "use_embedded_player": True,  # 是否使用内嵌播放器
            "player_volume": 50,  # 播放器音量 (0-100)
            "favorite_shortcut": "Ctrl+S",  # 收藏快捷键
        }
    }

    def __init__(self):
        """初始化配置"""
        self.app_dir = self._get_app_dir()
        self.config_path = os.path.join(self.app_dir, "config.json")
        self.config = self._load_config()

    def _get_app_dir(self):
        """获取应用程序数据目录"""
        home = Path.home()
        app_dir = os.path.join(home, ".vid2img-cropper")
        os.makedirs(app_dir, exist_ok=True)
        return app_dir

    def _load_config(self):
        """加载配置文件，如果不存在则创建默认配置"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                # 合并默认配置，确保新增配置项也存在
                merged_config = self.DEFAULT_CONFIG.copy()
                self._deep_update(merged_config, config)
                return merged_config
            except Exception as e:
                print(f"加载配置文件失败: {e}")
                return self.DEFAULT_CONFIG.copy()
        else:
            # 创建默认配置文件
            self.save_config(self.DEFAULT_CONFIG)
            return self.DEFAULT_CONFIG.copy()

    def _deep_update(self, d, u):
        """递归更新嵌套字典"""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._deep_update(d[k], v)
            else:
                d[k] = v

    def save_config(self, config=None):
        """保存配置到文件"""
        if config is None:
            config = self.config
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            print(f"保存配置文件失败: {e}")

    def get(self, section, key, default=None):
        """获取配置项"""
        try:
            return self.config[section][key]
        except KeyError:
            return default

    def set(self, section, key, value):
        """设置配置项"""
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
        self.save_config()

    def get_database_path(self):
        """获取数据库路径"""
        db_path = self.get("database", "path")
        if not db_path:
            db_path = os.path.join(self.app_dir, "crops.db")
        return db_path

    def get_output_dir(self):
        """获取输出目录"""
        output_dir = self.get("output", "output_dir")
        if not output_dir:
            output_dir = os.path.join(self.app_dir, "crops")
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
