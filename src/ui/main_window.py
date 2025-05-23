"""
主窗口UI模块
"""
import os
import sys
import time
import cv2
from datetime import timedelta
from pathlib import Path

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QFileDialog, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QTabWidget, QSplitter, QMessageBox, QGroupBox, QFormLayout,
    QLineEdit, QSlider, QStatusBar, QApplication, QListWidget, QListWidgetItem,
    QScrollArea, QSizePolicy, QProgressDialog
)
from PySide6.QtCore import Qt, QThread, Signal, QSize, QUrl, QDir, QCoreApplication
from PySide6.QtGui import QIcon, QPixmap, QDesktopServices, QKeySequence
from PySide6.QtWidgets import QKeySequenceEdit

from src.utils.config import Config
from src.core.database import Database
from src.core.video_processor import VideoProcessor
from src.core.image_processor import ImageProcessor
from src.ui.result_viewer import ResultViewer
from src.ui.image_processor_tab import ImageProcessorTab


class VideoProcessingThread(QThread):
    """视频处理线程"""

    progress_updated = Signal(int, int, float, float, float, int, int)  # current, total, progress, elapsed, remaining, processed_frames, detected_faces
    status_updated = Signal(str)
    processing_finished = Signal(dict)
    video_processed = Signal(int, dict)  # 单个视频处理完成信号，参数：视频索引，处理结果

    def __init__(self, video_processor, video_path, index=0):
        """
        初始化视频处理线程

        Args:
            video_processor: 视频处理器
            video_path: 要处理的视频文件路径
            index: 视频在队列中的索引
        """
        super().__init__()
        self.video_processor = video_processor
        self.video_path = video_path
        self.index = index  # 当前处理的视频索引

    def run(self):
        """运行线程"""
        self.status_updated.emit(f"处理视频: {os.path.basename(self.video_path)}")

        result = self.video_processor.process_video(
            self.video_path,
            progress_callback=self.progress_updated.emit,
            status_callback=self.status_updated.emit
        )

        # 发送单个视频处理完成信号
        self.video_processed.emit(self.index, result)

        if not self.video_processor.running:
            self.status_updated.emit("处理已停止")

        # 处理完成
        self.processing_finished.emit({"results": [result]})

    def stop(self):
        """停止处理"""
        self.video_processor.stop()


class MainWindow(QMainWindow):
    """主窗口类"""

    def __init__(self):
        """初始化主窗口"""
        super().__init__()

        # 初始化配置和数据库
        self.config = Config()
        self.database = Database(self.config.get_database_path())
        self.video_processor = VideoProcessor(self.config, self.database)

        # 设置窗口属性
        self.setWindowTitle("Vid2Img Cropper")
        self.setMinimumSize(1000, 700)

        # 初始化UI
        self.init_ui()

        # 处理线程
        self.processing_thread = None

    def init_ui(self):
        """初始化UI组件"""
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QVBoxLayout(central_widget)

        # 创建选项卡
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # 视频处理选项卡
        self.processing_tab = QWidget()
        self.tab_widget.addTab(self.processing_tab, "视频处理")

        # 图片处理选项卡
        self.image_processor_tab = ImageProcessorTab(self.config)
        self.tab_widget.addTab(self.image_processor_tab, "图片处理")

        # 结果选项卡
        self.result_viewer = ResultViewer(self.config, self.database)
        self.tab_widget.addTab(self.result_viewer, "结果")

        # 设置选项卡
        self.settings_tab = QWidget()
        self.tab_widget.addTab(self.settings_tab, "设置")

        # 初始化处理选项卡
        self.init_processing_tab()

        # 初始化设置选项卡
        self.init_settings_tab()

        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪")

        # 连接信号
        self.tab_widget.currentChanged.connect(self.on_tab_changed)

    def init_processing_tab(self):
        """初始化处理选项卡"""
        layout = QVBoxLayout(self.processing_tab)

        # 创建主分割器，允许用户调整各区域大小
        main_splitter = QSplitter(Qt.Vertical)
        main_splitter.setHandleWidth(2)
        main_splitter.setChildrenCollapsible(False)

        # 文件选择区域
        file_group = QGroupBox("视频文件队列")
        file_layout = QVBoxLayout(file_group)

        file_select_layout = QHBoxLayout()
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setReadOnly(True)
        self.file_path_edit.setPlaceholderText("选择视频文件...")
        file_select_layout.addWidget(self.file_path_edit)

        self.browse_button = QPushButton("浏览文件...")
        self.browse_button.clicked.connect(self.on_browse_clicked)
        file_select_layout.addWidget(self.browse_button)

        self.browse_folder_button = QPushButton("导入文件夹...")
        self.browse_folder_button.clicked.connect(self.on_browse_folder_clicked)
        file_select_layout.addWidget(self.browse_folder_button)

        file_layout.addLayout(file_select_layout)

        # 视频队列列表
        self.queue_list = QListWidget()
        self.queue_list.setMinimumHeight(100)
        file_layout.addWidget(self.queue_list)

        # 队列操作按钮
        queue_buttons_layout = QHBoxLayout()

        self.add_to_queue_button = QPushButton("添加到队列")
        self.add_to_queue_button.clicked.connect(self.add_to_queue)
        queue_buttons_layout.addWidget(self.add_to_queue_button)

        self.remove_from_queue_button = QPushButton("从队列移除")
        self.remove_from_queue_button.clicked.connect(self.remove_from_queue)
        queue_buttons_layout.addWidget(self.remove_from_queue_button)

        self.clear_queue_button = QPushButton("清空队列")
        self.clear_queue_button.clicked.connect(self.clear_queue)
        queue_buttons_layout.addWidget(self.clear_queue_button)

        file_layout.addLayout(queue_buttons_layout)

        # 文件信息标签
        self.file_info_label = QLabel("未选择文件")
        file_layout.addWidget(self.file_info_label)

        # 将文件选择区域添加到主分割器
        main_splitter.addWidget(file_group)

        # 处理选项区域
        options_group = QGroupBox("处理选项")
        options_main_layout = QVBoxLayout(options_group)
        options_main_layout.setContentsMargins(0, 0, 0, 0)

        # 创建滚动区域
        options_scroll = QScrollArea()
        options_scroll.setWidgetResizable(True)
        options_scroll.setFrameShape(QScrollArea.NoFrame)
        options_scroll.setMinimumHeight(200)  # 设置最小高度
        options_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # 确保垂直滚动条可见
        options_scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # 允许滚动区域扩展

        # 创建内容小部件
        options_content = QWidget()
        options_layout = QFormLayout(options_content)
        options_layout.setContentsMargins(0, 0, 0, 0)
        options_layout.setSpacing(10)  # 增加表单项之间的间距

        # 设置滚动区域的内容
        options_scroll.setWidget(options_content)
        options_main_layout.addWidget(options_scroll)

        # 检测宽度
        self.detection_width_spin = QSpinBox()
        self.detection_width_spin.setRange(320, 1920)
        self.detection_width_spin.setSingleStep(80)
        self.detection_width_spin.setValue(self.config.get("processing", "detection_width", 640))
        self.detection_width_spin.setMinimumHeight(28)  # 增加控件高度
        self.detection_width_spin.setToolTip("视频帧缩放到此宽度进行人脸检测，值越大检测精度越高但速度越慢")
        options_layout.addRow("检测宽度:", self.detection_width_spin)

        # 每秒处理帧数
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 30)
        self.fps_spin.setValue(self.config.get("processing", "frames_per_second", 5))
        self.fps_spin.setMinimumHeight(28)  # 增加控件高度
        self.fps_spin.setToolTip("从视频中抽取帧的频率，值越高检测越全面但处理时间越长")
        options_layout.addRow("每秒处理帧数:", self.fps_spin)

        # 人脸检测器类型
        self.detector_type_combo = QComboBox()
        self.detector_type_combo.setMinimumHeight(28)  # 增加控件高度
        self.detector_type_combo.addItem("YuNet (通用人脸)", "yunet")
        self.detector_type_combo.addItem("Anime (动漫人脸)", "anime")
        self.detector_type_combo.addItem("YOLOv8 (高精度)", "yolov8")
        self.detector_type_combo.addItem("SCRFD (高精度)", "scrfd")
        self.detector_type_combo.setToolTip("选择人脸检测器类型：\n- YuNet: 通用人脸检测，速度和精度均衡\n- Anime: 专为动漫人脸优化\n- YOLOv8: 高精度通用人脸检测\n- SCRFD: 高精度人脸检测")

        # 设置当前值
        current_detector = self.config.get("processing", "detector_type", "yunet")
        index = self.detector_type_combo.findData(current_detector)
        if index >= 0:
            self.detector_type_combo.setCurrentIndex(index)

        options_layout.addRow("人脸检测器:", self.detector_type_combo)

        # 人脸识别模型类型
        self.face_recognition_model_combo = QComboBox()
        self.face_recognition_model_combo.setMinimumHeight(28)  # 增加控件高度
        self.face_recognition_model_combo.addItem("InsightFace (高精度)", "insightface")
        self.face_recognition_model_combo.addItem("OpenCV (轻量级)", "opencv")
        self.face_recognition_model_combo.setToolTip("选择人脸识别模型类型：\n- InsightFace: 高精度人脸识别，支持更准确的人脸分组\n- OpenCV: 轻量级人脸识别，无需下载额外模型")

        # 设置当前值
        current_model = self.config.get("processing", "face_recognition_model", "insightface")
        index = self.face_recognition_model_combo.findData(current_model)
        if index >= 0:
            self.face_recognition_model_combo.setCurrentIndex(index)

        options_layout.addRow("人脸识别模型:", self.face_recognition_model_combo)

        # InsightFace模型名称
        self.face_model_name_combo = QComboBox()
        self.face_model_name_combo.setMinimumHeight(28)  # 增加控件高度
        self.face_model_name_combo.addItem("buffalo_l (高精度)", "buffalo_l")
        self.face_model_name_combo.addItem("buffalo_m (中等精度)", "buffalo_m")
        self.face_model_name_combo.addItem("buffalo_s (轻量级)", "buffalo_s")
        self.face_model_name_combo.setToolTip("选择InsightFace模型大小：\n- buffalo_l: 高精度模型，准确度最高但速度较慢\n- buffalo_m: 中等精度模型，平衡准确度和速度\n- buffalo_s: 轻量级模型，速度最快但准确度较低")

        # 设置当前值
        current_model_name = self.config.get("processing", "face_recognition_model_name", "buffalo_l")
        index = self.face_model_name_combo.findData(current_model_name)
        if index >= 0:
            self.face_model_name_combo.setCurrentIndex(index)

        options_layout.addRow("InsightFace模型:", self.face_model_name_combo)

        # 人脸聚类方法
        self.face_clustering_method_combo = QComboBox()
        self.face_clustering_method_combo.setMinimumHeight(28)  # 增加控件高度
        self.face_clustering_method_combo.addItem("余弦相似度 (推荐)", "cosine")
        self.face_clustering_method_combo.addItem("欧氏距离", "euclidean")
        self.face_clustering_method_combo.setToolTip("选择人脸特征向量的相似度计算方法：\n- 余弦相似度: 计算向量夹角，对尺度不敏感，通常效果更好\n- 欧氏距离: 计算向量间的直线距离")

        # 设置当前值
        current_method = self.config.get("processing", "face_clustering_method", "cosine")
        index = self.face_clustering_method_combo.findData(current_method)
        if index >= 0:
            self.face_clustering_method_combo.setCurrentIndex(index)

        options_layout.addRow("人脸聚类方法:", self.face_clustering_method_combo)

        # 置信度阈值
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.1, 1.0)
        self.confidence_spin.setSingleStep(0.05)
        self.confidence_spin.setDecimals(2)
        self.confidence_spin.setMinimumHeight(28)  # 增加控件高度
        self.confidence_spin.setValue(self.config.get("processing", "confidence_threshold", 0.6))
        self.confidence_spin.setToolTip("人脸检测的置信度阈值，值越高要求越严格，可减少误检但可能漏检")
        options_layout.addRow("置信度阈值:", self.confidence_spin)

        # 跳过相似帧
        self.skip_similar_check = QCheckBox()
        self.skip_similar_check.setChecked(self.config.get("processing", "skip_similar_frames", True))
        self.skip_similar_check.setToolTip("启用后会跳过与前一帧相似的视频帧，可减少重复处理")
        options_layout.addRow("跳过相似帧:", self.skip_similar_check)

        # 帧相似度判断方法
        self.similarity_method_combo = QComboBox()
        self.similarity_method_combo.setMinimumHeight(28)  # 增加控件高度
        self.similarity_method_combo.addItem("感知哈希 (pHash)", "phash")
        self.similarity_method_combo.addItem("结构相似性 (SSIM)", "ssim")
        self.similarity_method_combo.setToolTip("选择判断帧相似度的方法：\n- 感知哈希: 速度快，对光照变化不敏感\n- 结构相似性: 更精确但计算量大")

        # 设置当前值
        current_method = self.config.get("processing", "similarity_method", "phash")
        index = self.similarity_method_combo.findData(current_method)
        if index >= 0:
            self.similarity_method_combo.setCurrentIndex(index)

        options_layout.addRow("相似度判断方法:", self.similarity_method_combo)

        # 帧相似度阈值
        self.similarity_spin = QDoubleSpinBox()
        self.similarity_spin.setRange(0.5, 1.0)
        self.similarity_spin.setSingleStep(0.05)
        self.similarity_spin.setDecimals(2)
        self.similarity_spin.setMinimumHeight(28)  # 增加控件高度
        self.similarity_spin.setValue(self.config.get("processing", "similarity_threshold", 0.9))
        self.similarity_spin.setToolTip("帧相似度阈值，用于判断两帧是否相似，值越高要求越严格")
        options_layout.addRow("帧相似度阈值:", self.similarity_spin)

        # 跳过相似人脸
        self.skip_similar_faces_check = QCheckBox()
        self.skip_similar_faces_check.setChecked(self.config.get("processing", "skip_similar_faces", True))
        self.skip_similar_faces_check.setToolTip("跳过与上一帧中相似的人脸，避免重复裁剪")
        options_layout.addRow("跳过相似人脸:", self.skip_similar_faces_check)

        # 裁剪边距
        self.padding_spin = QDoubleSpinBox()
        self.padding_spin.setRange(0.0, 1.0)
        self.padding_spin.setSingleStep(0.05)
        self.padding_spin.setDecimals(2)
        self.padding_spin.setMinimumHeight(28)  # 增加控件高度
        self.padding_spin.setValue(self.config.get("processing", "crop_padding", 0.2))
        self.padding_spin.setToolTip("裁剪人脸时额外添加的边距比例，值越大保留的周围区域越多")
        options_layout.addRow("裁剪边距:", self.padding_spin)

        # 裁剪宽高比
        self.aspect_ratio_combo = QComboBox()
        self.aspect_ratio_combo.setMinimumHeight(28)  # 增加控件高度
        self.aspect_ratio_combo.addItem("1:1 (正方形)", 1.0)
        self.aspect_ratio_combo.addItem("4:3", 4/3)
        self.aspect_ratio_combo.addItem("3:4", 3/4)
        self.aspect_ratio_combo.addItem("16:9", 16/9)
        self.aspect_ratio_combo.addItem("9:16", 9/16)
        self.aspect_ratio_combo.setToolTip("裁剪图像的宽高比，选择适合您需求的比例")

        # 设置当前值
        current_ratio = self.config.get("processing", "crop_aspect_ratio", 1.0)
        index = self.aspect_ratio_combo.findData(current_ratio)
        if index >= 0:
            self.aspect_ratio_combo.setCurrentIndex(index)

        options_layout.addRow("裁剪宽高比:", self.aspect_ratio_combo)

        # 最小人脸尺寸
        self.min_face_size_spin = QSpinBox()
        self.min_face_size_spin.setRange(20, 200)
        self.min_face_size_spin.setSingleStep(5)
        self.min_face_size_spin.setMinimumHeight(28)  # 增加控件高度
        self.min_face_size_spin.setValue(self.config.get("processing", "min_face_size", 40))
        self.min_face_size_spin.setToolTip("小于此尺寸的人脸将被忽略，避免裁剪过小不清晰的人脸")
        options_layout.addRow("最小人脸尺寸:", self.min_face_size_spin)

        # 自动人脸分组
        self.auto_face_grouping_check = QCheckBox()
        self.auto_face_grouping_check.setChecked(self.config.get("processing", "auto_face_grouping", True))
        self.auto_face_grouping_check.setToolTip("启用后会自动将相似人脸归为同一组，便于后续浏览和管理")
        options_layout.addRow("自动人脸分组:", self.auto_face_grouping_check)

        # 人脸相似度阈值
        self.face_similarity_spin = QDoubleSpinBox()
        self.face_similarity_spin.setRange(0.5, 0.98)
        self.face_similarity_spin.setSingleStep(0.05)
        self.face_similarity_spin.setDecimals(2)
        self.face_similarity_spin.setMinimumHeight(28)  # 增加控件高度
        self.face_similarity_spin.setValue(self.config.get("processing", "face_similarity_threshold", 0.85))
        self.face_similarity_spin.setToolTip("人脸特征相似度阈值，用于判断两个人脸是否为同一人，值越高要求越严格\nInsightFace建议使用0.85-0.9，OpenCV建议使用0.6-0.7")
        options_layout.addRow("人脸相似度阈值:", self.face_similarity_spin)

        # 输出格式
        self.format_combo = QComboBox()
        self.format_combo.setMinimumHeight(28)  # 增加控件高度
        self.format_combo.addItem("JPEG (.jpg)", "jpg")
        self.format_combo.addItem("PNG (.png)", "png")
        self.format_combo.addItem("WebP (.webp)", "webp")
        self.format_combo.setToolTip("选择裁剪图像的保存格式：\n- JPEG: 体积小，有损压缩\n- PNG: 无损压缩，支持透明度\n- WebP: 更高压缩率，同时支持有损和无损")

        # 设置当前值
        current_format = self.config.get("output", "format", "jpg")
        index = self.format_combo.findData(current_format)
        if index >= 0:
            self.format_combo.setCurrentIndex(index)

        options_layout.addRow("输出格式:", self.format_combo)

        # 输出质量
        self.quality_slider = QSlider(Qt.Horizontal)
        self.quality_slider.setRange(1, 100)
        self.quality_slider.setValue(self.config.get("output", "quality", 95))
        self.quality_label = QLabel(f"{self.quality_slider.value()}%")
        self.quality_slider.valueChanged.connect(lambda v: self.quality_label.setText(f"{v}%"))
        self.quality_slider.setToolTip("设置图像保存质量，值越高图像质量越好但文件体积越大")

        quality_layout = QHBoxLayout()
        quality_layout.addWidget(self.quality_slider)
        quality_layout.addWidget(self.quality_label)

        options_layout.addRow("输出质量:", quality_layout)

        # 将处理选项区域添加到主分割器
        main_splitter.addWidget(options_group)

        # 进度区域
        progress_group = QGroupBox("处理进度")
        # 设置大小策略为垂直最小
        progress_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        # 设置最大高度
        progress_group.setMaximumHeight(100)
        progress_layout = QVBoxLayout(progress_group)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        progress_layout.setSpacing(0)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)

        self.progress_label = QLabel("就绪")
        progress_layout.addWidget(self.progress_label)

        self.stats_label = QLabel("")
        progress_layout.addWidget(self.stats_label)

        # 将进度区域添加到主分割器
        main_splitter.addWidget(progress_group)

        # 按钮区域
        buttons_widget = QWidget()
        button_layout = QHBoxLayout(buttons_widget)
        button_layout.setContentsMargins(0, 0, 0, 0)

        self.start_button = QPushButton("开始处理")
        self.start_button.clicked.connect(self.on_start_clicked)
        button_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("停止")
        self.stop_button.clicked.connect(self.on_stop_clicked)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)

        # 将按钮区域添加到主分割器
        main_splitter.addWidget(buttons_widget)

        # 设置分割器初始比例 (文件选择:处理选项:进度:按钮)
        main_splitter.setSizes([200, 400, 100, 50])

        # 将主分割器添加到主布局
        layout.addWidget(main_splitter)

        # 添加弹性空间
        layout.addStretch()

    def init_settings_tab(self):
        """初始化设置选项卡"""
        layout = QVBoxLayout(self.settings_tab)

        # 输出目录设置
        output_group = QGroupBox("输出设置")
        output_layout = QFormLayout(output_group)

        output_dir_layout = QHBoxLayout()
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setText(self.config.get_output_dir())
        self.output_dir_edit.setReadOnly(True)
        output_dir_layout.addWidget(self.output_dir_edit)

        self.output_dir_button = QPushButton("浏览...")
        self.output_dir_button.clicked.connect(self.on_output_dir_clicked)
        output_dir_layout.addWidget(self.output_dir_button)

        output_layout.addRow("输出目录:", output_dir_layout)

        # 数据库路径
        db_layout = QHBoxLayout()
        self.db_path_edit = QLineEdit()
        self.db_path_edit.setText(self.config.get_database_path())
        self.db_path_edit.setReadOnly(True)
        db_layout.addWidget(self.db_path_edit)

        self.open_db_dir_button = QPushButton("打开目录")
        self.open_db_dir_button.clicked.connect(lambda: QDesktopServices.openUrl(
            QUrl.fromLocalFile(os.path.dirname(self.config.get_database_path()))))
        db_layout.addWidget(self.open_db_dir_button)

        output_layout.addRow("数据库路径:", db_layout)

        layout.addWidget(output_group)

        # UI设置
        ui_group = QGroupBox("界面设置")
        ui_layout = QFormLayout(ui_group)

        # 缩略图大小
        self.thumbnail_size_spin = QSpinBox()
        self.thumbnail_size_spin.setRange(50, 300)
        self.thumbnail_size_spin.setSingleStep(10)
        self.thumbnail_size_spin.setValue(self.config.get("ui", "thumbnail_size", 150))
        self.thumbnail_size_spin.valueChanged.connect(lambda v: self.config.set("ui", "thumbnail_size", v))
        self.thumbnail_size_spin.setToolTip("设置结果页面中缩略图的显示大小（像素）")
        ui_layout.addRow("缩略图大小:", self.thumbnail_size_spin)

        # 默认视频播放器
        player_layout = QHBoxLayout()
        self.default_player_edit = QLineEdit()
        self.default_player_edit.setText(self.config.get("ui", "default_player", ""))
        self.default_player_edit.setReadOnly(True)
        self.default_player_edit.setToolTip("设置用于打开视频的外部播放器路径")
        player_layout.addWidget(self.default_player_edit)

        self.default_player_button = QPushButton("浏览...")
        self.default_player_button.clicked.connect(self.on_default_player_clicked)
        player_layout.addWidget(self.default_player_button)

        ui_layout.addRow("默认视频播放器:", player_layout)

        # 收藏快捷键设置
        shortcut_layout = QHBoxLayout()
        self.favorite_shortcut_edit = QKeySequenceEdit()
        self.favorite_shortcut_edit.setKeySequence(QKeySequence(self.config.get("ui", "favorite_shortcut", "Ctrl+S")))
        self.favorite_shortcut_edit.setToolTip("设置用于收藏图片的键盘快捷键")
        shortcut_layout.addWidget(self.favorite_shortcut_edit)

        self.reset_shortcut_button = QPushButton("重置")
        self.reset_shortcut_button.clicked.connect(lambda: self.favorite_shortcut_edit.setKeySequence(QKeySequence("Ctrl+S")))
        shortcut_layout.addWidget(self.reset_shortcut_button)

        ui_layout.addRow("收藏快捷键:", shortcut_layout)

        layout.addWidget(ui_group)

        # 硬件加速设置
        hardware_group = QGroupBox("硬件加速")
        hardware_layout = QFormLayout(hardware_group)

        # 使用GPU加速
        self.use_gpu_check = QCheckBox()
        self.use_gpu_check.setChecked(self.config.get("processing", "use_gpu", False))
        self.use_gpu_check.setToolTip("启用GPU加速可以提高处理速度，但需要CUDA支持")
        hardware_layout.addRow("使用GPU加速:", self.use_gpu_check)

        # GPU内存限制
        self.gpu_memory_limit_spin = QSpinBox()
        self.gpu_memory_limit_spin.setRange(0, 16000)
        self.gpu_memory_limit_spin.setSingleStep(512)
        self.gpu_memory_limit_spin.setValue(self.config.get("processing", "gpu_memory_limit", 0))
        self.gpu_memory_limit_spin.setToolTip("GPU内存限制（MB），0表示不限制。如果遇到内存不足错误，可以设置此值")
        hardware_layout.addRow("GPU内存限制(MB):", self.gpu_memory_limit_spin)

        # 检测GPU可用性
        self.check_gpu_button = QPushButton("检测GPU可用性")
        self.check_gpu_button.clicked.connect(self.check_gpu_availability)
        hardware_layout.addRow("", self.check_gpu_button)

        layout.addWidget(hardware_group)

        # 关于信息
        about_group = QGroupBox("关于")
        about_layout = QVBoxLayout(about_group)

        about_label = QLabel("Vid2Img Cropper - 视频人脸裁剪工具")
        about_label.setAlignment(Qt.AlignCenter)
        about_layout.addWidget(about_label)

        version_label = QLabel("版本: 1.0.0")
        version_label.setAlignment(Qt.AlignCenter)
        about_layout.addWidget(version_label)

        layout.addWidget(about_group)

        # 添加弹性空间
        layout.addStretch()

        # 保存按钮
        self.save_settings_button = QPushButton("保存设置")
        self.save_settings_button.clicked.connect(self.save_settings)
        layout.addWidget(self.save_settings_button)

    def on_tab_changed(self, index):
        """
        切换选项卡时的处理

        Args:
            index: 选项卡索引
        """
        if index == 2:  # 结果选项卡
            self.result_viewer.refresh_results()

    def add_to_queue(self):
        """添加文件到队列"""
        file_paths = self.file_path_edit.text().split("; ")
        if not file_paths or not file_paths[0]:
            QMessageBox.warning(self, "警告", "请先选择视频文件")
            return

        # 添加到队列列表
        for path in file_paths:
            if path and os.path.exists(path):
                # 检查是否已在队列中
                exists = False
                for i in range(self.queue_list.count()):
                    if self.queue_list.item(i).data(Qt.UserRole) == path:
                        exists = True
                        break

                if not exists:
                    item = QListWidgetItem(os.path.basename(path))
                    item.setData(Qt.UserRole, path)
                    self.queue_list.addItem(item)

        # 清空文件选择框
        self.file_path_edit.clear()
        self.file_info_label.setText(f"队列中有 {self.queue_list.count()} 个文件")

    def remove_from_queue(self):
        """从队列中移除选中的文件"""
        selected_items = self.queue_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "警告", "请先选择要移除的文件")
            return

        for item in selected_items:
            row = self.queue_list.row(item)
            self.queue_list.takeItem(row)

        self.file_info_label.setText(f"队列中有 {self.queue_list.count()} 个文件")

    def clear_queue(self):
        """清空队列"""
        if self.queue_list.count() == 0:
            return

        reply = QMessageBox.question(
            self,
            "确认清空",
            "确定要清空队列吗？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.queue_list.clear()
            self.file_info_label.setText("队列已清空")

    def on_browse_clicked(self):
        """浏览文件按钮点击处理"""
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setNameFilter("视频文件 (*.mp4 *.avi *.mkv *.mov *.wmv *.flv *.webm)")

        if file_dialog.exec():
            file_paths = file_dialog.selectedFiles()
            if file_paths:
                self.file_path_edit.setText("; ".join(file_paths))
                self.file_info_label.setText(f"已选择 {len(file_paths)} 个文件")

    def on_browse_folder_clicked(self):
        """导入文件夹按钮点击处理"""
        dir_dialog = QFileDialog()
        dir_dialog.setFileMode(QFileDialog.Directory)
        dir_dialog.setOption(QFileDialog.ShowDirsOnly, True)

        if dir_dialog.exec():
            folder_path = dir_dialog.selectedFiles()[0]
            if folder_path:
                # 询问是否包含子文件夹
                include_subfolders = QMessageBox.question(
                    self,
                    "包含子文件夹",
                    "是否包含子文件夹中的视频文件？",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes
                ) == QMessageBox.Yes

                # 支持的视频文件扩展名
                video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm']

                # 创建进度对话框
                progress_dialog = QProgressDialog("正在扫描文件夹...", "取消", 0, 100, self)
                progress_dialog.setWindowTitle("导入文件夹")
                progress_dialog.setWindowModality(Qt.WindowModal)
                progress_dialog.setMinimumDuration(500)  # 500ms后显示
                progress_dialog.setValue(0)

                # 第一步：扫描文件夹中的所有文件
                all_files = []

                if include_subfolders:
                    # 包含子文件夹，使用os.walk递归扫描
                    for root, _, files in os.walk(folder_path):
                        for file in files:
                            all_files.append(os.path.join(root, file))

                            # 更新进度对话框
                            progress_dialog.setValue(10)  # 扫描完成占10%进度
                            QCoreApplication.processEvents()

                            if progress_dialog.wasCanceled():
                                return
                else:
                    # 不包含子文件夹，只扫描当前文件夹
                    for file in os.listdir(folder_path):
                        file_path = os.path.join(folder_path, file)
                        if os.path.isfile(file_path):
                            all_files.append(file_path)

                            # 更新进度对话框
                            progress_dialog.setValue(10)  # 扫描完成占10%进度
                            QCoreApplication.processEvents()

                            if progress_dialog.wasCanceled():
                                return

                # 第二步：筛选视频文件
                video_files = []
                total_files = len(all_files)

                if total_files == 0:
                    progress_dialog.close()
                    QMessageBox.information(self, "提示", f"文件夹 {folder_path} 中没有文件")
                    return

                for i, file_path in enumerate(all_files):
                    file_ext = os.path.splitext(file_path)[1].lower()
                    if file_ext in video_extensions:
                        video_files.append(file_path)

                    # 更新进度对话框
                    progress = 10 + int((i / total_files) * 40)  # 筛选占40%进度
                    progress_dialog.setValue(progress)
                    progress_dialog.setLabelText(f"正在筛选视频文件... ({i+1}/{total_files})")
                    QCoreApplication.processEvents()

                    if progress_dialog.wasCanceled():
                        return

                if not video_files:
                    progress_dialog.close()
                    QMessageBox.information(self, "提示", f"在文件夹 {folder_path} 中未找到支持的视频文件")
                    self.file_info_label.setText("未找到视频文件")
                    return

                # 第三步：添加到队列
                added_count = 0
                total_videos = len(video_files)

                for i, path in enumerate(video_files):
                    # 检查是否已在队列中
                    exists = False
                    for j in range(self.queue_list.count()):
                        if self.queue_list.item(j).data(Qt.UserRole) == path:
                            exists = True
                            break

                    if not exists:
                        item = QListWidgetItem(os.path.basename(path))
                        item.setData(Qt.UserRole, path)
                        self.queue_list.addItem(item)
                        added_count += 1

                    # 更新进度对话框
                    progress = 50 + int((i / total_videos) * 50)  # 添加占50%进度
                    progress_dialog.setValue(progress)
                    progress_dialog.setLabelText(f"正在添加视频到队列... ({i+1}/{total_videos})")
                    QCoreApplication.processEvents()

                    if progress_dialog.wasCanceled():
                        break

                # 完成
                progress_dialog.setValue(100)
                progress_dialog.close()

                # 更新状态
                if added_count > 0:
                    self.file_info_label.setText(f"从文件夹导入了 {added_count} 个视频文件，队列中共有 {self.queue_list.count()} 个文件")

                    # 询问是否立即开始处理
                    if self.queue_list.count() > 0 and not (self.processing_thread and self.processing_thread.isRunning()):
                        reply = QMessageBox.question(
                            self,
                            "开始处理",
                            f"已成功添加 {added_count} 个视频文件到队列，是否立即开始处理？",
                            QMessageBox.Yes | QMessageBox.No,
                            QMessageBox.No
                        )

                        if reply == QMessageBox.Yes:
                            self.on_start_clicked()
                else:
                    self.file_info_label.setText("未添加新的视频文件（可能已存在于队列中）")

    def on_output_dir_clicked(self):
        """输出目录按钮点击处理"""
        dir_dialog = QFileDialog()
        dir_dialog.setFileMode(QFileDialog.Directory)
        dir_dialog.setOption(QFileDialog.ShowDirsOnly, True)

        if dir_dialog.exec():
            dir_path = dir_dialog.selectedFiles()[0]
            if dir_path:
                self.output_dir_edit.setText(dir_path)
                self.config.set("output", "output_dir", dir_path)

    def on_default_player_clicked(self):
        """默认播放器按钮点击处理"""
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFile)

        # 设置文件过滤器，根据操作系统
        if sys.platform == "darwin":  # macOS
            file_dialog.setNameFilter("应用程序 (*.app);;所有文件 (*)")
        elif sys.platform == "win32":  # Windows
            file_dialog.setNameFilter("可执行文件 (*.exe);;所有文件 (*)")
        else:  # Linux 或其他
            file_dialog.setNameFilter("所有文件 (*)")

        if file_dialog.exec():
            file_path = file_dialog.selectedFiles()[0]
            if file_path:
                self.default_player_edit.setText(file_path)
                self.config.set("ui", "default_player", file_path)

    def on_start_clicked(self):
        """开始按钮点击处理"""
        # 检查队列中是否有视频
        if self.queue_list.count() == 0:
            # 如果队列为空，检查是否有选择的文件
            video_paths_text = self.file_path_edit.text()
            if video_paths_text:
                # 如果有选择的文件，询问是否添加到队列
                reply = QMessageBox.question(
                    self,
                    "添加到队列",
                    "当前队列为空，是否将选择的文件添加到队列？",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes
                )

                if reply == QMessageBox.Yes:
                    self.add_to_queue()
                else:
                    return
            else:
                QMessageBox.warning(self, "警告", "请先选择视频文件或添加文件到队列")
                return

        # 检查是否已经有处理线程在运行
        if self.processing_thread and self.processing_thread.isRunning():
            QMessageBox.warning(self, "警告", "已有视频正在处理中")
            return

        # 从队列中获取第一个视频路径
        if self.queue_list.count() > 0:
            video_path = self.queue_list.item(0).data(Qt.UserRole)
        else:
            QMessageBox.warning(self, "警告", "队列中没有有效的视频文件")
            return

        # 保存当前设置
        self.save_settings()

        # 更新视频处理器
        self.video_processor = VideoProcessor(self.config, self.database)

        # 创建并启动处理线程，只处理队列中的第一个视频
        self.processing_thread = VideoProcessingThread(self.video_processor, video_path, 0)
        self.processing_thread.progress_updated.connect(self.update_progress)
        self.processing_thread.status_updated.connect(self.update_status)
        self.processing_thread.processing_finished.connect(self.on_processing_finished)
        self.processing_thread.video_processed.connect(self.on_video_processed)

        # 更新UI状态
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText("正在处理...")
        self.stats_label.setText("")

        # 启动线程
        self.processing_thread.start()

    def on_stop_clicked(self):
        """停止按钮点击处理"""
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.stop()
            self.progress_label.setText("正在停止...")
            self.stop_button.setEnabled(False)

    def update_progress(self, current, total, progress, elapsed, remaining, processed_frames, detected_faces):
        """
        更新进度信息

        Args:
            current: 当前帧
            total: 总帧数
            progress: 进度比例
            elapsed: 已用时间
            remaining: 剩余时间
            processed_frames: 已处理帧数
            detected_faces: 已检测到的人脸数
        """
        self.progress_bar.setValue(int(progress * 100))

        elapsed_str = str(timedelta(seconds=int(elapsed)))
        remaining_str = str(timedelta(seconds=int(remaining)))

        self.progress_label.setText(f"进度: {current}/{total} ({progress:.1%})")
        self.stats_label.setText(
            f"已用时间: {elapsed_str} | 剩余时间: {remaining_str}\n"
            f"已处理帧数: {processed_frames} | 已检测人脸: {detected_faces}"
        )

    def update_status(self, message):
        """
        更新状态信息

        Args:
            message: 状态消息
        """
        self.status_bar.showMessage(message)

    def on_video_processed(self, index, result):
        """
        单个视频处理完成回调

        Args:
            index: 视频索引
            result: 处理结果
        """
        # 从队列中移除已处理的视频
        if index < self.queue_list.count():
            # 获取处理结果信息
            detected_faces = result.get("detected_faces", 0) if result.get("success", False) else 0

            # 移除队列中的第一个项目
            self.queue_list.takeItem(0)

            # 更新队列信息
            remaining = self.queue_list.count()
            if remaining > 0:
                self.file_info_label.setText(f"已处理完成一个视频，检测到 {detected_faces} 个人脸。队列中还有 {remaining} 个文件待处理")

                # 自动开始处理队列中的下一个视频
                # 等待当前线程完全结束
                if self.processing_thread:
                    self.processing_thread.wait()
                    self.processing_thread = None

                # 启动下一个视频的处理
                QApplication.processEvents()
                self.on_start_clicked()
            else:
                self.file_info_label.setText(f"已处理完成一个视频，检测到 {detected_faces} 个人脸。队列已清空")

                # 更新UI状态
                self.start_button.setEnabled(True)
                self.stop_button.setEnabled(False)
                self.progress_label.setText("处理完成")

                # 切换到结果选项卡
                self.tab_widget.setCurrentIndex(2)

            # 刷新UI
            QApplication.processEvents()

    def on_processing_finished(self, result):
        """
        单个视频处理完成回调（由处理线程发出）

        Args:
            result: 处理结果
        """
        # 如果队列为空，则显示处理完成的消息
        if self.queue_list.count() == 0:
            # 显示结果
            results = result.get("results", [])
            total_faces = sum(r.get("detected_faces", 0) for r in results if r.get("success", False))

            # 更新UI状态
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.progress_label.setText("处理完成")

            # 只有在队列为空时才显示处理完成的消息
            message = f"处理完成，共检测到 {total_faces} 个人脸。\n点击\"结果\"选项卡查看裁剪结果。"
            QMessageBox.information(self, "处理完成", message)

            # 切换到结果选项卡
            self.tab_widget.setCurrentIndex(2)

    def check_gpu_availability(self):
        """检测GPU可用性"""
        try:
            # 检查CUDA是否可用
            cuda_available = False
            cuda_devices = 0
            cuda_version = "未知"

            # 检查OpenCV CUDA支持
            opencv_cuda = False
            try:
                opencv_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0
                if opencv_cuda:
                    cuda_available = True
                    cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
            except Exception:
                pass

            # 检查ONNX Runtime CUDA支持
            onnx_cuda = False
            try:
                import onnxruntime
                providers = onnxruntime.get_available_providers()
                onnx_cuda = 'CUDAExecutionProvider' in providers
                if onnx_cuda:
                    cuda_available = True
            except ImportError:
                pass

            # 尝试获取CUDA版本
            try:
                import torch
                if torch.cuda.is_available():
                    cuda_available = True
                    cuda_devices = torch.cuda.device_count()
                    cuda_version = torch.version.cuda
            except ImportError:
                pass

            # 显示结果
            if cuda_available:
                message = f"GPU加速可用!\n\n"
                message += f"CUDA设备数量: {cuda_devices}\n"
                if cuda_version != "未知":
                    message += f"CUDA版本: {cuda_version}\n"
                message += f"OpenCV CUDA支持: {'是' if opencv_cuda else '否'}\n"
                message += f"ONNX Runtime CUDA支持: {'是' if onnx_cuda else '否'}\n\n"
                message += "您可以启用GPU加速以提高处理速度。"

                QMessageBox.information(self, "GPU检测结果", message)

                # 如果GPU可用但未启用，询问是否启用
                if not self.use_gpu_check.isChecked():
                    reply = QMessageBox.question(
                        self,
                        "启用GPU加速",
                        "检测到可用的GPU，是否启用GPU加速？",
                        QMessageBox.Yes | QMessageBox.No,
                        QMessageBox.Yes
                    )

                    if reply == QMessageBox.Yes:
                        self.use_gpu_check.setChecked(True)
            else:
                QMessageBox.warning(
                    self,
                    "GPU检测结果",
                    "未检测到可用的GPU或CUDA支持。\n\n"
                    "请确保您的系统安装了NVIDIA GPU驱动程序和CUDA。\n"
                    "应用将使用CPU模式运行，这可能会较慢。"
                )

                # 如果GPU不可用但启用了GPU选项，询问是否禁用
                if self.use_gpu_check.isChecked():
                    reply = QMessageBox.question(
                        self,
                        "禁用GPU加速",
                        "未检测到可用的GPU，是否禁用GPU加速选项？",
                        QMessageBox.Yes | QMessageBox.No,
                        QMessageBox.Yes
                    )

                    if reply == QMessageBox.Yes:
                        self.use_gpu_check.setChecked(False)

        except Exception as e:
            QMessageBox.warning(self, "GPU检测错误", f"检测GPU时发生错误:\n{str(e)}")

    def save_settings(self):
        """保存设置"""
        # 处理选项
        self.config.set("processing", "detection_width", self.detection_width_spin.value())
        self.config.set("processing", "frames_per_second", self.fps_spin.value())
        self.config.set("processing", "detector_type", self.detector_type_combo.currentData())
        self.config.set("processing", "confidence_threshold", self.confidence_spin.value())
        self.config.set("processing", "skip_similar_frames", self.skip_similar_check.isChecked())
        self.config.set("processing", "similarity_method", self.similarity_method_combo.currentData())
        self.config.set("processing", "similarity_threshold", self.similarity_spin.value())
        self.config.set("processing", "skip_similar_faces", self.skip_similar_faces_check.isChecked())
        self.config.set("processing", "crop_padding", self.padding_spin.value())
        self.config.set("processing", "crop_aspect_ratio", self.aspect_ratio_combo.currentData())
        self.config.set("processing", "min_face_size", self.min_face_size_spin.value())
        self.config.set("processing", "auto_face_grouping", self.auto_face_grouping_check.isChecked())

        # 人脸识别相关选项
        self.config.set("processing", "face_recognition_model", self.face_recognition_model_combo.currentData())
        self.config.set("processing", "face_recognition_model_name", self.face_model_name_combo.currentData())
        self.config.set("processing", "face_clustering_method", self.face_clustering_method_combo.currentData())
        self.config.set("processing", "face_similarity_threshold", self.face_similarity_spin.value())

        # GPU加速选项
        self.config.set("processing", "use_gpu", self.use_gpu_check.isChecked())
        self.config.set("processing", "gpu_memory_limit", self.gpu_memory_limit_spin.value())

        # 输出选项
        self.config.set("output", "format", self.format_combo.currentData())
        self.config.set("output", "quality", self.quality_slider.value())

        # UI选项
        self.config.set("ui", "thumbnail_size", self.thumbnail_size_spin.value())
        self.config.set("ui", "default_player", self.default_player_edit.text())
        self.config.set("ui", "favorite_shortcut", self.favorite_shortcut_edit.keySequence().toString())

        self.status_bar.showMessage("设置已保存", 3000)

    def closeEvent(self, event):
        """
        窗口关闭事件处理

        Args:
            event: 关闭事件
        """
        # 如果正在处理，询问是否确认退出
        if self.processing_thread and self.processing_thread.isRunning():
            reply = QMessageBox.question(
                self,
                "确认退出",
                "正在处理视频，确定要退出吗？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                self.processing_thread.stop()
                self.processing_thread.wait(1000)  # 等待线程结束
            else:
                event.ignore()
                return

        # 保存设置
        self.save_settings()

        event.accept()
