"""
主窗口UI模块
"""
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QFileDialog, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QTabWidget, QSplitter, QMessageBox, QGroupBox, QFormLayout,
    QLineEdit, QSlider, QStatusBar, QApplication
)
from PySide6.QtCore import Qt, QThread, Signal, QSize, QUrl, QDir
from PySide6.QtGui import QIcon, QPixmap, QDesktopServices

from src.utils.config import Config
from src.core.database import Database
from src.core.video_processor import VideoProcessor
from src.ui.result_viewer import ResultViewer


class VideoProcessingThread(QThread):
    """视频处理线程"""

    progress_updated = Signal(int, int, float, float, float, int, int)  # current, total, progress, elapsed, remaining, processed_frames, detected_faces
    status_updated = Signal(str)
    processing_finished = Signal(dict)

    def __init__(self, video_processor, video_paths):
        """
        初始化视频处理线程

        Args:
            video_processor: 视频处理器
            video_paths: 视频文件路径列表
        """
        super().__init__()
        self.video_processor = video_processor
        self.video_paths = video_paths

    def run(self):
        """运行线程"""
        results = []

        for i, video_path in enumerate(self.video_paths):
            self.status_updated.emit(f"处理视频 {i+1}/{len(self.video_paths)}: {os.path.basename(video_path)}")

            result = self.video_processor.process_video(
                video_path,
                progress_callback=self.progress_updated.emit,
                status_callback=self.status_updated.emit
            )

            results.append(result)

            if not self.video_processor.running:
                self.status_updated.emit("处理已停止")
                break

        self.processing_finished.emit({"results": results})

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

        # 处理选项卡
        self.processing_tab = QWidget()
        self.tab_widget.addTab(self.processing_tab, "处理")

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

        # 文件选择区域
        file_group = QGroupBox("视频文件")
        file_layout = QVBoxLayout(file_group)

        file_select_layout = QHBoxLayout()
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setReadOnly(True)
        self.file_path_edit.setPlaceholderText("选择视频文件...")
        file_select_layout.addWidget(self.file_path_edit)

        self.browse_button = QPushButton("浏览...")
        self.browse_button.clicked.connect(self.on_browse_clicked)
        file_select_layout.addWidget(self.browse_button)

        file_layout.addLayout(file_select_layout)

        # 文件信息标签
        self.file_info_label = QLabel("未选择文件")
        file_layout.addWidget(self.file_info_label)

        layout.addWidget(file_group)

        # 处理选项区域
        options_group = QGroupBox("处理选项")
        options_layout = QFormLayout(options_group)

        # 检测宽度
        self.detection_width_spin = QSpinBox()
        self.detection_width_spin.setRange(320, 1920)
        self.detection_width_spin.setSingleStep(80)
        self.detection_width_spin.setValue(self.config.get("processing", "detection_width", 640))
        options_layout.addRow("检测宽度:", self.detection_width_spin)

        # 每秒处理帧数
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 30)
        self.fps_spin.setValue(self.config.get("processing", "frames_per_second", 5))
        options_layout.addRow("每秒处理帧数:", self.fps_spin)

        # 人脸检测器类型
        self.detector_type_combo = QComboBox()
        self.detector_type_combo.addItem("YuNet (通用人脸)", "yunet")
        self.detector_type_combo.addItem("Anime (动漫人脸)", "anime")
        self.detector_type_combo.addItem("YOLOv8 (高精度)", "yolov8")
        self.detector_type_combo.addItem("SCRFD (高精度)", "scrfd")

        # 设置当前值
        current_detector = self.config.get("processing", "detector_type", "yunet")
        index = self.detector_type_combo.findData(current_detector)
        if index >= 0:
            self.detector_type_combo.setCurrentIndex(index)

        options_layout.addRow("人脸检测器:", self.detector_type_combo)

        # 置信度阈值
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.1, 1.0)
        self.confidence_spin.setSingleStep(0.05)
        self.confidence_spin.setDecimals(2)
        self.confidence_spin.setValue(self.config.get("processing", "confidence_threshold", 0.6))
        options_layout.addRow("置信度阈值:", self.confidence_spin)

        # 跳过相似帧
        self.skip_similar_check = QCheckBox()
        self.skip_similar_check.setChecked(self.config.get("processing", "skip_similar_frames", True))
        options_layout.addRow("跳过相似帧:", self.skip_similar_check)

        # 帧相似度判断方法
        self.similarity_method_combo = QComboBox()
        self.similarity_method_combo.addItem("感知哈希 (pHash)", "phash")
        self.similarity_method_combo.addItem("结构相似性 (SSIM)", "ssim")

        # 设置当前值
        current_method = self.config.get("processing", "similarity_method", "phash")
        index = self.similarity_method_combo.findData(current_method)
        if index >= 0:
            self.similarity_method_combo.setCurrentIndex(index)

        options_layout.addRow("相似度判断方法:", self.similarity_method_combo)

        # 相似度阈值
        self.similarity_spin = QDoubleSpinBox()
        self.similarity_spin.setRange(0.5, 1.0)
        self.similarity_spin.setSingleStep(0.05)
        self.similarity_spin.setDecimals(2)
        self.similarity_spin.setValue(self.config.get("processing", "similarity_threshold", 0.9))
        options_layout.addRow("相似度阈值:", self.similarity_spin)

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
        self.padding_spin.setValue(self.config.get("processing", "crop_padding", 0.2))
        options_layout.addRow("裁剪边距:", self.padding_spin)

        # 裁剪宽高比
        self.aspect_ratio_combo = QComboBox()
        self.aspect_ratio_combo.addItem("1:1 (正方形)", 1.0)
        self.aspect_ratio_combo.addItem("4:3", 4/3)
        self.aspect_ratio_combo.addItem("3:4", 3/4)
        self.aspect_ratio_combo.addItem("16:9", 16/9)
        self.aspect_ratio_combo.addItem("9:16", 9/16)

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
        self.min_face_size_spin.setValue(self.config.get("processing", "min_face_size", 40))
        self.min_face_size_spin.setToolTip("小于此尺寸的人脸将被忽略，避免裁剪过小不清晰的人脸")
        options_layout.addRow("最小人脸尺寸:", self.min_face_size_spin)

        # 自动人脸分组
        self.auto_face_grouping_check = QCheckBox()
        self.auto_face_grouping_check.setChecked(self.config.get("processing", "auto_face_grouping", True))
        options_layout.addRow("自动人脸分组:", self.auto_face_grouping_check)

        # 输出格式
        self.format_combo = QComboBox()
        self.format_combo.addItem("JPEG (.jpg)", "jpg")
        self.format_combo.addItem("PNG (.png)", "png")
        self.format_combo.addItem("WebP (.webp)", "webp")

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

        quality_layout = QHBoxLayout()
        quality_layout.addWidget(self.quality_slider)
        quality_layout.addWidget(self.quality_label)

        options_layout.addRow("输出质量:", quality_layout)

        layout.addWidget(options_group)

        # 进度区域
        progress_group = QGroupBox("处理进度")
        progress_layout = QVBoxLayout(progress_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)

        self.progress_label = QLabel("就绪")
        progress_layout.addWidget(self.progress_label)

        self.stats_label = QLabel("")
        progress_layout.addWidget(self.stats_label)

        layout.addWidget(progress_group)

        # 按钮区域
        button_layout = QHBoxLayout()

        self.start_button = QPushButton("开始处理")
        self.start_button.clicked.connect(self.on_start_clicked)
        button_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("停止")
        self.stop_button.clicked.connect(self.on_stop_clicked)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)

        layout.addLayout(button_layout)

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
        ui_layout.addRow("缩略图大小:", self.thumbnail_size_spin)

        # 默认视频播放器
        player_layout = QHBoxLayout()
        self.default_player_edit = QLineEdit()
        self.default_player_edit.setText(self.config.get("ui", "default_player", ""))
        self.default_player_edit.setReadOnly(True)
        player_layout.addWidget(self.default_player_edit)

        self.default_player_button = QPushButton("浏览...")
        self.default_player_button.clicked.connect(self.on_default_player_clicked)
        player_layout.addWidget(self.default_player_button)

        ui_layout.addRow("默认视频播放器:", player_layout)

        layout.addWidget(ui_group)

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
        if index == 1:  # 结果选项卡
            self.result_viewer.refresh_results()

    def on_browse_clicked(self):
        """浏览按钮点击处理"""
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setNameFilter("视频文件 (*.mp4 *.avi *.mkv *.mov *.wmv *.flv *.webm)")

        if file_dialog.exec():
            file_paths = file_dialog.selectedFiles()
            if file_paths:
                self.file_path_edit.setText("; ".join(file_paths))
                self.file_info_label.setText(f"已选择 {len(file_paths)} 个文件")

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
        # 获取视频文件路径
        video_paths_text = self.file_path_edit.text()
        if not video_paths_text:
            QMessageBox.warning(self, "警告", "请先选择视频文件")
            return

        video_paths = [path.strip() for path in video_paths_text.split(";")]

        # 保存当前设置
        self.save_settings()

        # 更新视频处理器
        self.video_processor = VideoProcessor(self.config, self.database)

        # 创建并启动处理线程
        self.processing_thread = VideoProcessingThread(self.video_processor, video_paths)
        self.processing_thread.progress_updated.connect(self.update_progress)
        self.processing_thread.status_updated.connect(self.update_status)
        self.processing_thread.processing_finished.connect(self.on_processing_finished)

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

    def on_processing_finished(self, result):
        """
        处理完成回调

        Args:
            result: 处理结果
        """
        # 更新UI状态
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_label.setText("处理完成")

        # 显示结果
        results = result.get("results", [])
        total_faces = sum(r.get("detected_faces", 0) for r in results if r.get("success", False))

        message = f"处理完成，共检测到 {total_faces} 个人脸。\n点击\"结果\"选项卡查看裁剪结果。"
        QMessageBox.information(self, "处理完成", message)

        # 切换到结果选项卡
        self.tab_widget.setCurrentIndex(1)

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

        # 输出选项
        self.config.set("output", "format", self.format_combo.currentData())
        self.config.set("output", "quality", self.quality_slider.value())

        # UI选项
        self.config.set("ui", "thumbnail_size", self.thumbnail_size_spin.value())
        self.config.set("ui", "default_player", self.default_player_edit.text())

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
