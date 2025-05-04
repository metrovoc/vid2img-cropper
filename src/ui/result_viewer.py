"""
结果查看器模块，用于显示和管理裁剪结果
"""
import os
import sys
import subprocess
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QComboBox, QFileDialog,
    QMessageBox, QSplitter, QGroupBox, QFormLayout, QSpinBox,
    QScrollArea, QGridLayout, QMenu, QApplication
)
from PySide6.QtCore import Qt, QSize, QUrl, QProcess, Signal, QThread
from PySide6.QtGui import QPixmap, QIcon, QDesktopServices, QAction


class ThumbnailLoader(QThread):
    """缩略图加载线程"""
    
    thumbnail_loaded = Signal(int, QPixmap)
    
    def __init__(self, crop_items, thumbnail_size):
        """
        初始化缩略图加载线程
        
        Args:
            crop_items: 裁剪项列表
            thumbnail_size: 缩略图大小
        """
        super().__init__()
        self.crop_items = crop_items
        self.thumbnail_size = thumbnail_size
        self.running = True
    
    def run(self):
        """运行线程"""
        for i, crop in enumerate(self.crop_items):
            if not self.running:
                break
            
            try:
                pixmap = QPixmap(crop["crop_image_path"])
                if not pixmap.isNull():
                    pixmap = pixmap.scaled(
                        self.thumbnail_size, self.thumbnail_size,
                        Qt.KeepAspectRatio, Qt.SmoothTransformation
                    )
                    self.thumbnail_loaded.emit(i, pixmap)
            except Exception as e:
                print(f"加载缩略图失败: {e}")
    
    def stop(self):
        """停止线程"""
        self.running = False


class CropThumbnail(QWidget):
    """裁剪缩略图组件"""
    
    clicked = Signal(dict)  # 点击信号，传递裁剪项
    
    def __init__(self, crop_item, thumbnail_size):
        """
        初始化裁剪缩略图组件
        
        Args:
            crop_item: 裁剪项
            thumbnail_size: 缩略图大小
        """
        super().__init__()
        self.crop_item = crop_item
        self.thumbnail_size = thumbnail_size
        
        self.init_ui()
    
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # 缩略图标签
        self.image_label = QLabel()
        self.image_label.setFixedSize(self.thumbnail_size, self.thumbnail_size)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ddd;")
        
        # 加载缩略图
        self.load_thumbnail()
        
        layout.addWidget(self.image_label)
        
        # 时间戳标签
        timestamp_ms = self.crop_item["timestamp_ms"]
        timestamp_str = self.format_timestamp(timestamp_ms)
        
        self.timestamp_label = QLabel(timestamp_str)
        self.timestamp_label.setAlignment(Qt.AlignCenter)
        self.timestamp_label.setStyleSheet("font-size: 10px;")
        
        layout.addWidget(self.timestamp_label)
        
        # 设置鼠标悬停效果
        self.setStyleSheet("""
            CropThumbnail:hover {
                background-color: #e0e0e0;
            }
        """)
    
    def load_thumbnail(self):
        """加载缩略图"""
        try:
            pixmap = QPixmap(self.crop_item["crop_image_path"])
            if not pixmap.isNull():
                pixmap = pixmap.scaled(
                    self.thumbnail_size, self.thumbnail_size,
                    Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                self.image_label.setPixmap(pixmap)
            else:
                self.image_label.setText("图像加载失败")
        except Exception as e:
            self.image_label.setText("图像加载失败")
            print(f"加载缩略图失败: {e}")
    
    def set_thumbnail(self, pixmap):
        """
        设置缩略图
        
        Args:
            pixmap: 缩略图
        """
        self.image_label.setPixmap(pixmap)
    
    def format_timestamp(self, timestamp_ms):
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
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def mousePressEvent(self, event):
        """
        鼠标按下事件处理
        
        Args:
            event: 鼠标事件
        """
        self.clicked.emit(self.crop_item)
        super().mousePressEvent(event)


class ResultViewer(QWidget):
    """结果查看器组件"""
    
    def __init__(self, config, database):
        """
        初始化结果查看器
        
        Args:
            config: 配置对象
            database: 数据库对象
        """
        super().__init__()
        self.config = config
        self.database = database
        self.current_video = None
        self.crop_items = []
        self.thumbnail_loader = None
        
        self.init_ui()
    
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        
        # 顶部控制区域
        control_layout = QHBoxLayout()
        
        # 视频选择
        self.video_combo = QComboBox()
        self.video_combo.setMinimumWidth(300)
        self.video_combo.currentIndexChanged.connect(self.on_video_changed)
        control_layout.addWidget(QLabel("选择视频:"))
        control_layout.addWidget(self.video_combo)
        
        # 刷新按钮
        self.refresh_button = QPushButton("刷新")
        self.refresh_button.clicked.connect(self.refresh_results)
        control_layout.addWidget(self.refresh_button)
        
        # 打开输出目录按钮
        self.open_output_dir_button = QPushButton("打开输出目录")
        self.open_output_dir_button.clicked.connect(self.open_output_dir)
        control_layout.addWidget(self.open_output_dir_button)
        
        control_layout.addStretch()
        
        layout.addLayout(control_layout)
        
        # 分割器
        splitter = QSplitter(Qt.Horizontal)
        
        # 缩略图区域
        self.thumbnails_widget = QWidget()
        self.thumbnails_layout = QGridLayout(self.thumbnails_widget)
        self.thumbnails_layout.setContentsMargins(10, 10, 10, 10)
        self.thumbnails_layout.setSpacing(10)
        
        # 滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.thumbnails_widget)
        
        splitter.addWidget(scroll_area)
        
        # 详情区域
        details_widget = QWidget()
        details_layout = QVBoxLayout(details_widget)
        
        # 详情组
        details_group = QGroupBox("详细信息")
        details_form = QFormLayout(details_group)
        
        self.detail_video_path = QLabel("")
        details_form.addRow("视频路径:", self.detail_video_path)
        
        self.detail_timestamp = QLabel("")
        details_form.addRow("时间戳:", self.detail_timestamp)
        
        self.detail_image_path = QLabel("")
        details_form.addRow("图像路径:", self.detail_image_path)
        
        self.detail_confidence = QLabel("")
        details_form.addRow("置信度:", self.detail_confidence)
        
        details_layout.addWidget(details_group)
        
        # 预览组
        preview_group = QGroupBox("预览")
        preview_layout = QVBoxLayout(preview_group)
        
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(300, 300)
        self.preview_label.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ddd;")
        preview_layout.addWidget(self.preview_label)
        
        details_layout.addWidget(preview_group)
        
        # 操作组
        actions_group = QGroupBox("操作")
        actions_layout = QVBoxLayout(actions_group)
        
        self.open_video_button = QPushButton("打开视频")
        self.open_video_button.clicked.connect(self.open_video)
        actions_layout.addWidget(self.open_video_button)
        
        self.jump_to_time_button = QPushButton("跳转到时间点")
        self.jump_to_time_button.clicked.connect(self.jump_to_time)
        actions_layout.addWidget(self.jump_to_time_button)
        
        self.open_image_button = QPushButton("打开图像")
        self.open_image_button.clicked.connect(self.open_image)
        actions_layout.addWidget(self.open_image_button)
        
        self.delete_crop_button = QPushButton("删除裁剪")
        self.delete_crop_button.clicked.connect(self.delete_crop)
        actions_layout.addWidget(self.delete_crop_button)
        
        details_layout.addWidget(actions_group)
        
        # 添加弹性空间
        details_layout.addStretch()
        
        splitter.addWidget(details_widget)
        
        # 设置分割器比例
        splitter.setSizes([700, 300])
        
        layout.addWidget(splitter)
        
        # 状态标签
        self.status_label = QLabel("")
        layout.addWidget(self.status_label)
        
        # 初始化右键菜单
        self.init_context_menu()
        
        # 禁用详情区域按钮
        self.disable_detail_buttons()
        
        # 加载视频列表
        self.load_videos()
    
    def init_context_menu(self):
        """初始化右键菜单"""
        self.context_menu = QMenu(self)
        
        self.action_open_image = QAction("打开图像", self)
        self.action_open_image.triggered.connect(self.open_image)
        self.context_menu.addAction(self.action_open_image)
        
        self.action_open_video = QAction("打开视频", self)
        self.action_open_video.triggered.connect(self.open_video)
        self.context_menu.addAction(self.action_open_video)
        
        self.action_jump_to_time = QAction("跳转到时间点", self)
        self.action_jump_to_time.triggered.connect(self.jump_to_time)
        self.context_menu.addAction(self.action_jump_to_time)
        
        self.context_menu.addSeparator()
        
        self.action_delete_crop = QAction("删除裁剪", self)
        self.action_delete_crop.triggered.connect(self.delete_crop)
        self.context_menu.addAction(self.action_delete_crop)
    
    def load_videos(self):
        """加载视频列表"""
        self.video_combo.clear()
        
        videos = self.database.get_videos()
        if videos:
            self.video_combo.addItem("全部视频", None)
            for video in videos:
                # 使用文件名作为显示文本
                display_name = os.path.basename(video)
                self.video_combo.addItem(display_name, video)
        else:
            self.status_label.setText("没有找到处理过的视频")
    
    def refresh_results(self):
        """刷新结果"""
        # 重新加载视频列表
        self.load_videos()
        
        # 加载当前选中视频的裁剪结果
        self.load_crops()
    
    def on_video_changed(self, index):
        """
        视频选择变化处理
        
        Args:
            index: 选中的索引
        """
        self.current_video = self.video_combo.currentData()
        self.load_crops()
    
    def load_crops(self):
        """加载裁剪结果"""
        # 清空缩略图区域
        self.clear_thumbnails()
        
        # 停止之前的缩略图加载线程
        if self.thumbnail_loader and self.thumbnail_loader.isRunning():
            self.thumbnail_loader.stop()
            self.thumbnail_loader.wait()
        
        # 从数据库加载裁剪结果
        self.crop_items = self.database.get_crops(self.current_video)
        
        if not self.crop_items:
            self.status_label.setText("没有找到裁剪结果")
            return
        
        # 获取缩略图大小
        thumbnail_size = self.config.get("ui", "thumbnail_size", 150)
        
        # 计算网格列数
        grid_width = self.thumbnails_widget.width()
        columns = max(1, grid_width // (thumbnail_size + 20))
        
        # 创建缩略图
        for i, crop in enumerate(self.crop_items):
            row = i // columns
            col = i % columns
            
            thumbnail = CropThumbnail(crop, thumbnail_size)
            thumbnail.clicked.connect(self.on_thumbnail_clicked)
            thumbnail.setContextMenuPolicy(Qt.CustomContextMenu)
            thumbnail.customContextMenuRequested.connect(lambda pos, t=thumbnail: self.show_context_menu(pos, t))
            
            self.thumbnails_layout.addWidget(thumbnail, row, col)
        
        # 启动缩略图加载线程
        self.thumbnail_loader = ThumbnailLoader(self.crop_items, thumbnail_size)
        self.thumbnail_loader.thumbnail_loaded.connect(self.update_thumbnail)
        self.thumbnail_loader.start()
        
        self.status_label.setText(f"找到 {len(self.crop_items)} 个裁剪结果")
    
    def update_thumbnail(self, index, pixmap):
        """
        更新缩略图
        
        Args:
            index: 索引
            pixmap: 缩略图
        """
        item = self.thumbnails_layout.itemAtPosition(index // self.get_columns(), index % self.get_columns())
        if item and item.widget():
            thumbnail = item.widget()
            thumbnail.set_thumbnail(pixmap)
    
    def get_columns(self):
        """
        获取网格列数
        
        Returns:
            列数
        """
        thumbnail_size = self.config.get("ui", "thumbnail_size", 150)
        grid_width = self.thumbnails_widget.width()
        return max(1, grid_width // (thumbnail_size + 20))
    
    def clear_thumbnails(self):
        """清空缩略图区域"""
        # 清除所有缩略图
        while self.thumbnails_layout.count():
            item = self.thumbnails_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # 清空详情区域
        self.clear_details()
    
    def clear_details(self):
        """清空详情区域"""
        self.detail_video_path.setText("")
        self.detail_timestamp.setText("")
        self.detail_image_path.setText("")
        self.detail_confidence.setText("")
        self.preview_label.clear()
        
        self.disable_detail_buttons()
    
    def disable_detail_buttons(self):
        """禁用详情区域按钮"""
        self.open_video_button.setEnabled(False)
        self.jump_to_time_button.setEnabled(False)
        self.open_image_button.setEnabled(False)
        self.delete_crop_button.setEnabled(False)
    
    def on_thumbnail_clicked(self, crop_item):
        """
        缩略图点击处理
        
        Args:
            crop_item: 裁剪项
        """
        # 更新详情区域
        self.update_details(crop_item)
    
    def show_context_menu(self, pos, thumbnail):
        """
        显示右键菜单
        
        Args:
            pos: 位置
            thumbnail: 缩略图组件
        """
        # 更新详情区域
        self.update_details(thumbnail.crop_item)
        
        # 显示菜单
        self.context_menu.exec_(thumbnail.mapToGlobal(pos))
    
    def update_details(self, crop_item):
        """
        更新详情区域
        
        Args:
            crop_item: 裁剪项
        """
        # 更新详情标签
        self.detail_video_path.setText(crop_item["video_path"])
        
        timestamp_ms = crop_item["timestamp_ms"]
        timestamp_str = self.format_timestamp(timestamp_ms)
        self.detail_timestamp.setText(timestamp_str)
        
        self.detail_image_path.setText(crop_item["crop_image_path"])
        
        confidence = crop_item.get("confidence")
        if confidence is not None:
            self.detail_confidence.setText(f"{confidence:.2f}")
        else:
            self.detail_confidence.setText("未知")
        
        # 加载预览图
        try:
            pixmap = QPixmap(crop_item["crop_image_path"])
            if not pixmap.isNull():
                preview_size = min(self.preview_label.width(), self.preview_label.height())
                pixmap = pixmap.scaled(
                    preview_size, preview_size,
                    Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                self.preview_label.setPixmap(pixmap)
            else:
                self.preview_label.setText("图像加载失败")
        except Exception as e:
            self.preview_label.setText("图像加载失败")
            print(f"加载预览图失败: {e}")
        
        # 启用按钮
        self.open_video_button.setEnabled(True)
        self.jump_to_time_button.setEnabled(True)
        self.open_image_button.setEnabled(True)
        self.delete_crop_button.setEnabled(True)
        
        # 保存当前选中的裁剪项
        self.current_crop = crop_item
    
    def format_timestamp(self, timestamp_ms):
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
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
    
    def open_video(self):
        """打开视频"""
        if not hasattr(self, "current_crop"):
            return
        
        video_path = self.current_crop["video_path"]
        if not os.path.exists(video_path):
            QMessageBox.warning(self, "错误", f"视频文件不存在: {video_path}")
            return
        
        # 使用系统默认程序打开视频
        QDesktopServices.openUrl(QUrl.fromLocalFile(video_path))
    
    def jump_to_time(self):
        """跳转到时间点"""
        if not hasattr(self, "current_crop"):
            return
        
        video_path = self.current_crop["video_path"]
        if not os.path.exists(video_path):
            QMessageBox.warning(self, "错误", f"视频文件不存在: {video_path}")
            return
        
        timestamp_ms = self.current_crop["timestamp_ms"]
        timestamp_seconds = timestamp_ms / 1000
        
        # 尝试使用不同的播放器打开视频并跳转到指定时间
        self.try_open_with_player(video_path, timestamp_seconds)
    
    def try_open_with_player(self, video_path, timestamp_seconds):
        """
        尝试使用不同的播放器打开视频
        
        Args:
            video_path: 视频路径
            timestamp_seconds: 时间戳（秒）
        """
        # 检测操作系统
        if sys.platform == "darwin":  # macOS
            # 尝试使用 IINA
            try:
                subprocess.Popen(["open", "-a", "IINA", video_path, "--args", f"--start={timestamp_seconds}"])
                return
            except:
                pass
            
            # 尝试使用 VLC
            try:
                subprocess.Popen(["open", "-a", "VLC", video_path, "--args", f"--start-time={int(timestamp_seconds)}"])
                return
            except:
                pass
            
            # 使用默认播放器
            QDesktopServices.openUrl(QUrl.fromLocalFile(video_path))
            QMessageBox.information(
                self, 
                "提示", 
                f"已打开视频，请手动跳转到 {self.format_timestamp(int(timestamp_seconds * 1000))}"
            )
        
        elif sys.platform == "win32":  # Windows
            # 尝试使用 VLC
            vlc_path = "C:\\Program Files\\VideoLAN\\VLC\\vlc.exe"
            if os.path.exists(vlc_path):
                try:
                    subprocess.Popen([vlc_path, video_path, f"--start-time={int(timestamp_seconds)}"])
                    return
                except:
                    pass
            
            # 使用默认播放器
            QDesktopServices.openUrl(QUrl.fromLocalFile(video_path))
            QMessageBox.information(
                self, 
                "提示", 
                f"已打开视频，请手动跳转到 {self.format_timestamp(int(timestamp_seconds * 1000))}"
            )
        
        else:  # Linux 或其他
            # 尝试使用 VLC
            try:
                subprocess.Popen(["vlc", video_path, f"--start-time={int(timestamp_seconds)}"])
                return
            except:
                pass
            
            # 使用默认播放器
            QDesktopServices.openUrl(QUrl.fromLocalFile(video_path))
            QMessageBox.information(
                self, 
                "提示", 
                f"已打开视频，请手动跳转到 {self.format_timestamp(int(timestamp_seconds * 1000))}"
            )
    
    def open_image(self):
        """打开图像"""
        if not hasattr(self, "current_crop"):
            return
        
        image_path = self.current_crop["crop_image_path"]
        if not os.path.exists(image_path):
            QMessageBox.warning(self, "错误", f"图像文件不存在: {image_path}")
            return
        
        # 使用系统默认程序打开图像
        QDesktopServices.openUrl(QUrl.fromLocalFile(image_path))
    
    def delete_crop(self):
        """删除裁剪"""
        if not hasattr(self, "current_crop"):
            return
        
        reply = QMessageBox.question(
            self, 
            "确认删除", 
            "确定要删除这个裁剪吗？\n这将从数据库中删除记录，但不会删除图像文件。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            crop_id = self.current_crop["id"]
            success = self.database.delete_crop(crop_id)
            
            if success:
                QMessageBox.information(self, "成功", "裁剪已删除")
                self.refresh_results()
            else:
                QMessageBox.warning(self, "错误", "删除裁剪失败")
    
    def open_output_dir(self):
        """打开输出目录"""
        output_dir = self.config.get_output_dir()
        if not os.path.exists(output_dir):
            QMessageBox.warning(self, "错误", f"输出目录不存在: {output_dir}")
            return
        
        # 使用系统默认文件管理器打开目录
        QDesktopServices.openUrl(QUrl.fromLocalFile(output_dir))
    
    def resizeEvent(self, event):
        """
        窗口大小变化事件处理
        
        Args:
            event: 大小变化事件
        """
        super().resizeEvent(event)
        
        # 如果有裁剪结果，重新加载以适应新的窗口大小
        if self.crop_items:
            self.load_crops()
