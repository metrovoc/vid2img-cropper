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
    QScrollArea, QGridLayout, QMenu, QApplication, QDialog, QLineEdit,
    QTabWidget, QStackedWidget, QSizePolicy
)
from PySide6.QtCore import Qt, QSize, QUrl, QProcess, Signal, QThread
from PySide6.QtGui import QPixmap, QIcon, QDesktopServices, QAction

from src.ui.video_player import VideoPlayer


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
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # 缩略图容器
        image_container = QWidget()
        image_container.setFixedSize(self.thumbnail_size + 8, self.thumbnail_size + 8)
        image_container.setStyleSheet("""
            QWidget {
                background-color: transparent;
                border-radius: 4px;
            }
        """)

        image_layout = QVBoxLayout(image_container)
        image_layout.setContentsMargins(4, 4, 4, 4)
        image_layout.setAlignment(Qt.AlignCenter)

        # 缩略图标签
        self.image_label = QLabel()
        self.image_label.setFixedSize(self.thumbnail_size, self.thumbnail_size)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("""
            background-color: #f0f0f0;
            border-radius: 4px;
            border: 1px solid #ddd;
        """)

        # 加载缩略图
        self.load_thumbnail()

        image_layout.addWidget(self.image_label)
        layout.addWidget(image_container)

        # 时间戳标签
        timestamp_ms = self.crop_item["timestamp_ms"]
        timestamp_str = self.format_timestamp(timestamp_ms)

        self.timestamp_label = QLabel(timestamp_str)
        self.timestamp_label.setAlignment(Qt.AlignCenter)
        self.timestamp_label.setStyleSheet("""
            font-size: 11px;
            color: #555;
            font-weight: 500;
            padding: 2px;
        """)

        layout.addWidget(self.timestamp_label)

        # 设置整体样式
        self.setStyleSheet("""
            CropThumbnail {
                background-color: transparent;
                border-radius: 6px;
            }

            CropThumbnail:hover {
                background-color: rgba(0, 0, 0, 0.05);
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
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # 创建主分割器，允许用户调整各区域大小
        main_splitter = QSplitter(Qt.Vertical)
        main_splitter.setHandleWidth(2)
        main_splitter.setChildrenCollapsible(False)

        # 顶部控制区域
        control_group = QGroupBox("控制面板")
        # 设置大小策略为垂直最小
        control_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        # 设置最大高度
        control_group.setMaximumHeight(120)
        control_layout = QHBoxLayout(control_group)
        control_layout.setContentsMargins(12, 16, 12, 12)
        control_layout.setSpacing(12)

        # 左侧控制区域
        left_control = QWidget()
        left_layout = QHBoxLayout(left_control)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        # 视频选择
        video_select_widget = QWidget()
        video_select_layout = QHBoxLayout(video_select_widget)
        video_select_layout.setContentsMargins(0, 0, 0, 0)
        video_select_layout.setSpacing(8)

        video_label = QLabel("选择视频:")
        video_label.setStyleSheet("font-weight: 500;")
        video_select_layout.addWidget(video_label)

        self.video_combo = QComboBox()
        self.video_combo.setMinimumWidth(250)
        self.video_combo.currentIndexChanged.connect(self.on_video_changed)
        video_select_layout.addWidget(self.video_combo)

        left_layout.addWidget(video_select_widget)

        # 人脸分组选择
        group_select_widget = QWidget()
        group_select_layout = QHBoxLayout(group_select_widget)
        group_select_layout.setContentsMargins(0, 0, 0, 0)
        group_select_layout.setSpacing(8)

        group_label = QLabel("人物分组:")
        group_label.setStyleSheet("font-weight: 500;")
        group_select_layout.addWidget(group_label)

        self.group_combo = QComboBox()
        self.group_combo.setMinimumWidth(180)
        self.group_combo.currentIndexChanged.connect(self.on_group_changed)
        group_select_layout.addWidget(self.group_combo)

        left_layout.addWidget(group_select_widget)

        control_layout.addWidget(left_control)

        # 右侧按钮区域
        right_control = QWidget()
        right_layout = QHBoxLayout(right_control)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        # 批量删除按钮
        self.delete_video_button = QPushButton("删除当前视频所有裁剪")
        self.delete_video_button.setProperty("class", "danger")
        self.delete_video_button.clicked.connect(self.delete_video_crops)
        right_layout.addWidget(self.delete_video_button)

        # 刷新按钮
        self.refresh_button = QPushButton("刷新")
        self.refresh_button.clicked.connect(self.refresh_results)
        right_layout.addWidget(self.refresh_button)

        # 打开输出目录按钮
        self.open_output_dir_button = QPushButton("打开输出目录")
        self.open_output_dir_button.clicked.connect(self.open_output_dir)
        right_layout.addWidget(self.open_output_dir_button)

        # 管理分组按钮
        self.manage_groups_button = QPushButton("管理分组")
        self.manage_groups_button.clicked.connect(self.manage_face_groups)
        right_layout.addWidget(self.manage_groups_button)

        control_layout.addWidget(right_control)
        control_layout.setStretch(0, 3)  # 左侧占比更大
        control_layout.setStretch(1, 2)  # 右侧占比较小

        # 将控制面板添加到主分割器
        main_splitter.addWidget(control_group)

        # 主内容区域
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(12)

        # 分割器
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(2)
        splitter.setChildrenCollapsible(False)

        # 缩略图区域
        thumbnails_group = QGroupBox("缩略图")
        thumbnails_layout_main = QVBoxLayout(thumbnails_group)
        thumbnails_layout_main.setContentsMargins(12, 16, 12, 12)

        self.thumbnails_widget = QWidget()
        self.thumbnails_layout = QGridLayout(self.thumbnails_widget)
        self.thumbnails_layout.setContentsMargins(4, 4, 4, 4)
        self.thumbnails_layout.setSpacing(12)

        # 滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.thumbnails_widget)
        scroll_area.setFrameShape(QScrollArea.NoFrame)

        thumbnails_layout_main.addWidget(scroll_area)
        splitter.addWidget(thumbnails_group)

        # 详情区域
        details_widget = QWidget()
        details_widget.setObjectName("details_widget")
        details_layout = QVBoxLayout(details_widget)
        details_layout.setContentsMargins(0, 0, 0, 0)
        details_layout.setSpacing(12)

        # 创建滚动区域来包裹所有详情内容
        details_scroll = QScrollArea()
        details_scroll.setWidgetResizable(True)
        details_scroll.setFrameShape(QScrollArea.NoFrame)
        details_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        details_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # 创建内容小部件
        details_content = QWidget()
        details_content_layout = QVBoxLayout(details_content)
        details_content_layout.setContentsMargins(0, 0, 0, 0)
        details_content_layout.setSpacing(12)

        # 详情组
        details_group = QGroupBox("详细信息")
        details_group.setMinimumWidth(300)
        details_form = QFormLayout(details_group)
        details_form.setContentsMargins(12, 16, 12, 12)
        details_form.setSpacing(8)
        details_form.setLabelAlignment(Qt.AlignRight)

        self.detail_video_path = QLabel("")
        self.detail_video_path.setWordWrap(True)
        self.detail_video_path.setTextInteractionFlags(Qt.TextSelectableByMouse)
        details_form.addRow("视频路径:", self.detail_video_path)

        self.detail_timestamp = QLabel("")
        self.detail_timestamp.setTextInteractionFlags(Qt.TextSelectableByMouse)
        details_form.addRow("时间戳:", self.detail_timestamp)

        self.detail_image_path = QLabel("")
        self.detail_image_path.setWordWrap(True)
        self.detail_image_path.setTextInteractionFlags(Qt.TextSelectableByMouse)
        details_form.addRow("图像路径:", self.detail_image_path)

        self.detail_confidence = QLabel("")
        self.detail_confidence.setTextInteractionFlags(Qt.TextSelectableByMouse)
        details_form.addRow("置信度:", self.detail_confidence)

        details_content_layout.addWidget(details_group)

        # 创建堆叠小部件，用于切换预览和视频播放器
        self.stacked_widget = QStackedWidget()

        # 预览组
        preview_widget = QWidget()
        preview_layout = QVBoxLayout(preview_widget)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(0)

        self.preview_label = QLabel()
        self.preview_label.setObjectName("preview_label")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(300, 300)
        preview_layout.addWidget(self.preview_label)

        # 视频播放器组
        video_widget = QWidget()
        video_layout = QVBoxLayout(video_widget)
        video_layout.setContentsMargins(0, 0, 0, 0)
        video_layout.setSpacing(0)

        # 创建视频播放器
        self.video_player = VideoPlayer()
        self.video_player.setMinimumSize(300, 350)  # 增加高度以确保控制器可见
        self.video_player.playback_finished.connect(self.on_playback_finished)

        # 设置初始音量
        volume = self.config.get("ui", "player_volume", 50)
        self.video_player.set_volume(volume)

        video_layout.addWidget(self.video_player)

        # 添加到堆叠小部件
        self.stacked_widget.addWidget(preview_widget)  # 索引0 - 预览模式
        self.stacked_widget.addWidget(video_widget)    # 索引1 - 视频模式

        # 默认显示预览模式
        self.stacked_widget.setCurrentIndex(0)

        # 添加到布局
        preview_group = QGroupBox("预览/视频")
        preview_group.setMinimumHeight(400)  # 确保有足够的高度
        preview_group_layout = QVBoxLayout(preview_group)
        preview_group_layout.setContentsMargins(12, 16, 12, 12)
        preview_group_layout.addWidget(self.stacked_widget)

        details_content_layout.addWidget(preview_group)

        # 操作组
        actions_group = QGroupBox("操作")
        actions_layout = QVBoxLayout(actions_group)
        actions_layout.setContentsMargins(12, 16, 12, 12)
        actions_layout.setSpacing(8)

        # 创建按钮网格布局
        button_grid = QGridLayout()
        button_grid.setSpacing(8)

        self.open_video_button = QPushButton("在外部播放器中打开视频")
        self.open_video_button.clicked.connect(self.open_video)
        button_grid.addWidget(self.open_video_button, 0, 0)

        self.jump_to_time_button = QPushButton("播放视频并跳转到时间点")
        self.jump_to_time_button.setProperty("class", "success")
        self.jump_to_time_button.clicked.connect(self.jump_to_time)
        button_grid.addWidget(self.jump_to_time_button, 0, 1)

        self.open_image_button = QPushButton("打开图像")
        self.open_image_button.clicked.connect(self.open_image)
        button_grid.addWidget(self.open_image_button, 1, 0)

        self.delete_crop_button = QPushButton("删除裁剪")
        self.delete_crop_button.setProperty("class", "danger")
        self.delete_crop_button.clicked.connect(self.delete_crop)
        button_grid.addWidget(self.delete_crop_button, 1, 1)

        actions_layout.addLayout(button_grid)
        details_content_layout.addWidget(actions_group)

        # 设置滚动区域的内容
        details_scroll.setWidget(details_content)

        # 添加滚动区域到详情布局
        details_layout.addWidget(details_scroll)

        splitter.addWidget(details_widget)

        # 设置分割器比例
        splitter.setSizes([700, 300])

        content_layout.addWidget(splitter)

        # 将主内容区域添加到主分割器
        main_splitter.addWidget(content_widget)

        # 状态区域
        status_group = QGroupBox("状态")
        # 设置大小策略为垂直最小
        status_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        # 设置最大高度
        status_group.setMaximumHeight(60)
        status_layout = QVBoxLayout(status_group)
        status_layout.setContentsMargins(12, 16, 12, 12)
        status_layout.setSpacing(4)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("font-weight: 500;")
        status_layout.addWidget(self.status_label)

        # 将状态区域添加到主分割器
        main_splitter.addWidget(status_group)

        # 设置分割器初始比例 (控制面板:主内容:状态)
        main_splitter.setSizes([100, 600, 40])

        # 将主分割器添加到主布局
        layout.addWidget(main_splitter)

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

        self.action_open_video = QAction("在外部播放器中打开视频", self)
        self.action_open_video.triggered.connect(self.open_video)
        self.context_menu.addAction(self.action_open_video)

        self.action_jump_to_time = QAction("播放视频并跳转到时间点", self)
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

        # 加载人脸分组
        self.load_face_groups()

    def load_face_groups(self):
        """加载人脸分组列表"""
        self.group_combo.clear()

        # 添加"全部人物"选项
        self.group_combo.addItem("全部人物", None)

        # 获取所有人脸分组
        face_groups = self.database.get_face_groups()
        for group in face_groups:
            self.group_combo.addItem(group['name'], group['id'])

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
        self.current_group = self.group_combo.currentData()
        self.load_crops()

    def on_group_changed(self, index):
        """
        人脸分组选择变化处理

        Args:
            index: 选中的索引
        """
        self.current_group = self.group_combo.currentData()
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
        if hasattr(self, 'current_group') and self.current_group is not None:
            # 按人脸分组筛选
            self.crop_items = self.database.get_crops_by_group(self.current_group)
        else:
            # 按视频筛选
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

        # 更新状态标签
        status_text = f"找到 {len(self.crop_items)} 个裁剪结果"

        # 添加筛选信息
        if hasattr(self, 'current_video') and self.current_video:
            video_name = os.path.basename(self.current_video)
            status_text += f" (视频: {video_name})"

        if hasattr(self, 'current_group') and self.current_group:
            group_name = self.group_combo.currentText()
            status_text += f" (人物: {group_name})"

        self.status_label.setText(status_text)

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

        # 切换回预览模式
        self.stacked_widget.setCurrentIndex(0)

        # 如果视频播放器正在播放，停止播放
        if hasattr(self, "video_player"):
            self.video_player.stop()

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

        # 检查是否使用内嵌播放器
        use_embedded_player = self.config.get("ui", "use_embedded_player", True)

        if use_embedded_player:
            # 使用内嵌播放器播放视频并跳转到指定时间点
            if self.video_player.load_video(video_path):
                # 设置音量
                volume = self.config.get("ui", "player_volume", 50)
                self.video_player.set_volume(volume)

                # 切换到视频播放器模式
                self.stacked_widget.setCurrentIndex(1)

                # 先设置跳转位置，然后开始播放
                # 视频播放器会在媒体加载完成后自动跳转到指定位置
                self.video_player.seek_to(timestamp_ms)

                # 开始播放
                self.video_player.play()
                return
            else:
                # 如果内嵌播放器加载失败，尝试使用外部播放器
                QMessageBox.warning(self, "警告", "内嵌播放器加载视频失败，尝试使用外部播放器")

        # 使用外部播放器
        self.try_open_with_player(video_path, timestamp_ms / 1000)

    def try_open_with_player(self, video_path, timestamp_seconds):
        """
        尝试使用不同的播放器打开视频

        Args:
            video_path: 视频路径
            timestamp_seconds: 时间戳（秒）
        """
        # 获取配置中的默认播放器
        default_player = self.config.get("ui", "default_player", "")
        player_found = False

        # 尝试使用特定于平台的播放器
        if sys.platform == "darwin":  # macOS
            # 检查是否安装了IINA
            try:
                result = subprocess.run(["which", "iina"], capture_output=True, text=True)
                iina_path = result.stdout.strip()
                if iina_path:
                    try:
                        subprocess.Popen(["iina", "--mpv-start=" + str(timestamp_seconds), video_path])
                        player_found = True
                    except Exception as e:
                        print(f"使用IINA打开视频失败: {e}")
            except Exception:
                pass

            # 如果IINA不可用，检查是否安装了VLC
            if not player_found:
                try:
                    result = subprocess.run(["which", "vlc"], capture_output=True, text=True)
                    vlc_path = result.stdout.strip()
                    if vlc_path:
                        try:
                            subprocess.Popen(["vlc", "--start-time=" + str(int(timestamp_seconds)), video_path])
                            player_found = True
                        except Exception as e:
                            print(f"使用VLC打开视频失败: {e}")
                except Exception:
                    pass

        elif sys.platform == "win32":  # Windows
            # 检查是否安装了VLC
            vlc_paths = [
                "C:\\Program Files\\VideoLAN\\VLC\\vlc.exe",
                "C:\\Program Files (x86)\\VideoLAN\\VLC\\vlc.exe"
            ]

            for path in vlc_paths:
                if os.path.exists(path):
                    try:
                        subprocess.Popen([path, "--start-time=" + str(int(timestamp_seconds)), video_path])
                        player_found = True
                        break
                    except Exception as e:
                        print(f"使用VLC打开视频失败: {e}")

        else:  # Linux 或其他
            # 检查是否安装了VLC
            try:
                result = subprocess.run(["which", "vlc"], capture_output=True, text=True)
                vlc_path = result.stdout.strip()
                if vlc_path:
                    try:
                        subprocess.Popen(["vlc", "--start-time=" + str(int(timestamp_seconds)), video_path])
                        player_found = True
                    except Exception as e:
                        print(f"使用VLC打开视频失败: {e}")
            except Exception:
                pass

        # 如果没有找到专用播放器，尝试使用配置的默认播放器
        if not player_found and default_player and os.path.exists(default_player):
            try:
                subprocess.Popen([default_player, video_path])
                QMessageBox.information(
                    self,
                    "提示",
                    f"已使用自定义播放器打开视频，请手动跳转到 {self.format_timestamp(int(timestamp_seconds * 1000))}"
                )
                player_found = True
            except Exception as e:
                print(f"使用自定义播放器打开视频失败: {e}")

        # 如果所有尝试都失败，使用系统默认播放器
        if not player_found:
            QDesktopServices.openUrl(QUrl.fromLocalFile(video_path))
            QMessageBox.information(
                self,
                "提示",
                f"已打开视频，请手动跳转到 {self.format_timestamp(int(timestamp_seconds * 1000))}"
            )

    def on_playback_finished(self):
        """视频播放完成处理"""
        # 切换回预览模式
        self.stacked_widget.setCurrentIndex(0)

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
            "确定要删除这个裁剪吗？\n这将从数据库中删除记录，并同时删除图像文件。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            crop_id = self.current_crop["id"]
            image_path = self.current_crop["crop_image_path"]

            # 先删除图像文件
            try:
                if os.path.exists(image_path):
                    os.remove(image_path)
            except Exception as e:
                print(f"删除图像文件失败: {e}")
                QMessageBox.warning(self, "警告", f"删除图像文件失败: {e}")

            # 再删除数据库记录
            success = self.database.delete_crop(crop_id)

            if success:
                QMessageBox.information(self, "成功", "裁剪已删除")
                self.refresh_results()
            else:
                QMessageBox.warning(self, "错误", "删除裁剪失败")

    def delete_video_crops(self):
        """删除当前视频的所有裁剪"""
        # 检查是否选择了视频
        video_path = self.video_combo.currentData()
        if not video_path:
            QMessageBox.warning(self, "警告", "请先选择一个视频")
            return

        # 确认删除
        video_name = os.path.basename(video_path)
        reply = QMessageBox.question(
            self,
            "确认批量删除",
            f"确定要删除视频 '{video_name}' 的所有裁剪吗？\n这将从数据库中删除所有相关记录，并同时删除所有图像文件。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        # 获取该视频的所有裁剪
        crops = self.database.get_crops(video_path)
        if not crops:
            QMessageBox.information(self, "提示", "没有找到该视频的裁剪")
            return

        # 删除所有图像文件
        deleted_files = 0
        for crop in crops:
            image_path = crop["crop_image_path"]
            try:
                if os.path.exists(image_path):
                    os.remove(image_path)
                    deleted_files += 1
            except Exception as e:
                print(f"删除图像文件失败: {e}")

        # 删除数据库记录
        deleted_records = self.database.delete_crops_by_video(video_path)

        # 显示结果
        QMessageBox.information(
            self,
            "删除完成",
            f"已删除 {deleted_records} 条记录和 {deleted_files} 个图像文件"
        )

        # 刷新结果
        self.refresh_results()

    def open_output_dir(self):
        """打开输出目录"""
        output_dir = self.config.get_output_dir()
        if not os.path.exists(output_dir):
            QMessageBox.warning(self, "错误", f"输出目录不存在: {output_dir}")
            return

        # 使用系统默认文件管理器打开目录
        QDesktopServices.openUrl(QUrl.fromLocalFile(output_dir))

    def manage_face_groups(self):
        """管理人脸分组"""
        # 获取所有人脸分组
        face_groups = self.database.get_face_groups()

        if not face_groups:
            QMessageBox.information(self, "人脸分组", "当前没有人脸分组。处理视频时启用自动人脸分组功能可以自动创建分组。")
            return

        # 创建对话框
        dialog = QDialog(self)
        dialog.setWindowTitle("管理人脸分组")
        dialog.setMinimumWidth(400)

        layout = QVBoxLayout(dialog)

        # 分组列表
        group_list = QListWidget()
        for group in face_groups:
            item = QListWidgetItem(f"{group['name']} ({self.database.count_crops(group_id=group['id'])} 张图片)")
            item.setData(Qt.UserRole, group['id'])
            group_list.addItem(item)

        layout.addWidget(QLabel("人脸分组:"))
        layout.addWidget(group_list)

        # 编辑区域
        edit_group = QGroupBox("编辑分组")
        edit_layout = QFormLayout(edit_group)

        name_edit = QLineEdit()
        edit_layout.addRow("分组名称:", name_edit)

        layout.addWidget(edit_group)

        # 按钮区域
        button_layout = QHBoxLayout()

        rename_button = QPushButton("重命名")
        delete_button = QPushButton("删除分组")
        merge_button = QPushButton("合并分组")
        close_button = QPushButton("关闭")

        button_layout.addWidget(rename_button)
        button_layout.addWidget(delete_button)
        button_layout.addWidget(merge_button)
        button_layout.addWidget(close_button)

        layout.addLayout(button_layout)

        # 连接信号
        def on_group_selected():
            selected_items = group_list.selectedItems()
            if selected_items:
                group_id = selected_items[0].data(Qt.UserRole)
                group = self.database.get_face_group(group_id)
                if group:
                    name_edit.setText(group['name'])
                    rename_button.setEnabled(True)
                    delete_button.setEnabled(True)
                    merge_button.setEnabled(True)
            else:
                name_edit.clear()
                rename_button.setEnabled(False)
                delete_button.setEnabled(False)
                merge_button.setEnabled(False)

        def on_rename():
            selected_items = group_list.selectedItems()
            if selected_items and name_edit.text().strip():
                group_id = selected_items[0].data(Qt.UserRole)
                new_name = name_edit.text().strip()

                # 更新分组名称
                if self.database.update_face_group(group_id, name=new_name):
                    QMessageBox.information(dialog, "成功", "分组已重命名")
                    # 刷新列表
                    self.refresh_results()
                    dialog.accept()
                else:
                    QMessageBox.warning(dialog, "错误", "重命名分组失败")

        def on_delete():
            selected_items = group_list.selectedItems()
            if selected_items:
                group_id = selected_items[0].data(Qt.UserRole)

                reply = QMessageBox.question(
                    dialog,
                    "确认删除",
                    "确定要删除此分组吗？分组中的图片将不会被删除，但会失去分组信息。",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )

                if reply == QMessageBox.Yes:
                    # 删除分组
                    if self.database.delete_face_group(group_id):
                        QMessageBox.information(dialog, "成功", "分组已删除")
                        # 刷新列表
                        self.refresh_results()
                        dialog.accept()
                    else:
                        QMessageBox.warning(dialog, "错误", "删除分组失败")

        def on_merge():
            selected_items = group_list.selectedItems()
            if selected_items:
                source_group_id = selected_items[0].data(Qt.UserRole)

                # 创建合并对话框
                merge_dialog = QDialog(dialog)
                merge_dialog.setWindowTitle("合并分组")

                merge_layout = QVBoxLayout(merge_dialog)
                merge_layout.addWidget(QLabel("选择要合并到的目标分组:"))

                target_combo = QComboBox()
                for group in face_groups:
                    if group['id'] != source_group_id:
                        target_combo.addItem(group['name'], group['id'])

                if target_combo.count() == 0:
                    QMessageBox.information(dialog, "提示", "没有其他分组可供合并")
                    return

                merge_layout.addWidget(target_combo)

                merge_buttons = QHBoxLayout()
                confirm_button = QPushButton("确认合并")
                cancel_button = QPushButton("取消")

                merge_buttons.addWidget(confirm_button)
                merge_buttons.addWidget(cancel_button)

                merge_layout.addLayout(merge_buttons)

                # 连接信号
                confirm_button.clicked.connect(merge_dialog.accept)
                cancel_button.clicked.connect(merge_dialog.reject)

                if merge_dialog.exec():
                    target_group_id = target_combo.currentData()

                    # 获取源分组的所有裁剪记录
                    crops = self.database.get_crops_by_group(source_group_id)

                    # 将所有裁剪记录分配到目标分组
                    success_count = 0
                    for crop in crops:
                        if self.database.assign_crop_to_group(crop['id'], target_group_id):
                            success_count += 1

                    # 删除源分组
                    self.database.delete_face_group(source_group_id)

                    QMessageBox.information(
                        dialog,
                        "合并完成",
                        f"已将 {success_count}/{len(crops)} 张图片合并到目标分组"
                    )

                    # 刷新列表
                    self.refresh_results()
                    dialog.accept()

        # 初始状态
        rename_button.setEnabled(False)
        delete_button.setEnabled(False)
        merge_button.setEnabled(False)

        # 连接信号
        group_list.itemSelectionChanged.connect(on_group_selected)
        rename_button.clicked.connect(on_rename)
        delete_button.clicked.connect(on_delete)
        merge_button.clicked.connect(on_merge)
        close_button.clicked.connect(dialog.accept)

        # 显示对话框
        dialog.exec()

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
