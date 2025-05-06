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
    QTabWidget, QStackedWidget, QSizePolicy, QCheckBox, QProgressDialog,
    QStatusBar
)
from PySide6.QtCore import Qt, QSize, QUrl, QProcess, Signal, QThread, QTimer, QEvent, QPoint, QObject
from PySide6.QtGui import QPixmap, QIcon, QDesktopServices, QAction, QKeySequence, QCursor

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
        self.cache = {}  # 缓存已加载的缩略图

    def run(self):
        """运行线程"""
        # 优先加载可见区域的缩略图
        batch_size = 10  # 每批处理的数量

        # 先处理前20个（可能在可见区域内）
        visible_range = min(20, len(self.crop_items))
        for i in range(visible_range):
            if not self.running:
                break

            self.load_thumbnail(i)

        # 然后批量处理剩余的
        for i in range(visible_range, len(self.crop_items), batch_size):
            if not self.running:
                break

            # 处理一批
            for j in range(i, min(i + batch_size, len(self.crop_items))):
                if not self.running:
                    break

                self.load_thumbnail(j)

            # 短暂暂停，让UI有机会响应
            self.msleep(10)

    def load_thumbnail(self, index):
        """加载单个缩略图"""
        crop = self.crop_items[index]
        image_path = crop["crop_image_path"]

        # 检查缓存
        if image_path in self.cache:
            self.thumbnail_loaded.emit(index, self.cache[image_path])
            return

        try:
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                pixmap = pixmap.scaled(
                    self.thumbnail_size, self.thumbnail_size,
                    Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                # 缓存缩略图
                self.cache[image_path] = pixmap
                self.thumbnail_loaded.emit(index, pixmap)
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
        layout.setContentsMargins(0, 0, 0, 0)
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
        image_layout.setContentsMargins(0, 0, 0, 0)
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

        # 底部信息区域
        bottom_layout = QHBoxLayout()
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(2)

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
        bottom_layout.addWidget(self.timestamp_label, 1)  # 1表示伸展因子

        # 收藏标记
        self.favorite_label = QLabel()
        self.favorite_label.setFixedSize(16, 16)
        self.favorite_label.setAlignment(Qt.AlignCenter)
        self.update_favorite_status()
        bottom_layout.addWidget(self.favorite_label)

        layout.addLayout(bottom_layout)

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

    def update_favorite_status(self):
        """更新收藏状态标记"""
        is_favorite = self.crop_item.get("is_favorite", 0)
        if is_favorite:
            # 显示黄色星标
            self.favorite_label.setStyleSheet("""
                QLabel {
                    color: #FFD700;
                    font-size: 14px;
                    font-weight: bold;
                }
            """)
            self.favorite_label.setText("★")
        else:
            # 清空星标
            self.favorite_label.setText("")

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

        # 快捷键收藏相关
        self.favorite_shortcut = QKeySequence(self.config.get("ui", "favorite_shortcut", "Ctrl+S"))  # 从配置中读取快捷键
        self.favorite_timer = QTimer(self)  # 用于长按收藏
        self.favorite_timer.setInterval(200)  # 200毫秒间隔
        self.favorite_timer.timeout.connect(self.favorite_current_under_cursor)
        self.is_favorite_key_pressed = False

        # 安装事件过滤器
        self.installEventFilter(self)

        self.init_ui()

    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
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
        control_layout.setContentsMargins(0, 0, 0, 0)
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

        # 收藏筛选
        favorite_widget = QWidget()
        favorite_layout = QHBoxLayout(favorite_widget)
        favorite_layout.setContentsMargins(0, 0, 0, 0)
        favorite_layout.setSpacing(8)

        favorite_label = QLabel("收藏:")
        favorite_label.setStyleSheet("font-weight: 500;")
        favorite_layout.addWidget(favorite_label)

        self.favorite_check = QCheckBox("只显示收藏")
        self.favorite_check.stateChanged.connect(self.on_favorite_filter_changed)
        favorite_layout.addWidget(self.favorite_check)

        left_layout.addWidget(favorite_widget)

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

        # 下载收藏按钮
        self.download_favorites_button = QPushButton("下载收藏图片")
        self.download_favorites_button.setProperty("class", "success")
        self.download_favorites_button.clicked.connect(self.download_favorites)
        right_layout.addWidget(self.download_favorites_button)

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
        thumbnails_layout_main.setContentsMargins(0, 0, 0, 0)

        self.thumbnails_widget = QWidget()
        self.thumbnails_layout = QGridLayout(self.thumbnails_widget)
        self.thumbnails_layout.setContentsMargins(0, 0, 0, 0)
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
        details_form.setContentsMargins(0, 0, 0, 0)
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
        preview_group_layout.setContentsMargins(0, 0, 0, 0)
        preview_group_layout.addWidget(self.stacked_widget)

        details_content_layout.addWidget(preview_group)

        # 操作组
        actions_group = QGroupBox("操作")
        actions_layout = QVBoxLayout(actions_group)
        actions_layout.setContentsMargins(0, 0, 0, 0)
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

        # 收藏按钮
        self.toggle_favorite_button = QPushButton("收藏/取消收藏")
        self.toggle_favorite_button.clicked.connect(self.toggle_favorite)
        button_grid.addWidget(self.toggle_favorite_button, 2, 0, 1, 2)  # 跨两列

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

        # 状态和分页区域
        status_group = QGroupBox("状态")
        # 设置大小策略为垂直最小
        status_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        # 设置最大高度
        status_group.setMaximumHeight(150)
        status_layout = QVBoxLayout(status_group)
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(2)

        # 状态标签
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("font-weight: 500;")
        status_layout.addWidget(self.status_label)

        # 创建状态栏（用于显示临时消息）
        self.status_bar = QStatusBar()
        self.status_bar.setSizeGripEnabled(False)
        status_layout.addWidget(self.status_bar)

        # 分页导航
        pagination_widget = QWidget()
        pagination_layout = QHBoxLayout(pagination_widget)
        pagination_layout.setContentsMargins(0, 0, 0, 0)
        pagination_layout.setSpacing(8)

        # 首页按钮
        self.first_page_button = QPushButton("首页")
        self.first_page_button.clicked.connect(lambda: self.navigate_to_page(0))
        pagination_layout.addWidget(self.first_page_button)

        # 上一页按钮
        self.prev_page_button = QPushButton("上一页")
        self.prev_page_button.clicked.connect(self.go_to_prev_page)
        pagination_layout.addWidget(self.prev_page_button)

        # 页码显示
        self.page_label = QLabel("第 0/0 页")
        self.page_label.setAlignment(Qt.AlignCenter)
        self.page_label.setMinimumWidth(100)
        pagination_layout.addWidget(self.page_label)

        # 下一页按钮
        self.next_page_button = QPushButton("下一页")
        self.next_page_button.clicked.connect(self.go_to_next_page)
        pagination_layout.addWidget(self.next_page_button)

        # 末页按钮
        self.last_page_button = QPushButton("末页")
        self.last_page_button.clicked.connect(lambda: self.navigate_to_page(-1))
        pagination_layout.addWidget(self.last_page_button)

        # 添加分页导航到状态布局
        status_layout.addWidget(pagination_widget)

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

        # 添加收藏/取消收藏菜单项
        self.action_toggle_favorite = QAction("收藏", self)
        self.action_toggle_favorite.triggered.connect(self.toggle_favorite)
        self.context_menu.addAction(self.action_toggle_favorite)

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

        # 设置页面大小和当前页
        self.page_size = 100  # 每页显示的项目数
        self.current_page = 0  # 当前页码，从0开始

        # 获取总记录数
        if hasattr(self, 'favorite_check') and self.favorite_check.isChecked():
            # 只显示收藏
            self.total_items = self.database.count_favorite_crops()
            self.filter_type = "favorite"
            self.filter_value = None
        elif hasattr(self, 'current_group') and self.current_group is not None:
            # 按人脸分组筛选
            self.total_items = self.database.count_crops(group_id=self.current_group)
            self.filter_type = "group"
            self.filter_value = self.current_group
        else:
            # 按视频筛选
            self.total_items = self.database.count_crops(video_path=self.current_video)
            self.filter_type = "video"
            self.filter_value = self.current_video

        # 计算总页数
        self.total_pages = (self.total_items + self.page_size - 1) // self.page_size

        if self.total_items == 0:
            self.status_label.setText("没有找到裁剪结果")
            return

        # 加载当前页的数据
        self.load_page(self.current_page)

        # 更新状态标签
        self.update_status_label()

    def load_page(self, page):
        """
        加载指定页的数据

        Args:
            page: 页码，从0开始
        """
        # 停止之前的缩略图加载线程
        if self.thumbnail_loader and self.thumbnail_loader.isRunning():
            self.thumbnail_loader.stop()
            self.thumbnail_loader.wait()

        # 计算偏移量
        offset = page * self.page_size

        # 从数据库加载裁剪结果
        if self.filter_type == "favorite":
            # 按收藏筛选
            self.crop_items = self.database.get_favorite_crops(
                limit=self.page_size,
                offset=offset
            )
        elif self.filter_type == "group":
            # 按人脸分组筛选
            self.crop_items = self.database.get_crops_by_group(
                self.filter_value,
                limit=self.page_size,
                offset=offset
            )
        else:
            # 按视频筛选
            self.crop_items = self.database.get_crops(
                self.filter_value,
                limit=self.page_size,
                offset=offset
            )

        if not self.crop_items:
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

        # 更新当前页码
        self.current_page = page

        # 更新状态标签
        self.update_status_label()

    def update_status_label(self):
        """更新状态标签"""
        # 计算当前显示的项目范围
        start_item = self.current_page * self.page_size + 1
        end_item = min(start_item + self.page_size - 1, self.total_items)

        status_text = f"显示 {start_item}-{end_item} / 共 {self.total_items} 个裁剪结果"

        # 添加分页信息
        status_text += f" (第 {self.current_page + 1}/{self.total_pages} 页)"

        # 添加筛选信息
        if self.filter_type == "favorite":
            status_text += " (仅显示收藏)"
        elif self.filter_type == "video" and self.filter_value:
            video_name = os.path.basename(self.filter_value)
            status_text += f" (视频: {video_name})"
        elif self.filter_type == "group" and self.filter_value:
            group_name = self.group_combo.currentText()
            status_text += f" (人物: {group_name})"

        self.status_label.setText(status_text)

        # 更新页码标签
        self.page_label.setText(f"第 {self.current_page + 1}/{self.total_pages} 页")

        # 更新分页按钮状态
        self.first_page_button.setEnabled(self.current_page > 0)
        self.prev_page_button.setEnabled(self.current_page > 0)
        self.next_page_button.setEnabled(self.current_page < self.total_pages - 1)
        self.last_page_button.setEnabled(self.current_page < self.total_pages - 1)

    def navigate_to_page(self, page):
        """
        导航到指定页面

        Args:
            page: 页码，从0开始，如果为-1则表示最后一页
        """
        if page == -1:
            page = self.total_pages - 1

        if page < 0 or page >= self.total_pages:
            return

        if page == self.current_page:
            return

        # 清空缩略图区域
        self.clear_thumbnails()

        # 加载新页面
        self.load_page(page)

    def go_to_prev_page(self):
        """导航到上一页"""
        if self.current_page > 0:
            self.navigate_to_page(self.current_page - 1)

    def go_to_next_page(self):
        """导航到下一页"""
        if self.current_page < self.total_pages - 1:
            self.navigate_to_page(self.current_page + 1)

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
        self.toggle_favorite_button.setEnabled(False)

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
        self.toggle_favorite_button.setEnabled(True)

        # 更新收藏按钮文本
        is_favorite = crop_item.get("is_favorite", 0)
        if is_favorite:
            self.toggle_favorite_button.setText("取消收藏")
            self.toggle_favorite_button.setProperty("class", "")
        else:
            self.toggle_favorite_button.setText("收藏")
            self.toggle_favorite_button.setProperty("class", "success")

        # 刷新样式
        self.toggle_favorite_button.style().unpolish(self.toggle_favorite_button)
        self.toggle_favorite_button.style().polish(self.toggle_favorite_button)

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

    def on_favorite_filter_changed(self, state):
        """
        收藏筛选变化处理

        Args:
            state: 复选框状态
        """
        # 重新加载裁剪结果
        self.load_crops()

    def toggle_favorite(self):
        """切换当前裁剪项的收藏状态"""
        if not hasattr(self, "current_crop"):
            return

        crop_id = self.current_crop["id"]

        # 切换收藏状态
        new_status = self.database.toggle_favorite(crop_id)

        if new_status is not None:
            # 更新当前裁剪项的收藏状态
            self.current_crop["is_favorite"] = new_status

            # 更新UI
            self.update_details(self.current_crop)

            # 如果当前是收藏筛选模式，可能需要刷新列表
            if hasattr(self, 'favorite_check') and self.favorite_check.isChecked():
                self.refresh_results()
            else:
                # 找到并更新对应的缩略图
                for i in range(self.thumbnails_layout.count()):
                    item = self.thumbnails_layout.itemAt(i)
                    if item and item.widget():
                        thumbnail = item.widget()
                        if thumbnail.crop_item["id"] == crop_id:
                            thumbnail.crop_item["is_favorite"] = new_status
                            thumbnail.update_favorite_status()
                            break

            # 显示提示
            status = "收藏" if new_status else "取消收藏"
            self.status_bar.showMessage(f"已{status}当前图片", 3000)

    def download_favorites(self):
        """下载所有收藏的图片"""
        # 获取所有收藏的裁剪项
        favorite_crops = self.database.get_favorite_crops()

        if not favorite_crops:
            QMessageBox.information(self, "提示", "没有收藏的图片")
            return

        # 选择保存目录
        dir_dialog = QFileDialog()
        dir_dialog.setFileMode(QFileDialog.Directory)
        dir_dialog.setOption(QFileDialog.ShowDirsOnly, True)

        if not dir_dialog.exec():
            return

        save_dir = dir_dialog.selectedFiles()[0]
        if not save_dir:
            return

        # 创建收藏文件夹
        favorites_dir = os.path.join(save_dir, "收藏图片")
        os.makedirs(favorites_dir, exist_ok=True)

        # 创建进度对话框
        progress_dialog = QProgressDialog("正在下载收藏图片...", "取消", 0, len(favorite_crops), self)
        progress_dialog.setWindowTitle("下载收藏")
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setValue(0)

        # 复制图片
        copied_count = 0
        for i, crop in enumerate(favorite_crops):
            # 更新进度
            progress_dialog.setValue(i)
            QApplication.processEvents()

            if progress_dialog.wasCanceled():
                break

            # 源文件路径
            src_path = crop["crop_image_path"]
            if not os.path.exists(src_path):
                continue

            # 目标文件路径
            filename = os.path.basename(src_path)
            dst_path = os.path.join(favorites_dir, filename)

            # 如果文件已存在，添加序号
            if os.path.exists(dst_path):
                name, ext = os.path.splitext(filename)
                dst_path = os.path.join(favorites_dir, f"{name}_{i}{ext}")

            try:
                # 复制文件
                import shutil
                shutil.copy2(src_path, dst_path)
                copied_count += 1
            except Exception as e:
                print(f"复制文件失败: {e}")

        # 完成
        progress_dialog.setValue(len(favorite_crops))

        # 显示结果
        QMessageBox.information(
            self,
            "下载完成",
            f"已成功下载 {copied_count}/{len(favorite_crops)} 张收藏图片到:\n{favorites_dir}"
        )

        # 打开目标文件夹
        QDesktopServices.openUrl(QUrl.fromLocalFile(favorites_dir))

    def eventFilter(self, obj, event):
        """
        事件过滤器，用于处理快捷键收藏功能

        Args:
            obj: 事件源对象
            event: 事件对象

        Returns:
            是否已处理事件
        """
        # 处理键盘事件
        if event.type() == QEvent.KeyPress:
            # 检查是否是收藏快捷键
            # 使用字符串形式的QKeySequence构造函数
            # 创建一个临时的QKeySequence来获取当前按键的字符串表示
            temp_seq = QKeySequence(event.key())
            # 获取修饰符的字符串表示
            modifiers_text = ""
            modifiers = event.modifiers()
            if modifiers & Qt.ControlModifier:
                modifiers_text += "Ctrl+"
            if modifiers & Qt.ShiftModifier:
                modifiers_text += "Shift+"
            if modifiers & Qt.AltModifier:
                modifiers_text += "Alt+"
            if modifiers & Qt.MetaModifier:
                modifiers_text += "Meta+"

            # 组合修饰符和键
            key_text = temp_seq.toString()
            if not key_text:
                return False  # 如果没有有效的键，直接返回

            current_key = QKeySequence(modifiers_text + key_text)

            # 比较当前按键与配置的快捷键
            if current_key.matches(self.favorite_shortcut) == QKeySequence.ExactMatch:
                self.is_favorite_key_pressed = True
                self.favorite_current_under_cursor()
                # 启动定时器，实现长按连续收藏
                self.favorite_timer.start()
                return True
        elif event.type() == QEvent.KeyRelease:
            # 检查是否释放了收藏快捷键
            if self.is_favorite_key_pressed:
                # 任何键释放时，如果收藏键已按下，则停止收藏
                self.is_favorite_key_pressed = False
                # 停止定时器
                self.favorite_timer.stop()
                return True

        # 继续传递事件
        return super().eventFilter(obj, event)

    def favorite_current_under_cursor(self):
        """收藏当前鼠标指向的图片"""
        # 获取当前鼠标位置
        cursor_pos = QCursor.pos()
        # 转换为缩略图区域的坐标
        thumbnails_pos = self.thumbnails_widget.mapFromGlobal(cursor_pos)

        # 检查鼠标是否在缩略图区域内
        if not self.thumbnails_widget.rect().contains(thumbnails_pos):
            return

        # 查找鼠标下方的缩略图
        for i in range(self.thumbnails_layout.count()):
            item = self.thumbnails_layout.itemAt(i)
            if item and item.widget():
                thumbnail = item.widget()
                # 检查鼠标是否在这个缩略图上
                thumbnail_pos = thumbnail.mapFromGlobal(cursor_pos)
                if thumbnail.rect().contains(thumbnail_pos):
                    # 找到了鼠标下方的缩略图，切换收藏状态
                    crop_id = thumbnail.crop_item["id"]
                    new_status = self.database.toggle_favorite(crop_id)

                    if new_status is not None:
                        # 更新缩略图的收藏状态
                        thumbnail.crop_item["is_favorite"] = new_status
                        thumbnail.update_favorite_status()

                        # 如果当前是收藏筛选模式，可能需要刷新列表
                        if hasattr(self, 'favorite_check') and self.favorite_check.isChecked():
                            # 延迟刷新，避免在长按时频繁刷新
                            if not hasattr(self, 'refresh_timer'):
                                self.refresh_timer = QTimer(self)
                                self.refresh_timer.setSingleShot(True)
                                self.refresh_timer.timeout.connect(self.refresh_results)

                            # 如果定时器已经在运行，重置它
                            if self.refresh_timer.isActive():
                                self.refresh_timer.stop()

                            # 设置500毫秒后刷新
                            self.refresh_timer.start(500)

                        # 显示提示
                        status = "收藏" if new_status else "取消收藏"
                        self.status_bar.showMessage(f"已{status}图片: {os.path.basename(thumbnail.crop_item['crop_image_path'])}", 2000)

                    # 找到并处理了一个缩略图，退出循环
                    break

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
        delete_empty_button = QPushButton("删除空白分组")
        close_button = QPushButton("关闭")

        button_layout.addWidget(rename_button)
        button_layout.addWidget(delete_button)
        button_layout.addWidget(merge_button)

        # 添加第二行按钮
        second_button_layout = QHBoxLayout()
        second_button_layout.addWidget(delete_empty_button)
        second_button_layout.addWidget(close_button)

        layout.addLayout(button_layout)
        layout.addLayout(second_button_layout)

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

        # 添加删除空白分组的功能
        def on_delete_empty_groups():
            # 获取空白分组
            empty_groups = self.database.get_empty_face_groups()

            if not empty_groups:
                QMessageBox.information(dialog, "提示", "没有找到空白分组")
                return

            # 确认删除
            reply = QMessageBox.question(
                dialog,
                "确认删除空白分组",
                f"确定要删除所有空白分组吗？共找到 {len(empty_groups)} 个空白分组。",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                # 删除所有空白分组
                deleted_count = self.database.delete_empty_face_groups()

                QMessageBox.information(
                    dialog,
                    "删除完成",
                    f"已删除 {deleted_count} 个空白分组"
                )

                # 刷新列表
                self.refresh_results()
                dialog.accept()

        # 连接信号
        group_list.itemSelectionChanged.connect(on_group_selected)
        rename_button.clicked.connect(on_rename)
        delete_button.clicked.connect(on_delete)
        merge_button.clicked.connect(on_merge)
        delete_empty_button.clicked.connect(on_delete_empty_groups)
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
        # 使用延迟重新加载，避免频繁调整窗口大小时多次重新加载
        if hasattr(self, 'resize_timer'):
            self.resize_timer.stop()
        else:
            # 创建一个单次触发的定时器
            self.resize_timer = QTimer()
            self.resize_timer.setSingleShot(True)
            self.resize_timer.timeout.connect(self.on_resize_timeout)

        # 设置定时器，300毫秒后触发重新加载
        self.resize_timer.start(300)

    def on_resize_timeout(self):
        """窗口大小变化后的延迟处理"""
        # 如果有裁剪结果，重新加载当前页面
        if hasattr(self, 'crop_items') and self.crop_items:
            # 清空缩略图区域
            self.clear_thumbnails()

            # 重新加载当前页面
            self.load_page(self.current_page)
