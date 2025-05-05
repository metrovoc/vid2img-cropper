"""
内嵌视频播放器组件
"""
import os
from datetime import timedelta

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSlider, QStyle, QSizePolicy
)
from PySide6.QtCore import Qt, QUrl, QTimer, Signal, QSize
from PySide6.QtGui import QIcon
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget


class VideoPlayer(QWidget):
    """视频播放器组件"""

    # 自定义信号
    playback_finished = Signal()

    def __init__(self, parent=None):
        """
        初始化视频播放器

        Args:
            parent: 父组件
        """
        super().__init__(parent)

        # 创建媒体播放器
        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)

        # 设置音量
        self.audio_output.setVolume(0.5)

        # 创建视频显示组件
        self.video_widget = QVideoWidget()
        self.video_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.media_player.setVideoOutput(self.video_widget)

        # 连接信号
        self.media_player.playbackStateChanged.connect(self.on_playback_state_changed)
        self.media_player.positionChanged.connect(self.on_position_changed)
        self.media_player.durationChanged.connect(self.on_duration_changed)
        self.media_player.errorOccurred.connect(self.on_error)
        self.media_player.mediaStatusChanged.connect(self.on_media_status_changed)

        # 初始化UI
        self.init_ui()

        # 更新定时器
        self.update_timer = QTimer(self)
        self.update_timer.setInterval(100)  # 100毫秒更新一次
        self.update_timer.timeout.connect(self.update_ui)

        # 初始状态
        self.video_path = None
        self.duration = 0
        self.pending_seek_position = None

    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 设置整体大小策略
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # 视频显示区域
        self.video_widget.setStyleSheet("""
            background-color: #000000;
        """)
        self.video_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.video_widget, 1)  # 设置拉伸因子为1，使视频区域占据所有可用空间

        # 控制面板容器
        controls_container = QWidget()
        controls_container.setStyleSheet("""
            background-color: rgba(0, 0, 0, 0.7);
            border-top: 1px solid rgba(255, 255, 255, 0.1);
        """)
        # 设置固定高度确保控制器始终可见且不会溢出
        controls_container.setFixedHeight(80)
        controls_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        controls_layout = QVBoxLayout(controls_container)
        controls_layout.setContentsMargins(8, 4, 8, 8)
        controls_layout.setSpacing(4)

        # 进度条
        self.position_slider = QSlider(Qt.Horizontal)
        self.position_slider.setRange(0, 0)
        self.position_slider.setMinimumHeight(20)  # 增加滑块高度使其更容易点击
        self.position_slider.sliderMoved.connect(self.set_position)
        self.position_slider.sliderPressed.connect(self.on_slider_pressed)
        self.position_slider.sliderReleased.connect(self.on_slider_released)
        self.position_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 6px;
                background: rgba(255, 255, 255, 0.2);
                border-radius: 3px;
                border: none;
            }

            QSlider::handle:horizontal {
                background: #2979ff;
                border: none;
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }

            QSlider::sub-page:horizontal {
                background: #2979ff;
                border-radius: 3px;
            }
        """)
        controls_layout.addWidget(self.position_slider)

        # 底部控制区域
        bottom_controls = QHBoxLayout()
        bottom_controls.setContentsMargins(0, 0, 0, 0)
        bottom_controls.setSpacing(8)

        # 播放/暂停按钮
        self.play_button = QPushButton()
        self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.play_button.clicked.connect(self.toggle_play)
        self.play_button.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                color: white;
                padding: 4px;
                border-radius: 16px;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 0.1);
            }
        """)
        self.play_button.setIconSize(QSize(28, 28))
        self.play_button.setFixedSize(36, 36)
        bottom_controls.addWidget(self.play_button)

        # 停止按钮
        self.stop_button = QPushButton()
        self.stop_button.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
        self.stop_button.clicked.connect(self.stop)
        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                color: white;
                padding: 4px;
                border-radius: 16px;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 0.1);
            }
        """)
        self.stop_button.setIconSize(QSize(28, 28))
        self.stop_button.setFixedSize(36, 36)
        bottom_controls.addWidget(self.stop_button)

        # 当前时间
        self.time_label = QLabel("00:00:00 / 00:00:00")
        self.time_label.setStyleSheet("""
            color: white;
            font-size: 12px;
            padding: 0 8px;
        """)
        bottom_controls.addWidget(self.time_label)

        # 弹性空间
        bottom_controls.addStretch()

        # 音量控制区域
        volume_layout = QHBoxLayout()
        volume_layout.setContentsMargins(0, 0, 0, 0)
        volume_layout.setSpacing(4)

        # 音量按钮
        self.volume_button = QPushButton()
        self.volume_button.setIcon(self.style().standardIcon(QStyle.SP_MediaVolume))
        self.volume_button.clicked.connect(self.toggle_mute)
        self.volume_button.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                color: white;
                padding: 4px;
                border-radius: 16px;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 0.1);
            }
        """)
        self.volume_button.setIconSize(QSize(24, 24))
        self.volume_button.setFixedSize(32, 32)
        volume_layout.addWidget(self.volume_button)

        # 音量滑块
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(50)
        self.volume_slider.setMaximumWidth(80)
        self.volume_slider.setMinimumHeight(20)  # 增加滑块高度使其更容易点击
        self.volume_slider.valueChanged.connect(self.set_volume)
        self.volume_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 4px;
                background: rgba(255, 255, 255, 0.2);
                border-radius: 2px;
                border: none;
            }

            QSlider::handle:horizontal {
                background: white;
                border: none;
                width: 12px;
                height: 12px;
                margin: -4px 0;
                border-radius: 6px;
            }

            QSlider::sub-page:horizontal {
                background: white;
                border-radius: 2px;
            }
        """)
        volume_layout.addWidget(self.volume_slider)

        bottom_controls.addLayout(volume_layout)
        controls_layout.addLayout(bottom_controls)

        layout.addWidget(controls_container)

    def load_video(self, video_path):
        """
        加载视频

        Args:
            video_path: 视频文件路径
        """
        if not os.path.exists(video_path):
            return False

        # 重置待处理的跳转位置
        self.pending_seek_position = None

        self.video_path = video_path
        self.media_player.setSource(QUrl.fromLocalFile(video_path))
        self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

        return True

    def play(self):
        """播放视频"""
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            return

        self.media_player.play()
        self.update_timer.start()

    def pause(self):
        """暂停视频"""
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PausedState:
            return

        self.media_player.pause()

    def stop(self):
        """停止视频"""
        self.media_player.stop()
        self.update_timer.stop()

    def toggle_play(self):
        """切换播放/暂停状态"""
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.pause()
        else:
            self.play()

    def toggle_mute(self):
        """切换静音状态"""
        self.audio_output.setMuted(not self.audio_output.isMuted())

        if self.audio_output.isMuted():
            self.volume_button.setIcon(self.style().standardIcon(QStyle.SP_MediaVolumeMuted))
        else:
            self.volume_button.setIcon(self.style().standardIcon(QStyle.SP_MediaVolume))

    def set_volume(self, volume):
        """
        设置音量

        Args:
            volume: 音量值 (0-100)
        """
        self.audio_output.setVolume(volume / 100.0)

    def set_position(self, position):
        """
        设置播放位置

        Args:
            position: 位置（毫秒）
        """
        self.media_player.setPosition(position)

    def seek_to(self, position_ms):
        """
        跳转到指定位置

        Args:
            position_ms: 位置（毫秒）
        """
        # 获取当前媒体状态
        current_status = self.media_player.mediaStatus()
        status_names = {
            QMediaPlayer.MediaStatus.NoMedia: "NoMedia",
            QMediaPlayer.MediaStatus.LoadingMedia: "LoadingMedia",
            QMediaPlayer.MediaStatus.LoadedMedia: "LoadedMedia",
            QMediaPlayer.MediaStatus.StalledMedia: "StalledMedia",
            QMediaPlayer.MediaStatus.BufferingMedia: "BufferingMedia",
            QMediaPlayer.MediaStatus.BufferedMedia: "BufferedMedia",
            QMediaPlayer.MediaStatus.EndOfMedia: "EndOfMedia",
            QMediaPlayer.MediaStatus.InvalidMedia: "InvalidMedia"
        }
        status_name = status_names.get(current_status, f"未知状态({current_status})")
        print(f"seek_to({position_ms}ms) 当前媒体状态: {status_name}")

        # 检查媒体状态，如果媒体已加载完成，直接跳转
        if current_status in [QMediaPlayer.MediaStatus.LoadedMedia,
                             QMediaPlayer.MediaStatus.BufferedMedia]:
            print(f"媒体已加载，直接跳转到 {position_ms}ms")
            self.media_player.setPosition(position_ms)
        else:
            # 如果媒体尚未加载完成，保存跳转位置，等待媒体加载完成后再跳转
            print(f"媒体尚未加载完成，保存待处理的跳转位置: {position_ms}ms")
            self.pending_seek_position = position_ms

    def on_playback_state_changed(self, state):
        """
        播放状态变化处理

        Args:
            state: 播放状态
        """
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        else:
            self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

        if state == QMediaPlayer.PlaybackState.StoppedState:
            self.playback_finished.emit()

    def on_position_changed(self, position):
        """
        播放位置变化处理

        Args:
            position: 当前位置（毫秒）
        """
        if not self.position_slider.isSliderDown():
            self.position_slider.setValue(position)

        # 更新时间标签
        self.update_time_label(position, self.duration)

    def on_duration_changed(self, duration):
        """
        视频时长变化处理

        Args:
            duration: 视频时长（毫秒）
        """
        self.duration = duration
        self.position_slider.setRange(0, duration)

        # 更新时间标签
        self.update_time_label(self.media_player.position(), duration)

    def on_error(self, error, error_string):
        """
        错误处理

        Args:
            error: 错误代码
            error_string: 错误描述
        """
        print(f"播放器错误: {error} - {error_string}")

    def on_media_status_changed(self, status):
        """
        媒体状态变化处理

        Args:
            status: 媒体状态
        """
        # 输出媒体状态变化，便于调试
        status_names = {
            QMediaPlayer.MediaStatus.NoMedia: "NoMedia",
            QMediaPlayer.MediaStatus.LoadingMedia: "LoadingMedia",
            QMediaPlayer.MediaStatus.LoadedMedia: "LoadedMedia",
            QMediaPlayer.MediaStatus.StalledMedia: "StalledMedia",
            QMediaPlayer.MediaStatus.BufferingMedia: "BufferingMedia",
            QMediaPlayer.MediaStatus.BufferedMedia: "BufferedMedia",
            QMediaPlayer.MediaStatus.EndOfMedia: "EndOfMedia",
            QMediaPlayer.MediaStatus.InvalidMedia: "InvalidMedia"
        }
        status_name = status_names.get(status, f"未知状态({status})")
        print(f"媒体状态变化: {status_name}")

        # 当媒体加载完成后，处理待处理的跳转请求
        if status in [QMediaPlayer.MediaStatus.LoadedMedia, QMediaPlayer.MediaStatus.BufferedMedia]:
            if self.pending_seek_position is not None:
                print(f"执行待处理的跳转: {self.pending_seek_position}ms")
                # 执行待处理的跳转
                self.media_player.setPosition(self.pending_seek_position)
                # 清除待处理的跳转位置
                self.pending_seek_position = None

    def on_slider_pressed(self):
        """滑块按下处理"""
        self.update_timer.stop()

    def on_slider_released(self):
        """滑块释放处理"""
        self.set_position(self.position_slider.value())
        self.update_timer.start()

    def update_ui(self):
        """更新UI"""
        # 如果播放器不在播放状态，停止更新
        if self.media_player.playbackState() != QMediaPlayer.PlaybackState.PlayingState:
            self.update_timer.stop()
            return

        # 检查媒体状态，只有在媒体已加载完成的状态下才更新UI
        if self.media_player.mediaStatus() in [QMediaPlayer.MediaStatus.LoadedMedia,
                                              QMediaPlayer.MediaStatus.BufferedMedia,
                                              QMediaPlayer.MediaStatus.EndOfMedia]:
            # 更新当前位置
            position = self.media_player.position()
            if not self.position_slider.isSliderDown():
                self.position_slider.setValue(position)

            # 更新时间标签
            self.update_time_label(position, self.duration)

    def update_time_label(self, position, duration):
        """
        更新时间标签

        Args:
            position: 当前位置（毫秒）
            duration: 总时长（毫秒）
        """
        position_str = self.format_time(position)
        duration_str = self.format_time(duration)
        self.time_label.setText(f"{position_str} / {duration_str}")

    def format_time(self, milliseconds):
        """
        格式化时间

        Args:
            milliseconds: 毫秒数

        Returns:
            格式化的时间字符串
        """
        seconds = int(milliseconds / 1000)
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60

        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
