"""
图片处理选项卡模块
"""
import os
import time
from pathlib import Path
from datetime import timedelta

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QFileDialog, QGroupBox, QFormLayout,
    QLineEdit, QListWidget, QListWidgetItem, QMessageBox,
    QSplitter, QProgressDialog, QSizePolicy, QScrollArea
)
from PySide6.QtCore import Qt, QSize, QCoreApplication, QUrl
from PySide6.QtGui import QDesktopServices

from src.core.image_processor import ImageProcessor
from src.ui.image_processing_thread import ImageProcessingThread


class ImageProcessorTab(QWidget):
    """图片处理选项卡"""

    def __init__(self, config):
        """
        初始化图片处理选项卡

        Args:
            config: 配置对象
        """
        super().__init__()

        self.config = config
        self.image_processor = ImageProcessor(config)
        self.processing_thread = None

        # 初始化UI
        self.init_ui()

    def init_ui(self):
        """初始化UI组件"""
        layout = QVBoxLayout(self)

        # 创建主分割器，允许用户调整各区域大小
        main_splitter = QSplitter(Qt.Vertical)
        main_splitter.setHandleWidth(2)
        main_splitter.setChildrenCollapsible(False)

        # 文件选择区域
        file_group = QGroupBox("图片文件队列")
        file_layout = QVBoxLayout(file_group)

        file_select_layout = QHBoxLayout()
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setReadOnly(True)
        self.file_path_edit.setPlaceholderText("选择图片文件...")
        file_select_layout.addWidget(self.file_path_edit)

        self.browse_button = QPushButton("浏览文件...")
        self.browse_button.clicked.connect(self.on_browse_clicked)
        file_select_layout.addWidget(self.browse_button)

        self.browse_folder_button = QPushButton("导入文件夹...")
        self.browse_folder_button.clicked.connect(self.on_browse_folder_clicked)
        file_select_layout.addWidget(self.browse_folder_button)

        file_layout.addLayout(file_select_layout)

        # 文件信息
        self.file_info_label = QLabel("未选择文件")
        file_layout.addWidget(self.file_info_label)

        # 队列控制
        queue_control_layout = QHBoxLayout()

        self.add_queue_button = QPushButton("添加到队列")
        self.add_queue_button.clicked.connect(self.add_to_queue)
        queue_control_layout.addWidget(self.add_queue_button)

        self.clear_queue_button = QPushButton("清空队列")
        self.clear_queue_button.clicked.connect(self.clear_queue)
        queue_control_layout.addWidget(self.clear_queue_button)

        self.remove_selected_button = QPushButton("移除选中")
        self.remove_selected_button.clicked.connect(self.remove_selected)
        queue_control_layout.addWidget(self.remove_selected_button)

        file_layout.addLayout(queue_control_layout)

        # 队列列表
        self.queue_list = QListWidget()
        self.queue_list.setMinimumHeight(100)
        file_layout.addWidget(self.queue_list)

        # 添加到分割器
        main_splitter.addWidget(file_group)

        # 处理控制区域
        control_group = QGroupBox("处理控制")
        control_layout = QVBoxLayout(control_group)

        # 输出目录设置
        output_dir_layout = QHBoxLayout()

        self.output_dir_label = QLabel("输出目录:")
        output_dir_layout.addWidget(self.output_dir_label)

        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setText(self.config.get_image_output_dir())
        self.output_dir_edit.setReadOnly(True)
        output_dir_layout.addWidget(self.output_dir_edit, 1)

        self.output_dir_button = QPushButton("更改...")
        self.output_dir_button.clicked.connect(self.on_output_dir_clicked)
        output_dir_layout.addWidget(self.output_dir_button)

        control_layout.addLayout(output_dir_layout)

        # 处理按钮
        process_buttons_layout = QHBoxLayout()

        self.start_button = QPushButton("开始处理")
        self.start_button.clicked.connect(self.on_start_clicked)
        process_buttons_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("停止处理")
        self.stop_button.clicked.connect(self.on_stop_clicked)
        self.stop_button.setEnabled(False)
        process_buttons_layout.addWidget(self.stop_button)

        control_layout.addLayout(process_buttons_layout)

        # 进度条
        progress_layout = QHBoxLayout()

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)

        self.progress_label = QLabel("就绪")
        progress_layout.addWidget(self.progress_label)

        control_layout.addLayout(progress_layout)

        # 统计信息
        self.stats_label = QLabel("")
        control_layout.addWidget(self.stats_label)

        # 添加到分割器
        main_splitter.addWidget(control_group)

        # 输出区域
        output_group = QGroupBox("处理输出")
        output_layout = QVBoxLayout(output_group)

        # 输出日志
        self.output_log = QLabel("准备就绪")
        self.output_log.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.output_log.setWordWrap(True)
        self.output_log.setTextInteractionFlags(Qt.TextSelectableByMouse)

        # 创建滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.output_log)

        output_layout.addWidget(scroll_area)

        # 添加到分割器
        main_splitter.addWidget(output_group)

        # 设置分割器初始比例
        main_splitter.setSizes([200, 100, 300])

        # 添加到主布局
        layout.addWidget(main_splitter)

    def on_browse_clicked(self):
        """浏览文件按钮点击处理"""
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setNameFilter("图片文件 (*.jpg *.jpeg *.png *.bmp *.webp)")

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
                    "是否包含子文件夹中的图片文件？",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes
                ) == QMessageBox.Yes

                # 支持的图片文件扩展名
                image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']

                # 创建进度对话框
                progress_dialog = QProgressDialog("正在扫描文件夹...", "取消", 0, 100, self)
                progress_dialog.setWindowTitle("导入文件夹")
                progress_dialog.setWindowModality(Qt.WindowModal)
                progress_dialog.setMinimumDuration(500)  # 500ms后显示
                progress_dialog.setValue(0)

                # 第一步：扫描文件夹中的所有文件
                all_files = []

                if include_subfolders:
                    # 包含子文件夹
                    for root, _, files in os.walk(folder_path):
                        for file in files:
                            all_files.append(os.path.join(root, file))
                else:
                    # 不包含子文件夹
                    for file in os.listdir(folder_path):
                        file_path = os.path.join(folder_path, file)
                        if os.path.isfile(file_path):
                            all_files.append(file_path)

                total_files = len(all_files)
                if total_files == 0:
                    progress_dialog.close()
                    QMessageBox.information(self, "提示", f"文件夹 {folder_path} 中没有文件")
                    return

                # 更新进度对话框
                progress_dialog.setLabelText(f"找到 {total_files} 个文件，正在筛选图片文件...")
                progress_dialog.setValue(10)
                QCoreApplication.processEvents()

                # 第二步：筛选图片文件
                image_files = []

                for i, file_path in enumerate(all_files):
                    file_ext = os.path.splitext(file_path)[1].lower()
                    if file_ext in image_extensions:
                        image_files.append(file_path)

                    # 更新进度对话框
                    progress = 10 + int((i / total_files) * 40)  # 筛选占40%进度
                    progress_dialog.setValue(progress)
                    progress_dialog.setLabelText(f"正在筛选图片文件... ({i+1}/{total_files})")
                    QCoreApplication.processEvents()

                    if progress_dialog.wasCanceled():
                        return

                if not image_files:
                    progress_dialog.close()
                    QMessageBox.information(self, "提示", f"在文件夹 {folder_path} 中未找到支持的图片文件")
                    self.file_info_label.setText("未找到图片文件")
                    return

                # 更新文件路径编辑框和信息标签
                self.file_path_edit.setText("; ".join(image_files))
                self.file_info_label.setText(f"已选择 {len(image_files)} 个文件")

                # 关闭进度对话框
                progress_dialog.close()

    def add_to_queue(self):
        """添加文件到队列"""
        file_paths_text = self.file_path_edit.text()
        if not file_paths_text:
            QMessageBox.warning(self, "警告", "请先选择图片文件")
            return

        file_paths = file_paths_text.split("; ")

        # 检查文件是否已在队列中
        existing_paths = []
        for i in range(self.queue_list.count()):
            existing_paths.append(self.queue_list.item(i).data(Qt.UserRole))

        # 添加新文件到队列
        added_count = 0
        for path in file_paths:
            if path not in existing_paths:
                item = QListWidgetItem(os.path.basename(path))
                item.setData(Qt.UserRole, path)
                self.queue_list.addItem(item)
                added_count += 1

        if added_count > 0:
            self.file_info_label.setText(f"已添加 {added_count} 个文件到队列")
        else:
            self.file_info_label.setText("所有文件已在队列中")

    def clear_queue(self):
        """清空队列"""
        if self.queue_list.count() > 0:
            reply = QMessageBox.question(
                self,
                "清空队列",
                "确定要清空队列吗？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                self.queue_list.clear()
                self.file_info_label.setText("队列已清空")

    def remove_selected(self):
        """移除选中的队列项"""
        selected_items = self.queue_list.selectedItems()
        if not selected_items:
            QMessageBox.information(self, "提示", "请先选择要移除的项")
            return

        for item in selected_items:
            row = self.queue_list.row(item)
            self.queue_list.takeItem(row)

        self.file_info_label.setText(f"已移除 {len(selected_items)} 个项")

    def on_start_clicked(self):
        """开始按钮点击处理"""
        # 检查队列中是否有图片
        if self.queue_list.count() == 0:
            # 如果队列为空，检查是否有选择的文件
            file_paths_text = self.file_path_edit.text()
            if file_paths_text:
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
                QMessageBox.warning(self, "警告", "请先选择图片文件或添加文件到队列")
                return

        # 检查是否已经有处理线程在运行
        if self.processing_thread and self.processing_thread.isRunning():
            QMessageBox.warning(self, "警告", "已有图片正在处理中")
            return

        # 获取队列中的所有图片路径
        image_paths = []
        for i in range(self.queue_list.count()):
            image_paths.append(self.queue_list.item(i).data(Qt.UserRole))

        if not image_paths:
            QMessageBox.warning(self, "警告", "队列中没有有效的图片文件")
            return

        # 创建并启动处理线程
        self.image_processor = ImageProcessor(self.config)
        self.processing_thread = ImageProcessingThread(self.image_processor, image_paths, batch_mode=True)
        self.processing_thread.progress_updated.connect(self.update_progress)
        self.processing_thread.status_updated.connect(self.update_status)
        self.processing_thread.processing_finished.connect(self.on_processing_finished)
        self.processing_thread.image_processed.connect(self.on_image_processed)

        # 更新UI状态
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText("正在处理...")
        self.stats_label.setText("")

        # 启动线程
        self.processing_thread.start()

    def on_output_dir_clicked(self):
        """输出目录按钮点击处理"""
        dir_dialog = QFileDialog()
        dir_dialog.setFileMode(QFileDialog.Directory)
        dir_dialog.setOption(QFileDialog.ShowDirsOnly, True)

        if dir_dialog.exec():
            dir_path = dir_dialog.selectedFiles()[0]
            if dir_path:
                self.output_dir_edit.setText(dir_path)
                self.config.set("output", "image_output_dir", dir_path)
                self.update_status(f"已更改图片输出目录为: {dir_path}")

                # 更新图片处理器的输出目录
                self.image_processor = ImageProcessor(self.config)

    def on_stop_clicked(self):
        """停止按钮点击处理"""
        if self.processing_thread and self.processing_thread.isRunning():
            reply = QMessageBox.question(
                self,
                "停止处理",
                "确定要停止当前处理吗？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                self.processing_thread.stop()
                self.update_status("正在停止处理...")

    def update_progress(self, current, total, percentage, elapsed, remaining, processed_images, detected_faces):
        """
        更新进度信息

        Args:
            current: 当前处理的索引
            total: 总数
            percentage: 进度百分比
            elapsed: 已用时间（秒）
            remaining: 剩余时间（秒）
            processed_images: 已处理图片数
            detected_faces: 已检测到的人脸数
        """
        # 更新进度条
        self.progress_bar.setValue(int(percentage * 100))

        # 更新进度标签
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        remaining_str = str(timedelta(seconds=int(remaining)))
        self.progress_label.setText(f"进度: {current}/{total} ({int(percentage * 100)}%) - 已用时间: {elapsed_str} - 剩余时间: {remaining_str}")

        # 更新统计信息
        self.stats_label.setText(f"已处理 {processed_images} 张图片，检测到 {detected_faces} 个人脸")

    def update_status(self, status):
        """
        更新状态信息

        Args:
            status: 状态信息
        """
        # 添加时间戳
        timestamp = time.strftime("%H:%M:%S")
        status_text = f"[{timestamp}] {status}"

        # 更新输出日志
        current_text = self.output_log.text()
        if current_text == "准备就绪":
            self.output_log.setText(status_text)
        else:
            self.output_log.setText(f"{current_text}\n{status_text}")

    def on_processing_finished(self, result):
        """
        处理完成回调

        Args:
            result: 处理结果
        """
        # 更新UI状态
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

        # 显示处理结果
        results = result.get("results", [])
        success_count = sum(1 for r in results if r.get("success", False))
        failed_count = len(results) - success_count
        total_faces = sum(r.get("detected_faces", 0) for r in results if r.get("success", False))

        self.update_status(f"处理完成: 成功 {success_count} 个，失败 {failed_count} 个，共检测到 {total_faces} 个人脸")

        # 显示输出目录并询问是否打开
        output_dir = self.config.get_image_output_dir()
        reply = QMessageBox.question(
            self,
            "处理完成",
            f"所有图片处理完成，共检测到 {total_faces} 个人脸。\n\n输出目录: {output_dir}\n\n是否打开输出目录？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )

        if reply == QMessageBox.Yes:
            QDesktopServices.openUrl(QUrl.fromLocalFile(output_dir))

        # 询问是否清空队列
        if success_count == len(results) and failed_count == 0:
            reply = QMessageBox.question(
                self,
                "清空队列",
                "是否清空处理队列？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )

            if reply == QMessageBox.Yes:
                self.queue_list.clear()
                self.file_info_label.setText("队列已清空")

    def on_image_processed(self, index, result):
        """
        单个图片处理完成回调

        Args:
            index: 图片索引
            result: 处理结果
        """
        # 更新队列项状态
        if index < self.queue_list.count():
            item = self.queue_list.item(index)
            if result.get("success", False):
                item.setText(f"{os.path.basename(result['image_path'])} - 检测到 {result.get('detected_faces', 0)} 个人脸")
            else:
                item.setText(f"{os.path.basename(result['image_path'])} - 处理失败: {result.get('error', '未知错误')}")
