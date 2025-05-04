#!/usr/bin/env python3
"""
Vid2Img Cropper - 视频人脸裁剪工具

这个程序可以自动检测视频中的人物脸部，将包含脸部的画面区域裁剪出来，
并保存这些裁剪后的图片。同时，记录每次裁剪对应的原始视频时间点，
以便用户可以从裁剪结果快速跳转回视频的相应位置。
"""
import os
import sys
import argparse
from pathlib import Path

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon

from src.ui.main_window import MainWindow


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Vid2Img Cropper - 视频人脸裁剪工具")
    parser.add_argument("--video", type=str, help="要处理的视频文件路径")
    return parser.parse_args()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 创建应用程序
    app = QApplication(sys.argv)
    app.setApplicationName("Vid2Img Cropper")
    
    # 设置应用程序图标
    icon_path = os.path.join(os.path.dirname(__file__), "resources", "icon.png")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    
    # 创建主窗口
    window = MainWindow()
    window.show()
    
    # 如果指定了视频文件，自动加载
    if args.video and os.path.exists(args.video):
        window.file_path_edit.setText(args.video)
        window.file_info_label.setText(f"已选择文件: {os.path.basename(args.video)}")
    
    # 运行应用程序
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
