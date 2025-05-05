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
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 在导入其他模块之前设置InsightFace环境变量
from src.utils.paths import set_insightface_env_vars
from src.utils.insightface_patch import patch_insightface

# 设置InsightFace环境变量
insightface_dir = set_insightface_env_vars()
logger.info(f"InsightFace模型目录: {insightface_dir}")

# 应用InsightFace补丁
patch_success = patch_insightface()
if patch_success:
    logger.info("成功应用InsightFace补丁，模型将下载到自定义目录")
else:
    logger.warning("未能应用InsightFace补丁，模型可能会下载到默认目录")

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon
from qt_material import apply_stylesheet

from src.ui.main_window import MainWindow


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Vid2Img Cropper - 视频人脸裁剪工具")
    parser.add_argument("--video", type=str, help="要处理的视频文件路径")
    parser.add_argument("--theme", type=str, default="light_blue.xml",
                        help="Material主题 (例如: light_blue.xml, dark_teal.xml)")
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

    # 应用Material Design风格
    extra = {
        # 按钮颜色
        'danger': '#F44336',
        'warning': '#FFC107',
        'success': '#4CAF50',

        # 字体
        'font_family': 'Roboto',
        'font_size': '13px',

        # 密度缩放
        'density_scale': '0',
    }

    # 应用样式表
    apply_stylesheet(app, theme=args.theme,
                    invert_secondary=(args.theme.startswith('light_')), extra=extra)

    # 创建主窗口
    window = MainWindow()
    window.show()

    # 如果指定了视频文件，自动加载
    if args.video and os.path.exists(args.video):
        window.file_path_edit.setText(args.video)
        window.file_info_label.setText(f"已选择文件: {os.path.basename(args.video)}")

    # 运行应用程序
    sys.exit(app.exec())


if __name__ in {"__main__", "__mp_main__"}:
    main()
