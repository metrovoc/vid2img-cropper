#!/usr/bin/env python3
"""
手动下载InsightFace模型

此脚本用于手动下载InsightFace模型，解决自动下载失败的问题。
"""
import os
import sys
import logging
import argparse
import tempfile
import zipfile
import urllib.request
import shutil
from pathlib import Path

from src.utils.paths import get_insightface_models_dir

# 配置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_model(model_name="buffalo_l"):
    """
    下载InsightFace模型

    Args:
        model_name: 模型名称，可选 "buffalo_l", "buffalo_m", "buffalo_s"

    Returns:
        是否成功
    """
    # 获取模型目录
    model_dir = os.path.join(get_insightface_models_dir(), model_name)
    os.makedirs(model_dir, exist_ok=True)

    # 模型文件路径 - 检查两种可能的模型文件
    model_file_mbf = os.path.join(model_dir, 'w600k_mbf.onnx')
    model_file_r50 = os.path.join(model_dir, 'w600k_r50.onnx')

    # 如果任一模型已存在，询问是否覆盖
    if os.path.exists(model_file_mbf) or os.path.exists(model_file_r50):
        existing_file = model_file_mbf if os.path.exists(model_file_mbf) else model_file_r50
        logger.info(f"模型文件已存在: {existing_file}")
        response = input("模型文件已存在，是否覆盖? (y/n): ")
        if response.lower() != 'y':
            logger.info("保留现有模型文件")
            return True

    # 根据模型名称选择URL
    if model_name == "buffalo_l":
        url = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
    elif model_name == "buffalo_m":
        url = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_m.zip"
    elif model_name == "buffalo_s":
        url = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip"
    else:
        logger.error(f"不支持的模型名称: {model_name}")
        return False

    logger.info(f"开始从 {url} 下载模型...")

    # 下载并解压模型
    with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_file:
        temp_path = temp_file.name

    try:
        # 下载模型
        urllib.request.urlretrieve(url, temp_path)
        logger.info(f"下载完成，文件保存到: {temp_path}")

        # 解压模型
        logger.info(f"正在解压模型到 {model_dir}...")
        with zipfile.ZipFile(temp_path, 'r') as zip_ref:
            # 创建临时目录用于解压
            temp_extract_dir = tempfile.mkdtemp()
            try:
                # 先解压到临时目录
                zip_ref.extractall(temp_extract_dir)

                # 查找解压后的模型文件
                for root, dirs, files in os.walk(temp_extract_dir):
                    for file in files:
                        if file.endswith('.onnx'):
                            # 找到onnx文件，复制到目标目录
                            src_file = os.path.join(root, file)
                            dst_file = os.path.join(model_dir, file)
                            logger.info(f"复制模型文件: {src_file} -> {dst_file}")
                            shutil.copy2(src_file, dst_file)
            finally:
                # 清理临时目录
                shutil.rmtree(temp_extract_dir, ignore_errors=True)

        # 检查是否存在任一模型文件
        if os.path.exists(model_file_mbf):
            logger.info(f"模型下载并解压成功: {model_file_mbf}")
            return True
        elif os.path.exists(model_file_r50):
            logger.info(f"模型下载并解压成功: {model_file_r50}")
            return True
        else:
            # 检查目录中是否有任何ONNX文件
            if os.path.exists(model_dir):
                onnx_files = [f for f in os.listdir(model_dir) if f.endswith('.onnx')]
                if onnx_files:
                    logger.info(f"找到以下ONNX模型文件: {onnx_files}")
                    return True

            logger.error(f"解压后未找到任何模型文件")
            return False
    except Exception as e:
        logger.error(f"下载或解压模型失败: {e}")
        return False
    finally:
        # 删除临时文件
        if os.path.exists(temp_path):
            os.unlink(temp_path)

def try_insightface_download(model_name="buffalo_l"):
    """
    尝试使用InsightFace内置功能下载模型

    Args:
        model_name: 模型名称

    Returns:
        是否成功
    """
    try:
        # 动态导入InsightFace，避免在未安装时报错
        try:
            import insightface
            from insightface.app import FaceAnalysis
        except ImportError:
            logger.error("未安装InsightFace库，请使用pip install insightface安装")
            return False

        logger.info(f"尝试使用InsightFace内置功能下载模型 {model_name}...")
        app = FaceAnalysis(name=model_name)
        app.prepare(ctx_id=0)

        # 检查模型目录是否存在任何ONNX文件
        model_dir = os.path.join(get_insightface_models_dir(), model_name)
        if os.path.exists(model_dir):
            onnx_files = [f for f in os.listdir(model_dir) if f.endswith('.onnx')]
            if onnx_files:
                logger.info(f"找到以下ONNX模型文件: {onnx_files}")
                return True
            else:
                logger.warning(f"InsightFace内置下载后未找到任何ONNX模型文件")
                return False
        else:
            logger.warning(f"InsightFace内置下载后未找到模型目录: {model_dir}")
            return False
    except Exception as e:
        logger.error(f"使用InsightFace内置功能下载失败: {e}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="下载InsightFace模型")
    parser.add_argument("--model", type=str, default="buffalo_l",
                        choices=["buffalo_l", "buffalo_m", "buffalo_s"],
                        help="模型名称: buffalo_l (高精度), buffalo_m (中等精度), buffalo_s (轻量级)")
    parser.add_argument("--method", type=str, default="both",
                        choices=["direct", "insightface", "both"],
                        help="下载方法: direct (直接下载), insightface (使用InsightFace内置功能), both (两种方法都尝试)")

    args = parser.parse_args()

    if args.method in ["insightface", "both"]:
        if try_insightface_download(args.model):
            logger.info("使用InsightFace内置功能下载成功")
            return 0
        elif args.method == "insightface":
            logger.error("使用InsightFace内置功能下载失败")
            return 1

    if args.method in ["direct", "both"]:
        if download_model(args.model):
            logger.info("直接下载模型成功")
            return 0
        else:
            logger.error("直接下载模型失败")
            return 1

    return 1

if __name__ in {"__main__", "__mp_main__"}:
    sys.exit(main())
