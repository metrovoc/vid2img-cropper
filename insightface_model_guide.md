# InsightFace 模型下载指南

本指南提供了如何解决 InsightFace 模型下载失败的问题。

## 问题描述

在使用 InsightFace 进行人脸识别时，可能会遇到以下错误：

```
特征提取模型不存在: /Users/username/.insightface/models/buffalo_l/w600k_mbf.onnx
```

这是因为 InsightFace 需要下载模型文件，但自动下载可能会失败。

## 解决方法

### 方法 1：使用提供的下载脚本

我们提供了一个专门的下载脚本，可以帮助您下载 InsightFace 模型：

```bash
python download_insightface_models.py --model buffalo_l
```

参数说明：

- `--model`: 模型名称，可选 `buffalo_l` (高精度), `buffalo_m` (中等精度), `buffalo_s` (轻量级)
- `--method`: 下载方法，可选 `direct` (直接下载), `insightface` (使用 InsightFace 内置功能), `both` (两种方法都尝试)

### 方法 2：手动下载并安装

如果上述方法不起作用，您可以手动下载并安装模型：

1. 访问 [InsightFace Releases](https://github.com/deepinsight/insightface/releases/tag/v0.7)
2. 下载对应的模型文件，例如 `buffalo_l.zip`
3. 解压文件到 `~/.insightface/models/` 目录
4. 确保 `~/.insightface/models/buffalo_l/w600k_mbf.onnx` 文件存在

### 方法 3：使用 InsightFace Python 包

您也可以尝试使用 InsightFace 的 Python 包来下载模型：

```bash
pip install --upgrade insightface
```

然后运行以下 Python 代码：

```python
import insightface
from insightface.app import FaceAnalysis
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0)
```

## 模型文件位置

InsightFace 模型文件通常保存在以下位置：

- Linux/macOS: `~/.insightface/models/`
- Windows: `C:\Users\<username>\.insightface\models\`
