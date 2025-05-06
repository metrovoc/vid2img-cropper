"""
图片处理线程模块
"""
from PySide6.QtCore import QThread, Signal


class ImageProcessingThread(QThread):
    """图片处理线程类"""
    
    # 自定义信号
    progress_updated = Signal(int, int, float, float, float, int, int)  # 当前, 总数, 百分比, 已用时间, 剩余时间, 处理图片数, 检测到的人脸数
    status_updated = Signal(str)  # 状态信息
    processing_finished = Signal(dict)  # 处理完成，返回结果
    image_processed = Signal(int, dict)  # 单个图片处理完成，返回索引和结果
    
    def __init__(self, image_processor, image_paths, batch_mode=True):
        """
        初始化图片处理线程
        
        Args:
            image_processor: 图片处理器实例
            image_paths: 图片路径或路径列表
            batch_mode: 是否批量处理模式
        """
        super().__init__()
        self.image_processor = image_processor
        self.image_paths = image_paths if isinstance(image_paths, list) else [image_paths]
        self.batch_mode = batch_mode
        self.index = 0
    
    def run(self):
        """运行线程"""
        self.status_updated.emit(f"处理图片: {len(self.image_paths)} 个文件")
        
        if self.batch_mode:
            # 批量处理模式
            results = self.image_processor.process_images(
                self.image_paths,
                progress_callback=self.progress_updated.emit,
                status_callback=self.status_updated.emit
            )
            
            # 发送处理完成信号
            self.processing_finished.emit({"results": results})
        else:
            # 单个处理模式
            results = []
            for i, image_path in enumerate(self.image_paths):
                self.index = i
                result = self.image_processor.process_image(
                    image_path,
                    progress_callback=self.progress_updated.emit,
                    status_callback=self.status_updated.emit
                )
                results.append(result)
                
                # 发送单个图片处理完成信号
                self.image_processed.emit(i, result)
                
                if not self.image_processor.running:
                    break
            
            # 发送处理完成信号
            self.processing_finished.emit({"results": results})
        
        if not self.image_processor.running:
            self.status_updated.emit("处理已停止")
    
    def stop(self):
        """停止处理"""
        self.image_processor.stop()
