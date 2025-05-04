"""
数据库模块，负责存储和检索裁剪图像的元数据
"""
import os
import sqlite3
import json
from pathlib import Path


class Database:
    """数据库管理类"""
    
    def __init__(self, db_path):
        """初始化数据库连接"""
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """初始化数据库表结构"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建crops表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS crops (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_path TEXT NOT NULL,
            timestamp_ms INTEGER NOT NULL,
            crop_image_path TEXT NOT NULL,
            bounding_box TEXT,
            confidence REAL,
            frame_width INTEGER,
            frame_height INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # 创建索引
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_video_path ON crops(video_path)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON crops(timestamp_ms)')
        
        conn.commit()
        conn.close()
    
    def add_crop(self, video_path, timestamp_ms, crop_image_path, bounding_box=None, 
                 confidence=None, frame_width=None, frame_height=None):
        """
        添加一条裁剪记录
        
        Args:
            video_path: 视频文件路径
            timestamp_ms: 时间戳（毫秒）
            crop_image_path: 裁剪图像保存路径
            bounding_box: 边界框坐标 [x, y, width, height]
            confidence: 检测置信度
            frame_width: 原始帧宽度
            frame_height: 原始帧高度
        
        Returns:
            新记录的ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 将边界框转换为JSON字符串
        if bounding_box is not None:
            bounding_box = json.dumps(bounding_box)
        
        cursor.execute('''
        INSERT INTO crops (
            video_path, timestamp_ms, crop_image_path, 
            bounding_box, confidence, frame_width, frame_height
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            video_path, timestamp_ms, crop_image_path, 
            bounding_box, confidence, frame_width, frame_height
        ))
        
        crop_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return crop_id
    
    def get_crops(self, video_path=None, limit=None, offset=None):
        """
        获取裁剪记录
        
        Args:
            video_path: 可选，按视频路径筛选
            limit: 可选，限制返回记录数
            offset: 可选，结果偏移量
        
        Returns:
            裁剪记录列表
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # 使结果可以通过列名访问
        cursor = conn.cursor()
        
        query = "SELECT * FROM crops"
        params = []
        
        if video_path:
            query += " WHERE video_path = ?"
            params.append(video_path)
        
        query += " ORDER BY video_path, timestamp_ms"
        
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
            
            if offset is not None:
                query += " OFFSET ?"
                params.append(offset)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        # 转换为字典列表
        results = []
        for row in rows:
            item = dict(row)
            # 将JSON字符串转换回Python对象
            if item['bounding_box']:
                item['bounding_box'] = json.loads(item['bounding_box'])
            results.append(item)
        
        conn.close()
        return results
    
    def get_crop_by_id(self, crop_id):
        """
        通过ID获取裁剪记录
        
        Args:
            crop_id: 裁剪记录ID
        
        Returns:
            裁剪记录或None
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM crops WHERE id = ?", (crop_id,))
        row = cursor.fetchone()
        
        if row:
            item = dict(row)
            if item['bounding_box']:
                item['bounding_box'] = json.loads(item['bounding_box'])
            conn.close()
            return item
        
        conn.close()
        return None
    
    def get_videos(self):
        """
        获取所有视频路径
        
        Returns:
            视频路径列表
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT DISTINCT video_path FROM crops ORDER BY video_path")
        rows = cursor.fetchall()
        
        conn.close()
        return [row[0] for row in rows]
    
    def count_crops(self, video_path=None):
        """
        计算裁剪记录数量
        
        Args:
            video_path: 可选，按视频路径筛选
        
        Returns:
            记录数量
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT COUNT(*) FROM crops"
        params = []
        
        if video_path:
            query += " WHERE video_path = ?"
            params.append(video_path)
        
        cursor.execute(query, params)
        count = cursor.fetchone()[0]
        
        conn.close()
        return count
    
    def delete_crop(self, crop_id):
        """
        删除裁剪记录
        
        Args:
            crop_id: 裁剪记录ID
        
        Returns:
            是否成功
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM crops WHERE id = ?", (crop_id,))
        success = cursor.rowcount > 0
        
        conn.commit()
        conn.close()
        
        return success
    
    def delete_crops_by_video(self, video_path):
        """
        删除视频的所有裁剪记录
        
        Args:
            video_path: 视频路径
        
        Returns:
            删除的记录数
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM crops WHERE video_path = ?", (video_path,))
        count = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        return count
