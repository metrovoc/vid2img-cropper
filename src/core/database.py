"""
数据库模块，负责存储和检索裁剪图像的元数据
"""
import os
import sqlite3
import json
from pathlib import Path
import hashlib


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

        # 创建face_groups表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS face_groups (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            feature_vector TEXT,
            sample_image_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')

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
            group_id INTEGER,
            feature_vector TEXT,
            image_hash TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (group_id) REFERENCES face_groups(id)
        )
        ''')

        # 检查是否需要添加group_id列
        cursor.execute("PRAGMA table_info(crops)")
        columns = [column[1] for column in cursor.fetchall()]
        if "group_id" not in columns:
            cursor.execute("ALTER TABLE crops ADD COLUMN group_id INTEGER")

        # 检查是否需要添加feature_vector列
        if "feature_vector" not in columns:
            cursor.execute("ALTER TABLE crops ADD COLUMN feature_vector TEXT")

        # 检查是否需要添加image_hash列
        if "image_hash" not in columns:
            cursor.execute("ALTER TABLE crops ADD COLUMN image_hash TEXT")

        # 创建索引
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_video_path ON crops(video_path)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON crops(timestamp_ms)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_group_id ON crops(group_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_image_hash ON crops(image_hash)')

        conn.commit()
        conn.close()

    def add_crop(self, video_path, timestamp_ms, crop_image_path, bounding_box=None,
                 confidence=None, frame_width=None, frame_height=None, group_id=None, feature_vector=None, image_hash=None):
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
            group_id: 人脸分组ID
            feature_vector: 人脸特征向量
            image_hash: 图像哈希值

        Returns:
            新记录的ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 将边界框转换为JSON字符串
        if bounding_box is not None:
            bounding_box = json.dumps(bounding_box)

        # 将特征向量转换为JSON字符串
        if feature_vector is not None:
            feature_vector = json.dumps(feature_vector)

        cursor.execute('''
        INSERT INTO crops (
            video_path, timestamp_ms, crop_image_path,
            bounding_box, confidence, frame_width, frame_height,
            group_id, feature_vector, image_hash
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            video_path, timestamp_ms, crop_image_path,
            bounding_box, confidence, frame_width, frame_height,
            group_id, feature_vector, image_hash
        ))

        crop_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return crop_id

    def get_crops(self, video_path=None, group_id=None, limit=None, offset=None):
        """
        获取裁剪记录

        Args:
            video_path: 可选，按视频路径筛选
            group_id: 可选，按人脸分组筛选
            limit: 可选，限制返回记录数
            offset: 可选，结果偏移量

        Returns:
            裁剪记录列表
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # 使结果可以通过列名访问
        cursor = conn.cursor()

        query = "SELECT c.*, g.name as group_name FROM crops c LEFT JOIN face_groups g ON c.group_id = g.id"
        params = []
        where_clauses = []

        if video_path:
            where_clauses.append("c.video_path = ?")
            params.append(video_path)

        if group_id is not None:
            where_clauses.append("c.group_id = ?")
            params.append(group_id)

        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)

        query += " ORDER BY c.video_path, c.timestamp_ms"

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
            if item['feature_vector']:
                item['feature_vector'] = json.loads(item['feature_vector'])
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

        cursor.execute("""
            SELECT c.*, g.name as group_name
            FROM crops c
            LEFT JOIN face_groups g ON c.group_id = g.id
            WHERE c.id = ?
        """, (crop_id,))
        row = cursor.fetchone()

        if row:
            item = dict(row)
            if item['bounding_box']:
                item['bounding_box'] = json.loads(item['bounding_box'])
            if item['feature_vector']:
                item['feature_vector'] = json.loads(item['feature_vector'])
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

    def count_crops(self, video_path=None, group_id=None):
        """
        计算裁剪记录数量

        Args:
            video_path: 可选，按视频路径筛选
            group_id: 可选，按人脸分组筛选

        Returns:
            记录数量
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = "SELECT COUNT(*) FROM crops"
        params = []
        where_clauses = []

        if video_path:
            where_clauses.append("video_path = ?")
            params.append(video_path)

        if group_id is not None:
            where_clauses.append("group_id = ?")
            params.append(group_id)

        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)

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

    # 人脸分组相关方法

    def add_face_group(self, name, feature_vector=None, sample_image_path=None):
        """
        添加人脸分组

        Args:
            name: 分组名称
            feature_vector: 特征向量
            sample_image_path: 示例图像路径

        Returns:
            新分组的ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 将特征向量转换为JSON字符串
        if feature_vector is not None:
            feature_vector = json.dumps(feature_vector)

        cursor.execute('''
        INSERT INTO face_groups (name, feature_vector, sample_image_path)
        VALUES (?, ?, ?)
        ''', (name, feature_vector, sample_image_path))

        group_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return group_id

    def get_face_groups(self):
        """
        获取所有人脸分组

        Returns:
            人脸分组列表
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM face_groups ORDER BY name")
        rows = cursor.fetchall()

        # 转换为字典列表
        results = []
        for row in rows:
            item = dict(row)
            # 将JSON字符串转换回Python对象
            if item['feature_vector']:
                item['feature_vector'] = json.loads(item['feature_vector'])
            results.append(item)

        conn.close()
        return results

    def get_empty_face_groups(self):
        """
        获取空白人脸分组（没有关联裁剪图像的分组）

        Returns:
            空白人脸分组列表
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # 查询没有关联裁剪图像的分组
        cursor.execute("""
            SELECT g.* FROM face_groups g
            LEFT JOIN (
                SELECT DISTINCT group_id FROM crops WHERE group_id IS NOT NULL
            ) c ON g.id = c.group_id
            WHERE c.group_id IS NULL
            ORDER BY g.name
        """)
        rows = cursor.fetchall()

        # 转换为字典列表
        results = []
        for row in rows:
            item = dict(row)
            # 将JSON字符串转换回Python对象
            if item['feature_vector']:
                item['feature_vector'] = json.loads(item['feature_vector'])
            results.append(item)

        conn.close()
        return results

    def get_face_group(self, group_id):
        """
        获取指定ID的人脸分组

        Args:
            group_id: 分组ID

        Returns:
            人脸分组信息
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM face_groups WHERE id = ?", (group_id,))
        row = cursor.fetchone()

        if row:
            item = dict(row)
            if item['feature_vector']:
                item['feature_vector'] = json.loads(item['feature_vector'])
            conn.close()
            return item

        conn.close()
        return None

    def update_face_group(self, group_id, name=None, feature_vector=None, sample_image_path=None):
        """
        更新人脸分组

        Args:
            group_id: 分组ID
            name: 分组名称
            feature_vector: 特征向量
            sample_image_path: 示例图像路径

        Returns:
            是否成功
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 构建更新语句
        update_fields = []
        params = []

        if name is not None:
            update_fields.append("name = ?")
            params.append(name)

        if feature_vector is not None:
            update_fields.append("feature_vector = ?")
            params.append(json.dumps(feature_vector))

        if sample_image_path is not None:
            update_fields.append("sample_image_path = ?")
            params.append(sample_image_path)

        if not update_fields:
            conn.close()
            return False

        # 添加分组ID
        params.append(group_id)

        # 执行更新
        cursor.execute(f'''
        UPDATE face_groups SET {", ".join(update_fields)}
        WHERE id = ?
        ''', params)

        success = cursor.rowcount > 0
        conn.commit()
        conn.close()

        return success

    def delete_face_group(self, group_id):
        """
        删除人脸分组

        Args:
            group_id: 分组ID

        Returns:
            是否成功
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 先将关联的裁剪记录的group_id设为NULL
        cursor.execute("UPDATE crops SET group_id = NULL WHERE group_id = ?", (group_id,))

        # 删除分组
        cursor.execute("DELETE FROM face_groups WHERE id = ?", (group_id,))
        success = cursor.rowcount > 0

        conn.commit()
        conn.close()

        return success

    def delete_empty_face_groups(self):
        """
        删除所有空白人脸分组（没有关联裁剪图像的分组）

        Returns:
            删除的分组数量
        """
        # 获取所有空白分组
        empty_groups = self.get_empty_face_groups()

        if not empty_groups:
            return 0

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 删除所有空白分组
        group_ids = [group['id'] for group in empty_groups]
        placeholders = ','.join(['?'] * len(group_ids))

        cursor.execute(f"DELETE FROM face_groups WHERE id IN ({placeholders})", group_ids)
        deleted_count = cursor.rowcount

        conn.commit()
        conn.close()

        return deleted_count

    def assign_crop_to_group(self, crop_id, group_id):
        """
        将裁剪记录分配到指定分组

        Args:
            crop_id: 裁剪记录ID
            group_id: 分组ID

        Returns:
            是否成功
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("UPDATE crops SET group_id = ? WHERE id = ?", (group_id, crop_id))
        success = cursor.rowcount > 0

        conn.commit()
        conn.close()

        return success

    def get_crops_by_group(self, group_id, limit=None, offset=None):
        """
        获取指定分组的裁剪记录

        Args:
            group_id: 分组ID
            limit: 限制返回记录数
            offset: 结果偏移量

        Returns:
            裁剪记录列表
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = "SELECT * FROM crops WHERE group_id = ?"
        params = [group_id]

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
            if item['feature_vector']:
                item['feature_vector'] = json.loads(item['feature_vector'])
            results.append(item)

        conn.close()
        return results

    def get_crop_by_image_hash(self, image_hash):
        """
        通过图像哈希获取裁剪记录

        Args:
            image_hash: 图像哈希值

        Returns:
            匹配的裁剪记录或None
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM crops WHERE image_hash = ? LIMIT 1", (image_hash,))
        row = cursor.fetchone()

        if row:
            item = dict(row)
            if item['bounding_box']:
                item['bounding_box'] = json.loads(item['bounding_box'])
            if item['feature_vector']:
                item['feature_vector'] = json.loads(item['feature_vector'])
            conn.close()
            return item

        conn.close()
        return None

    def compute_image_hash(self, image_path):
        """
        计算图像文件的哈希值

        Args:
            image_path: 图像文件路径

        Returns:
            图像哈希值
        """
        try:
            with open(image_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            return file_hash
        except Exception as e:
            print(f"计算图像哈希失败: {e}")
            return None
