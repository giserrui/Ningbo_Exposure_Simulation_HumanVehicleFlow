#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
路网碳排放数据预处理 - 增强版（修复维度问题）
"""

import numpy as np
import pandas as pd
import os
import json
import warnings
from datetime import datetime
from collections import defaultdict
from scipy import stats

# GIS相关
import fiona
from shapely.geometry import shape, LineString

# 机器学习
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from scipy.spatial import distance_matrix
import networkx as nx

warnings.filterwarnings('ignore')

np.random.seed(42)


# ==================== 配置类 ====================
class DataConfig:
    """数据预处理配置"""
    
    # ========== 输入文件路径 ==========
    SHAPEFILE_1105 = r"D:\深圳市数据\宁波市\公共路网_全天碳排放有效\公共路网_线段_2025-11-05.shp"
    SHAPEFILE_1106 = r"D:\深圳市数据\宁波市\公共路网_全天碳排放有效\公共路网_线段_2025-11-06.shp"
    SHAPEFILE_1107 = r"D:\深圳市数据\宁波市\公共路网_全天碳排放有效\公共路网_线段_2025-11-07.shp"
    SHAPEFILE_1108 = r"D:\深圳市数据\宁波市\公共路网_全天碳排放有效\公共路网_线段_2025-11-08.shp"
    SHAPEFILE_1109 = r"D:\深圳市数据\宁波市\公共路网_全天碳排放有效\公共路网_线段_2025-11-09.shp"
    
    SYNTAX_CSV = r"D:\宁波市输出\公共路网585_空间句法完整指标.csv"
    OUTPUT_DIR = r"D:\宁波市输出\STGCN_Enhanced_Data"
    
    # ========== 邻接矩阵参数 ==========
    ADJACENCY_METHOD = 'hybrid'
    ADJACENCY_K = 8
    ADJACENCY_PERCENTILE = 5
    TOPOLOGY_WEIGHT = 0.3
    
    # ========== 数据预处理参数 ==========
    USE_LOG_TRANSFORM = True
    LOG_OFFSET = 1.0
    NORMALIZATION_METHOD = 'robust'
    OUTLIER_METHOD = 'iqr'
    IQR_MULTIPLIER = 1.5
    ZSCORE_THRESHOLD = 3
    
    # ========== STGCN参数 ==========
    SEQ_LEN = 12
    PRE_LEN = 3
    
    # ========== 速度特征 ==========
    USE_SPEED_FEATURE = True
    SPEED_FIELDS = ['speed', 'avg_speed', 'mean_speed']
    
    VERBOSE = True


# ==================== 辅助函数 ====================
def convert_to_native(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_native(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_native(item) for item in obj]
    else:
        return obj


def detect_outliers_iqr(data, multiplier=1.5):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    outliers = (data < lower_bound) | (data > upper_bound)
    return outliers, lower_bound, upper_bound


def handle_outliers(data, method='iqr', iqr_multiplier=1.5, zscore_threshold=3):
    data_clean = data.copy()
    
    if method == 'iqr':
        for col in range(data.shape[1]):
            col_data = data[:, col]
            non_zero_idx = col_data > 0
            if np.sum(non_zero_idx) > 10:
                col_data_nonzero = col_data[non_zero_idx]
                outliers, lower, upper = detect_outliers_iqr(col_data_nonzero, iqr_multiplier)
                if np.any(outliers):
                    outlier_indices = np.where(non_zero_idx)[0][outliers]
                    median_val = np.median(col_data_nonzero[~outliers])
                    data_clean[outlier_indices, col] = median_val
                    if len(outlier_indices) > 0:
                        print(f"    列{col}: 替换了 {len(outlier_indices)} 个异常值")
    return data_clean


# ==================== 1. 邻接矩阵构建器 ====================
class EnhancedRoadAdjacencyBuilder:
    def __init__(self, shp_file_path, config):
        self.shp_path = shp_file_path
        self.config = config
        self.segments = []
        self.centroids = None
        self.n_segments = 0
        self.adjacency_raw = None
        self.adjacency_normalized = None
        
    def load_segments(self):
        print("\n" + "=" * 60)
        print("步骤1.1: 加载路网线段数据")
        print("=" * 60)
        
        if not os.path.exists(self.shp_path):
            raise FileNotFoundError(f"文件不存在: {self.shp_path}")
        
        with fiona.open(self.shp_path, 'r') as source:
            for idx, feature in enumerate(source):
                geom = shape(feature['geometry'])
                if geom.geom_type == 'LineString':
                    coords = list(geom.coords)
                    centroid = np.mean(coords, axis=0)
                    
                    props = feature['properties']
                    fid = props.get('FID', props.get('fid', props.get('ID', idx)))
                    
                    self.segments.append({
                        'index': idx,
                        'fid': fid,
                        'centroid': centroid,
                        'length': geom.length,
                        'geometry': geom
                    })
        
        self.n_segments = len(self.segments)
        self.centroids = np.array([s['centroid'] for s in self.segments])
        
        print(f"✅ 加载了 {self.n_segments} 条路段")
        return self.segments
    
    def get_topological_connections(self, tolerance=0.01):
        print("\n获取拓扑连接...")
        
        endpoints = []
        for seg in self.segments:
            coords = list(seg['geometry'].coords)
            endpoints.append((coords[0][0], coords[0][1], seg['index']))
            endpoints.append((coords[-1][0], coords[-1][1], seg['index']))
        
        from scipy.spatial import KDTree
        points = np.array([[p[0], p[1]] for p in endpoints])
        tree = KDTree(points)
        
        adjacency = np.zeros((self.n_segments, self.n_segments))
        matched_pairs = set()
        
        for i, (x, y, seg_i) in enumerate(endpoints):
            indices = tree.query_ball_point([x, y], tolerance)
            for j in indices:
                if i != j:
                    _, _, seg_j = endpoints[j]
                    if seg_i != seg_j:
                        pair_key = tuple(sorted([seg_i, seg_j]))
                        if pair_key not in matched_pairs:
                            adjacency[seg_i, seg_j] = 1
                            adjacency[seg_j, seg_i] = 1
                            matched_pairs.add(pair_key)
        
        n_topo_edges = np.sum(adjacency) // 2
        print(f"  - 拓扑连接数: {n_topo_edges}")
        return adjacency
    
    def build_adjacency_hybrid(self, base_k=8, topology_weight=0.3):
        print(f"\n构建混合邻接矩阵 (k={base_k}, topology_weight={topology_weight})")
        
        nn = NearestNeighbors(n_neighbors=base_k+1, metric='euclidean')
        nn.fit(self.centroids)
        distances, indices = nn.kneighbors(self.centroids)
        
        topological_adj = self.get_topological_connections()
        
        adjacency = np.zeros((self.n_segments, self.n_segments))
        
        for i in range(self.n_segments):
            for j_idx, dist in zip(indices[i][1:], distances[i][1:]):
                spatial_weight = 1.0 / (dist + 0.1)
                topo_weight = topological_adj[i, j_idx]
                combined_weight = (1 - topology_weight) * spatial_weight + topology_weight * topo_weight
                if combined_weight > 0:
                    adjacency[i, j_idx] = combined_weight
                    adjacency[j_idx, i] = combined_weight
        
        return adjacency
    
    def build_adjacency_mst_augmented(self, base_adjacency):
        print("\n使用MST增强图连通性...")
        
        G = nx.Graph()
        G.add_nodes_from(range(self.n_segments))
        
        for i in range(self.n_segments):
            for j in range(i+1, self.n_segments):
                if base_adjacency[i, j] > 0:
                    G.add_edge(i, j, weight=base_adjacency[i, j])
        
        components = list(nx.connected_components(G))
        
        if len(components) > 1:
            print(f"  - 发现 {len(components)} 个连通分量，添加MST边...")
            for i in range(len(components) - 1):
                min_dist = float('inf')
                best_pair = None
                for node_i in components[i]:
                    for node_j in components[i+1]:
                        dist = np.linalg.norm(self.centroids[node_i] - self.centroids[node_j])
                        if dist < min_dist:
                            min_dist = dist
                            best_pair = (node_i, node_j)
                if best_pair:
                    weight = 1.0 / (min_dist + 0.1)
                    base_adjacency[best_pair[0], best_pair[1]] = weight
                    base_adjacency[best_pair[1], best_pair[0]] = weight
        
        return base_adjacency
    
    def build(self):
        self.load_segments()
        
        if self.config.ADJACENCY_METHOD == 'hybrid':
            self.adjacency_raw = self.build_adjacency_hybrid(
                base_k=self.config.ADJACENCY_K,
                topology_weight=self.config.TOPOLOGY_WEIGHT
            )
        else:
            raise ValueError(f"不支持的方法: {self.config.ADJACENCY_METHOD}")
        
        self.adjacency_raw = self.build_adjacency_mst_augmented(self.adjacency_raw)
        
        n_edges = np.sum(self.adjacency_raw > 0) // 2
        degrees = np.sum(self.adjacency_raw > 0, axis=1)
        
        print(f"\n✅ 邻接矩阵构建完成:")
        print(f"  - 形状: {self.adjacency_raw.shape}")
        print(f"  - 边数: {n_edges}")
        print(f"  - 平均度数: {np.mean(degrees):.2f}")
        
        return self.adjacency_raw
    
    def normalize_adjacency(self, add_self_loop=True):
        print("\n归一化邻接矩阵...")
        
        adjacency = self.adjacency_raw.copy()
        if add_self_loop:
            adjacency = adjacency + np.eye(adjacency.shape[0])
        
        degrees = np.sum(adjacency, axis=1)
        d_inv_sqrt = np.power(degrees, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
        d_mat_inv_sqrt = np.diag(d_inv_sqrt)
        
        normalized = d_mat_inv_sqrt @ adjacency @ d_mat_inv_sqrt
        
        print(f"✅ 归一化完成: {normalized.shape}")
        self.adjacency_normalized = normalized
        return normalized


# ==================== 2. 碳排放矩阵构建器 ====================
class EnhancedCarbonMatrixBuilder:
    def __init__(self, shp_files_dict, n_segments, config):
        self.shp_files = shp_files_dict
        self.n_segments = n_segments
        self.config = config
        self.n_days = len(shp_files_dict)
        self.n_hours = self.n_days * 24
        self.carbon_raw = None
        self.carbon_cleaned = None
        self.carbon_normalized = None
        self.speed_matrix = None
        
    def build_carbon_matrix(self):
        print("\n" + "=" * 60)
        print("步骤2: 构建碳排放和速度矩阵")
        print("=" * 60)
        
        carbon_matrix = np.zeros((self.n_hours, self.n_segments))
        speed_matrix = np.zeros((self.n_hours, self.n_segments))
        
        for day_idx, (date_str, shp_path) in enumerate(sorted(self.shp_files.items())):
            print(f"\n处理第 {day_idx+1}/{self.n_days} 天: {date_str}")
            
            if not os.path.exists(shp_path):
                print(f"  ⚠️ 文件不存在: {shp_path}")
                continue
            
            daily_carbon = np.zeros((24, self.n_segments))
            daily_speed = np.zeros((24, self.n_segments))
            
            with fiona.open(shp_path, 'r') as source:
                for feature in source:
                    props = feature['properties']
                    
                    fid = props.get('FID')
                    if fid is None:
                        fid = props.get('fid')
                    if fid is None:
                        fid = props.get('ID')
                    
                    if fid is not None and 0 <= fid < self.n_segments:
                        for hour in range(24):
                            co2_col = f'co2_{hour:02d}'
                            if co2_col in props:
                                val = props[co2_col]
                                if val is not None and not np.isnan(val):
                                    daily_carbon[hour, fid] = float(val)
                            
                            if self.config.USE_SPEED_FEATURE:
                                for sp_col in self.config.SPEED_FIELDS:
                                    if sp_col in props:
                                        speed_val = props[sp_col]
                                        if speed_val is not None and not np.isnan(speed_val):
                                            daily_speed[hour, fid] = float(speed_val)
                                        break
            
            hour_start = day_idx * 24
            hour_end = (day_idx + 1) * 24
            carbon_matrix[hour_start:hour_end, :] = daily_carbon
            speed_matrix[hour_start:hour_end, :] = daily_speed
            
            nonzero_ratio = np.sum(daily_carbon > 0) / daily_carbon.size * 100
            print(f"  ✅ 碳排放: 非零率={nonzero_ratio:.1f}%")
        
        self.carbon_raw = carbon_matrix
        self.speed_matrix = speed_matrix
        
        print(f"\n✅ 矩阵构建完成")
        print(f"  - 碳排放维度: {carbon_matrix.shape}")
        print(f"  - 碳排放范围: [{np.min(carbon_matrix):.2f}, {np.max(carbon_matrix):.2f}]")
        
        return carbon_matrix, speed_matrix
    
    def process_carbon_data(self):
        print("\n处理碳排放异常值...")
        self.carbon_cleaned = handle_outliers(self.carbon_raw, self.config.OUTLIER_METHOD)
        
        if self.config.USE_LOG_TRANSFORM:
            print("\n应用对数变换...")
            carbon_log = np.log1p(self.carbon_cleaned)
            print(f"  - 变换后范围: [{np.min(carbon_log):.4f}, {np.max(carbon_log):.4f}]")
            data_to_norm = carbon_log
        else:
            data_to_norm = self.carbon_cleaned
        
        print(f"\n归一化碳排放 (方法: {self.config.NORMALIZATION_METHOD})")
        
        if self.config.NORMALIZATION_METHOD == 'robust':
            scaler = RobustScaler()
            data_reshaped = data_to_norm.reshape(-1, data_to_norm.shape[1])
            self.carbon_normalized = scaler.fit_transform(data_reshaped)
            self.carbon_normalized = self.carbon_normalized.reshape(data_to_norm.shape)
        
        print(f"  - 归一化后范围: [{np.min(self.carbon_normalized):.4f}, {np.max(self.carbon_normalized):.4f}]")
        
        return self.carbon_normalized


# ==================== 3. 空间句法矩阵构建器 ====================
class SyntaxMatrixBuilder:
    def __init__(self, syntax_csv_path, n_segments, config):
        self.syntax_csv_path = syntax_csv_path
        self.n_segments = n_segments
        self.config = config
        self.syntax_normalized = None
        self.metric_names = None
        
    def build_syntax_matrix(self):
        print("\n" + "=" * 60)
        print("步骤3: 构建空间句法矩阵")
        print("=" * 60)
        
        syntax_metrics = [
            'Connectivity', 'GlobalIntegration', 'LocalIntegrationR',
            'ControlValue', 'Visibility', 'IsovistArea', 'LocalDepthR',
            'AvgDepth', 'DepthStd', 'NormalizedIntegration',
            'TransparencyAdjustedVisibility', 'VisibleSurfacePoints'
        ]
        
        df = pd.read_csv(self.syntax_csv_path)
        print(f"✅ 加载CSV: {df.shape}")
        
        available_metrics = [m for m in syntax_metrics if m in df.columns]
        print(f"可用指标: {len(available_metrics)}个")
        
        if 'FID' in df.columns:
            df = df.sort_values('FID').reset_index(drop=True)
        
        syntax_raw = df[available_metrics].values.astype(float)
        self.metric_names = available_metrics
        
        if syntax_raw.shape[0] != self.n_segments:
            print(f"  ⚠️ 维度不匹配: {syntax_raw.shape[0]} vs {self.n_segments}")
            aligned_matrix = np.zeros((self.n_segments, len(available_metrics)))
            min_rows = min(syntax_raw.shape[0], self.n_segments)
            aligned_matrix[:min_rows, :] = syntax_raw[:min_rows, :]
            syntax_raw = aligned_matrix
        
        print(f"\n归一化空间句法数据 (方法: {self.config.NORMALIZATION_METHOD})")
        
        if self.config.NORMALIZATION_METHOD == 'robust':
            scaler = RobustScaler()
            self.syntax_normalized = scaler.fit_transform(syntax_raw)
        
        print(f"  - 归一化后范围: [{np.min(self.syntax_normalized):.4f}, {np.max(self.syntax_normalized):.4f}]")
        
        return self.syntax_normalized


# ==================== 4. STGCN样本生成器（修复版）====================
class STGCNSampleGenerator:
    def __init__(self, carbon_data, syntax_data, speed_data, config):
        self.carbon_data = carbon_data  # (T, N)
        self.syntax_data = syntax_data  # (N, D)
        self.speed_data = speed_data    # (T, N)
        self.config = config
        
    def generate_samples(self):
        print("\n" + "=" * 60)
        print("步骤4: 生成STGCN训练样本")
        print("=" * 60)
        
        T, N = self.carbon_data.shape
        D = self.syntax_data.shape[1]
        
        input_features = 1  # 碳排放
        if self.config.USE_SPEED_FEATURE and self.speed_data is not None:
            input_features += 1
            print(f"  - 包含速度特征")
        
        total_features = input_features + D
        print(f"  - 总特征维度: {total_features}")
        
        n_samples = T - self.config.SEQ_LEN - self.config.PRE_LEN + 1
        print(f"\n可生成样本数: {n_samples}")
        
        # 初始化数组 - 修复：正确设置维度
        X = np.zeros((n_samples, self.config.SEQ_LEN, N, input_features))
        y = np.zeros((n_samples, self.config.PRE_LEN, N, 1))
        
        # 生成滑动窗口样本 - 修复：不使用.T转置
        for i in range(n_samples):
            # 碳排放: carbon_data[i:i+seq_len, :] 形状 (seq_len, N)
            X[i, :, :, 0] = self.carbon_data[i:i+self.config.SEQ_LEN, :]
            
            # 速度（如果有）
            if self.config.USE_SPEED_FEATURE and self.speed_data is not None:
                X[i, :, :, 1] = self.speed_data[i:i+self.config.SEQ_LEN, :]
            
            # 目标: 未来pre_len个时间步
            y[i, :, :, 0] = self.carbon_data[i+self.config.SEQ_LEN:i+self.config.SEQ_LEN+self.config.PRE_LEN, :]
        
        # 扩展静态特征到每个时间步
        # syntax_data: (N, D) -> 扩展到 (n_samples, seq_len, N, D)
        syntax_expanded = np.zeros((n_samples, self.config.SEQ_LEN, N, D))
        for s in range(n_samples):
            for t in range(self.config.SEQ_LEN):
                syntax_expanded[s, t, :, :] = self.syntax_data
        
        # 拼接特征
        X_combined = np.concatenate([X, syntax_expanded], axis=-1)
        
        print(f"\n✅ 样本生成完成:")
        print(f"  - X形状: {X_combined.shape}")
        print(f"  - y形状: {y.shape}")
        
        # 划分数据集（时间顺序）
        train_end = int(n_samples * 0.7)
        val_end = train_end + int(n_samples * 0.15)
        
        X_train = X_combined[:train_end]
        y_train = y[:train_end]
        X_val = X_combined[train_end:val_end]
        y_val = y[train_end:val_end]
        X_test = X_combined[val_end:]
        y_test = y[val_end:]
        
        print(f"\n数据集划分:")
        print(f"  - 训练集: {X_train.shape[0]} 样本")
        print(f"  - 验证集: {X_val.shape[0]} 样本")
        print(f"  - 测试集: {X_test.shape[0]} 样本")
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test,
            'n_samples': n_samples,
            'input_features': total_features
        }


# ==================== 5. 主流水线 ====================
class DataPreparationPipeline:
    def __init__(self, config):
        self.config = config
        self.results = {}
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        
    def run(self):
        print("=" * 60)
        print("🚀 增强版路网碳排放数据预处理系统")
        print("=" * 60)
        
        # 步骤1: 构建邻接矩阵
        adj_builder = EnhancedRoadAdjacencyBuilder(self.config.SHAPEFILE_1105, self.config)
        adjacency_raw = adj_builder.build()
        adjacency_normalized = adj_builder.normalize_adjacency(add_self_loop=True)
        
        self.results['adjacency_raw'] = adjacency_raw
        self.results['adjacency_normalized'] = adjacency_normalized
        self.results['n_nodes'] = adj_builder.n_segments
        
        # 步骤2: 构建碳排放和速度矩阵
        shp_files = {
            '1105': self.config.SHAPEFILE_1105,
            '1106': self.config.SHAPEFILE_1106,
            '1107': self.config.SHAPEFILE_1107,
            '1108': self.config.SHAPEFILE_1108,
            '1109': self.config.SHAPEFILE_1109,
        }
        
        carbon_builder = EnhancedCarbonMatrixBuilder(shp_files, adj_builder.n_segments, self.config)
        carbon_raw, speed_raw = carbon_builder.build_carbon_matrix()
        
        self.results['carbon_raw'] = carbon_raw
        self.results['carbon_normalized'] = carbon_builder.process_carbon_data()
        
        if self.config.USE_SPEED_FEATURE:
            self.results['speed_raw'] = speed_raw
            # 速度归一化
            speed_cleaned = handle_outliers(speed_raw, self.config.OUTLIER_METHOD)
            if self.config.USE_LOG_TRANSFORM:
                speed_log = np.log1p(speed_cleaned)
                speed_norm_data = speed_log
            else:
                speed_norm_data = speed_cleaned
            
            scaler = RobustScaler()
            speed_reshaped = speed_norm_data.reshape(-1, speed_norm_data.shape[1])
            self.results['speed_normalized'] = scaler.fit_transform(speed_reshaped)
            self.results['speed_normalized'] = self.results['speed_normalized'].reshape(speed_norm_data.shape)
        
        self.results['n_hours'] = carbon_builder.n_hours
        
        # 步骤3: 构建空间句法矩阵
        syntax_builder = SyntaxMatrixBuilder(self.config.SYNTAX_CSV, adj_builder.n_segments, self.config)
        syntax_normalized = syntax_builder.build_syntax_matrix()
        
        self.results['syntax_normalized'] = syntax_normalized
        self.results['syntax_metrics'] = syntax_builder.metric_names
        
        # 步骤4: 生成STGCN样本
        speed_data = self.results.get('speed_normalized') if self.config.USE_SPEED_FEATURE else None
        sample_gen = STGCNSampleGenerator(
            self.results['carbon_normalized'],
            syntax_normalized,
            speed_data,
            self.config
        )
        stgcn_data = sample_gen.generate_samples()
        self.results.update(stgcn_data)
        
        # 步骤5: 保存数据
        self._save_data()
        
        print("\n" + "=" * 60)
        print("🎉 数据预处理完成！")
        print("=" * 60)
        print(f"📁 输出目录: {self.config.OUTPUT_DIR}")
        
        return self.results
    
    def _save_data(self):
        print("\n" + "=" * 60)
        print("步骤5: 保存数据")
        print("=" * 60)
        
        output_dir = self.config.OUTPUT_DIR
        
        np.save(os.path.join(output_dir, "adjacency_normalized.npy"), self.results['adjacency_normalized'])
        np.save(os.path.join(output_dir, "carbon_normalized.npy"), self.results['carbon_normalized'])
        np.save(os.path.join(output_dir, "syntax_normalized.npy"), self.results['syntax_normalized'])
        
        if self.config.USE_SPEED_FEATURE:
            np.save(os.path.join(output_dir, "speed_normalized.npy"), self.results['speed_normalized'])
        
        np.savez(os.path.join(output_dir, "stgcn_data.npz"),
                 X_train=self.results['X_train'], y_train=self.results['y_train'],
                 X_val=self.results['X_val'], y_val=self.results['y_val'],
                 X_test=self.results['X_test'], y_test=self.results['y_test'],
                 seq_len=self.config.SEQ_LEN, pre_len=self.config.PRE_LEN)
        
        print(f"✅ 数据保存完成")


def main():
    config = DataConfig()
    pipeline = DataPreparationPipeline(config)
    results = pipeline.run()
    
    print("\n📖 使用说明:")
    print("   data = np.load('stgcn_data.npz')")
    print("   X_train, y_train = data['X_train'], data['y_train']")
    
    return results


if __name__ == "__main__":
    results = main()