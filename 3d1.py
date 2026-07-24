#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
3D Space-Syntax-like 指标计算（基于 SHP 建筑 + 路网）
深度类指标专项计算：
1. 局部深度（Local Depth）: 半径R内视点到建筑被看点的深度均值
2. 全局深度（Global Depth）: 所有视点到建筑被看点的深度均值（含遮挡点）
3. 深度标准差（Depth Std）: 深度值的离散程度，反映可达性均匀性
4. 局部平均深度（Local Mean Depth R）: 半径R内深度值的总和均值
"""
import os
import math
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
from tqdm import tqdm
from scipy.spatial import cKDTree

# ----------------  参数区  ---------------- #
BUILDING_SHP = r"D:\深圳市数据\宁波市\宁波建筑.shp"
ROAD_SHP     = r"D:\深圳市数据\宁波市\宁波市道路路网_筛选后\330200_filtered.shp"
OUT_DIR      = r"D:\宁波市输出\3D指标\depth_metrics"
os.makedirs(OUT_DIR, exist_ok=True)

# 分批参数
BATCH_SIZE = 50        # 每批处理视点数（深度计算复杂，建议减小批次）
RESUME = True          # 是否断点续跑
TEMP_DIR = os.path.join(OUT_DIR, "temp_batches")
os.makedirs(TEMP_DIR, exist_ok=True)

# 计算参数
EYE_HEIGHT = 1.6       # 视点高(m)
MAX_DEPTH = 20         # 深度上限
VIS_R = 150            # 可视搜索半径(m)
LOCAL_RADIUS = 500     # 局部深度计算半径(m) - 文档推荐500米
HEIGHT_FIELD = 'height'  # 建筑高度字段
DEFAULT_HEIGHT = 15      # 缺字段默认高

# 深度计算参数
MIN_DEPTH = 1.0        # 深度下限（直接可见）
DEPTH_INCREMENT = 1.0  # 每层遮挡深度增加值
MAX_OCCLUSION_LAYERS = 3  # 最大遮挡层级（简化计算）
# ----------------------------------------- #

# ----------------  工具函数  ---------------- #
def save_batch(batch_data, batch_idx):
    """保存单批次结果（CSV）"""
    csv_path = os.path.join(TEMP_DIR, f"batch_{batch_idx:04d}.csv")
    pd.DataFrame(batch_data).to_csv(csv_path, index=False, float_format='%.4f')
    
    # 记录已完成批次
    completed_path = os.path.join(TEMP_DIR, "completed_batches.json")
    completed = []
    if os.path.exists(completed_path):
        with open(completed_path, 'r', encoding='utf-8') as f:
            completed = json.load(f)
    if batch_idx not in completed:
        completed.append(batch_idx)
        with open(completed_path, 'w', encoding='utf-8') as f:
            json.dump(sorted(completed), f, ensure_ascii=False)
    
    print(f"✅ 批次 {batch_idx} 已保存（{len(batch_data)} 条数据）")

def load_completed_batches():
    """加载已完成的批次索引"""
    completed_path = os.path.join(TEMP_DIR, "completed_batches.json")
    if os.path.exists(completed_path):
        with open(completed_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def merge_batches(final_out_dir):
    """合并所有批次为最终文件"""
    completed = load_completed_batches()
    if not completed:
        print("⚠️ 无已完成批次，跳过合并")
        return
    
    # 合并CSV
    all_data = []
    for batch_idx in sorted(completed):
        csv_path = os.path.join(TEMP_DIR, f"batch_{batch_idx:04d}.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            all_data.append(df)
    
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        final_df = final_df.drop_duplicates(subset=['vp_id'])
        
        # 计算指标统计
        stats = calculate_global_stats(final_df)
        
        # 保存最终CSV
        final_csv = os.path.join(final_out_dir, "depth_metrics_final.csv")
        final_df.to_csv(final_csv, index=False, float_format='%.4f')
        
        # 保存统计报告
        stats_path = os.path.join(final_out_dir, "depth_statistics_report.txt")
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write("深度类指标统计报告\n")
            f.write("=" * 60 + "\n")
            f.write(f"总视点数: {len(final_df)}\n")
            f.write(f"局部半径: {LOCAL_RADIUS}米\n")
            f.write(f"最大深度限制: {MAX_DEPTH}\n\n")
            
            for metric, values in stats.items():
                f.write(f"{metric}:\n")
                for stat_name, stat_value in values.items():
                    f.write(f"  {stat_name}: {stat_value}\n")
                f.write("\n")
        
        print(f"\n🎉 所有批次合并完成！")
        print(f"📊 总计 {len(final_df)} 个视点")
        print(f"📁 最终CSV：{final_csv}")
        print(f"📊 统计报告：{stats_path}")
        
        # 打印关键统计
        print("\n📈 深度指标关键统计:")
        print(f"  局部深度均值: {final_df['local_depth'].mean():.3f} (范围: {final_df['local_depth'].min():.3f}-{final_df['local_depth'].max():.3f})")
        print(f"  全局深度均值: {final_df['global_depth'].mean():.3f} (范围: {final_df['global_depth'].min():.3f}-{final_df['global_depth'].max():.3f})")
        print(f"  局部平均深度R均值: {final_df['local_mean_depth_R'].mean():.3f}")
        print(f"  深度标准差均值: {final_df['depth_std'].mean():.3f}")
        
    else:
        print("⚠️ 无有效批次数据可合并")

def calculate_global_stats(df):
    """计算全局统计信息"""
    stats = {}
    
    depth_metrics = ['local_depth', 'global_depth', 'local_mean_depth_R', 'depth_std']
    
    for metric in depth_metrics:
        if metric in df.columns:
            data = df[metric]
            stats[metric] = {
                '均值': round(float(data.mean()), 4),
                '中位数': round(float(data.median()), 4),
                '标准差': round(float(data.std()), 4),
                '最小值': round(float(data.min()), 4),
                '最大值': round(float(data.max()), 4),
                'Q1': round(float(data.quantile(0.25)), 4),
                'Q3': round(float(data.quantile(0.75)), 4)
            }
    
    return stats

# ----------------  几何函数  ---------------- #
def vector_normalize(v):
    """向量归一化"""
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-8 else v

def ray_triangle_intersect(ro, rd, v0, v1, v2):
    """Möller–Trumbore 射线-三角求交，返回是否相交及交点距离"""
    eps = 1e-6
    edge1 = v1 - v0
    edge2 = v2 - v0
    h = np.cross(rd, edge2)
    a = edge1.dot(h)
    if abs(a) < eps:
        return False, float('inf')
    f = 1.0 / a
    s = ro - v0
    u = f * s.dot(h)
    if u < 0.0 or u > 1.0:
        return False, float('inf')
    q = np.cross(s, edge1)
    v = f * rd.dot(q)
    if v < 0.0 or u + v > 1.0:
        return False, float('inf')
    t = f * edge2.dot(q)
    return t > eps, t

def building_to_triangles(row):
    """
    单栋建筑 -> 3D 三角列表
    返回: [(v0,v1,v2,b_idx), ...]
    """
    tri_list = []
    poly = row.geometry
    if poly is None or poly.is_empty:
        return tri_list
    if poly.geom_type != 'Polygon':
        return tri_list
    
    # 获取建筑高度
    try:
        height = float(row.get(HEIGHT_FIELD, DEFAULT_HEIGHT))
        height = max(0.1, height)  # 确保高度为正
    except:
        height = DEFAULT_HEIGHT
    
    coords = list(poly.exterior.coords)[:-1]
    n = len(coords)
    if n < 3:
        return tri_list
    
    # 底面三角形
    for i in range(1, n - 1):
        a = np.array([coords[0][0], coords[0][1], 0.])
        b = np.array([coords[i][0], coords[i][1], 0.])
        c = np.array([coords[i + 1][0], coords[i + 1][1], 0.])
        tri_list.append((a, b, c, row.name))
    
    # 顶面三角形
    for i in range(1, n - 1):
        a = np.array([coords[0][0], coords[0][1], height])
        b = np.array([coords[i + 1][0], coords[i + 1][1], height])
        c = np.array([coords[i][0], coords[i][1], height])
        tri_list.append((a, b, c, row.name))
    
    # 侧面三角形
    for i in range(n):
        p1 = np.array([coords[i][0], coords[i][1], 0.])
        p2 = np.array([coords[i][0], coords[i][1], height])
        p3 = np.array([coords[(i + 1) % n][0], coords[(i + 1) % n][1], 0.])
        p4 = np.array([coords[(i + 1) % n][0], coords[(i + 1) % n][1], height])
        tri_list.append((p1, p2, p3, row.name))
        tri_list.append((p2, p4, p3, row.name))
    
    return tri_list

def buildings_to_triangles(gdf):
    """全部建筑 -> 扁平三角列表"""
    triangles = []
    for idx, row in tqdm(gdf.iterrows(), desc="🔨 建筑三角化", total=len(gdf)):
        row.name = idx  # 用于标记建筑 id
        triangles.extend(building_to_triangles(row))
    return triangles

# ----------------  视点生成  ---------------- #
def generate_viewpoints(road_gdf, height=EYE_HEIGHT, sample_interval=100):
    """
    从路网生成3D视点（沿路径均匀采样）
    返回视点列表
    """
    vp_list = []
    vp_id = 0
    
    for idx, row in tqdm(road_gdf.iterrows(), desc="📍 生成视点", total=len(road_gdf)):
        line = row.geometry
        if line is None or line.is_empty:
            continue
        
        # 计算采样点数
        line_length = line.length
        if line_length < sample_interval:
            # 短路径取中点
            sample_points = [line.interpolate(0.5, normalized=True)]
        else:
            num_samples = max(2, int(line_length / sample_interval))
            sample_points = [line.interpolate(i/(num_samples-1), normalized=True) 
                           for i in range(num_samples)]
        
        for point in sample_points:
            vp_list.append({
                'vp_id': vp_id,
                'x': round(point.x, 6),
                'y': round(point.y, 6),
                'z': round(height, 2),
                'ref': str(row.get('name', row.get('ref', f'road_{idx}'))),
                'road_length': round(line_length, 2),
                'idx_on_road': len(sample_points)
            })
            vp_id += 1
    
    print(f"\n📌 共生成 {len(vp_list)} 个有效视点")
    return vp_list

# ----------------  深度指标核心计算函数  ---------------- #
def calculate_viewpoint_depth(current_vp, target_vps, triangles, tri_kdt):
    """
    计算当前视点到目标视点的深度列表
    深度定义：视线穿过的遮挡建筑层数 + 1
    返回：深度列表
    """
    current_point = np.array([current_vp['x'], current_vp['y'], current_vp['z']])
    depth_list = []
    
    for target_vp in target_vps:
        target_point = np.array([target_vp['x'], target_vp['y'], target_vp['z']])
        
        # 计算视线方向
        direction = target_point - current_point
        distance_2d = np.linalg.norm(direction[:2])
        
        # 如果距离太近，视为直接可见
        if distance_2d < 1.0:
            depth_list.append(MIN_DEPTH)
            continue
        
        # 归一化视线方向
        ray_direction = vector_normalize(direction)
        
        # 查询可视半径内的建筑三角形
        candidate_indices = tri_kdt.query_ball_point(current_point[:2], VIS_R)
        candidate_triangles = [triangles[i] for i in candidate_indices]
        
        # 计算遮挡层数
        occlusion_layers = 0
        has_direct_intersection = False
        
        for tri in candidate_triangles:
            v0, v1, v2, building_id = tri
            
            # 检查视线是否与三角形相交
            intersects, t = ray_triangle_intersect(current_point, ray_direction, v0, v1, v2)
            
            if intersects and 0 < t < distance_2d:
                # 如果交点在视线路径上，增加遮挡层数
                occlusion_layers += 1
                
                # 检查是否为直接视线（目标建筑自身）
                if np.array_equal(v0[:2], target_point[:2]) or \
                   np.array_equal(v1[:2], target_point[:2]) or \
                   np.array_equal(v2[:2], target_point[:2]):
                    has_direct_intersection = True
        
        # 计算深度值
        if has_direct_intersection:
            # 直接看到目标建筑，深度为1
            depth = MIN_DEPTH
        else:
            # 深度 = 1 + 遮挡层数 * 增量
            depth = MIN_DEPTH + occlusion_layers * DEPTH_INCREMENT
        
        # 限制深度范围
        depth = min(max(MIN_DEPTH, depth), MAX_DEPTH)
        depth_list.append(depth)
    
    return depth_list

def calculate_depth_metrics_for_viewpoint(current_vp, all_vps, vp_kdt, triangles, tri_kdt):
    """
    计算单个视点的所有深度指标
    
    指标定义：
    1. 局部深度（Local Depth）：半径R内视点到建筑被看点的深度均值
    2. 全局深度（Global Depth）：所有视点到建筑被看点的深度均值（含遮挡点）
    3. 深度标准差（Depth Std）：深度值的离散程度
    4. 局部平均深度（Local Mean Depth R）：半径R内深度值的总和均值
    """
    current_coords = (current_vp['x'], current_vp['y'])
    current_id = current_vp['vp_id']
    
    # 1. 获取局部视点（半径R内，排除自身）
    local_indices = vp_kdt.query_ball_point(current_coords, LOCAL_RADIUS)
    local_indices = [i for i in local_indices if i != current_id and i < len(all_vps)]
    local_vps = [all_vps[i] for i in local_indices]
    
    # 2. 获取全局视点（所有视点，排除自身）
    global_indices = [i for i in range(len(all_vps)) if i != current_id]
    global_vps = [all_vps[i] for i in global_indices]
    
    # 3. 计算局部深度
    local_depths = []
    if local_vps:
        local_depths = calculate_viewpoint_depth(current_vp, local_vps, triangles, tri_kdt)
    
    # 4. 计算全局深度
    global_depths = []
    if global_vps:
        # 为减少计算量，对全局视点进行采样
        if len(global_vps) > 1000:
            sample_indices = np.random.choice(len(global_vps), 1000, replace=False)
            sampled_global_vps = [global_vps[i] for i in sample_indices]
            global_depths = calculate_viewpoint_depth(current_vp, sampled_global_vps, triangles, tri_kdt)
        else:
            global_depths = calculate_viewpoint_depth(current_vp, global_vps, triangles, tri_kdt)
    
    # 5. 计算各个指标
    metrics = {}
    
    # 局部深度（Local Depth）
    if local_depths:
        metrics['local_depth'] = round(np.mean(local_depths), 3)
        metrics['local_depth_min'] = round(np.min(local_depths), 3)
        metrics['local_depth_max'] = round(np.max(local_depths), 3)
        metrics['local_view_count'] = len(local_depths)
    else:
        metrics['local_depth'] = MAX_DEPTH
        metrics['local_depth_min'] = MAX_DEPTH
        metrics['local_depth_max'] = MAX_DEPTH
        metrics['local_view_count'] = 0
    
    # 全局深度（Global Depth）
    if global_depths:
        metrics['global_depth'] = round(np.mean(global_depths), 3)
        metrics['global_depth_min'] = round(np.min(global_depths), 3)
        metrics['global_depth_max'] = round(np.max(global_depths), 3)
        metrics['global_view_count'] = len(global_depths)
    else:
        metrics['global_depth'] = MAX_DEPTH
        metrics['global_depth_min'] = MAX_DEPTH
        metrics['global_depth_max'] = MAX_DEPTH
        metrics['global_view_count'] = 0
    
    # 深度标准差（Depth Std）
    all_depths = []
    if local_depths:
        all_depths.extend(local_depths)
    if global_depths:
        all_depths.extend(global_depths)
    
    if all_depths:
        metrics['depth_std'] = round(np.std(all_depths), 3)
        metrics['depth_var'] = round(np.var(all_depths), 3)
    else:
        metrics['depth_std'] = 0.0
        metrics['depth_var'] = 0.0
    
    # 局部平均深度（Local Mean Depth R）
    if local_depths:
        # 文档中的定义：深度值的总和均值
        metrics['local_mean_depth_R'] = round(np.sum(local_depths) / max(1, len(local_depths)), 3)
    else:
        metrics['local_mean_depth_R'] = MAX_DEPTH
    
    # 其他辅助指标
    metrics['depth_range'] = metrics['local_depth_max'] - metrics['local_depth_min']
    metrics['depth_median'] = round(np.median(local_depths) if local_depths else MAX_DEPTH, 3)
    
    return metrics

# ----------------  分批处理主逻辑  ---------------- #
def process_depth_batches(vp_list, triangles, tri_kdt):
    """
    分批处理所有视点的深度指标计算
    """
    # 构建视点空间索引
    print("\n📇 构建视点空间索引...")
    vp_coords = [(vp['x'], vp['y']) for vp in vp_list]
    vp_kdt = cKDTree(vp_coords)
    
    # 加载已完成批次
    completed_batches = load_completed_batches()
    total_batches = math.ceil(len(vp_list) / BATCH_SIZE)
    
    print(f"\n📦 开始深度指标分批计算")
    print(f"   总视点数: {len(vp_list)}")
    print(f"   总批次: {total_batches}")
    print(f"   每批视点数: {BATCH_SIZE}")
    print(f"   局部半径: {LOCAL_RADIUS}米")
    print(f"   已完成批次: {len(completed_batches)}")
    
    # 逐批处理
    for batch_idx in range(total_batches):
        # 跳过已完成批次
        if RESUME and batch_idx in completed_batches:
            print(f"\n⏭️  批次 {batch_idx}/{total_batches} 已完成，跳过")
            continue
        
        # 提取本批次视点
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min((batch_idx + 1) * BATCH_SIZE, len(vp_list))
        batch_vps = vp_list[start_idx:end_idx]
        
        if not batch_vps:
            print(f"\n⚠️  批次 {batch_idx} 无数据，跳过")
            continue
        
        # 本批次计算
        print(f"\n🚀 处理批次 {batch_idx}/{total_batches}（视点 {start_idx}-{end_idx-1}，共 {len(batch_vps)} 个）")
        batch_results = []
        
        for vp in tqdm(batch_vps, desc=f"计算深度指标", leave=False):
            try:
                # 计算深度指标
                depth_metrics = calculate_depth_metrics_for_viewpoint(
                    vp, vp_list, vp_kdt, triangles, tri_kdt
                )
                
                # 合并原始视点信息和深度指标
                result = vp.copy()
                result.update(depth_metrics)
                batch_results.append(result)
                
            except Exception as e:
                print(f"\n❌ 视点 {vp['vp_id']} 深度计算失败：{str(e)[:100]}")
                # 失败时填充默认值
                result = vp.copy()
                result.update({
                    'local_depth': MAX_DEPTH,
                    'global_depth': MAX_DEPTH,
                    'depth_std': 0.0,
                    'local_mean_depth_R': MAX_DEPTH,
                    'local_depth_min': MAX_DEPTH,
                    'local_depth_max': MAX_DEPTH,
                    'global_depth_min': MAX_DEPTH,
                    'global_depth_max': MAX_DEPTH,
                    'local_view_count': 0,
                    'global_view_count': 0,
                    'depth_var': 0.0,
                    'depth_range': 0.0,
                    'depth_median': MAX_DEPTH
                })
                batch_results.append(result)
        
        # 保存本批次结果
        save_batch(batch_results, batch_idx)
        
        # 打印本批次统计
        if batch_results:
            df_batch = pd.DataFrame(batch_results)
            print(f"   本批次统计:")
            print(f"     局部深度均值: {df_batch['local_depth'].mean():.3f}")
            print(f"     全局深度均值: {df_batch['global_depth'].mean():.3f}")
            print(f"     局部平均深度R均值: {df_batch['local_mean_depth_R'].mean():.3f}")
    
    # 合并所有批次
    print("\n🔄 合并所有批次结果...")
    merge_batches(OUT_DIR)

# ----------------  主流程  ---------------- #
def main():
    print("="*70)
    print("🎯 3D空间深度指标专项计算")
    print("="*70)
    
    import time
    start_time = time.time()
    
    # 1. 读取数据
    print("\n📥 读取建筑与路网数据...")
    try:
        bld_gdf = gpd.read_file(BUILDING_SHP)
        road_gdf = gpd.read_file(ROAD_SHP)
    except Exception as e:
        print(f"❌ 数据读取失败：{e}")
        return
    
    # 2. 统一投影
    print("\n🌐 统一坐标系统为EPSG:3857...")
    target_crs = "EPSG:3857"
    if bld_gdf.crs and bld_gdf.crs.is_geographic:
        bld_gdf = bld_gdf.to_crs(target_crs)
    if road_gdf.crs and road_gdf.crs.is_geographic:
        road_gdf = road_gdf.to_crs(target_crs)
    
    # 3. 过滤无效几何
    print("\n🧹 过滤无效几何...")
    bld_gdf = bld_gdf[
        bld_gdf.geometry.notnull() & 
        bld_gdf.geometry.is_valid & 
        (bld_gdf.geometry.type == 'Polygon')
    ].reset_index(drop=True)
    
    road_gdf = road_gdf[
        road_gdf.geometry.notnull() & 
        road_gdf.geometry.is_valid & 
        (road_gdf.geometry.type == 'LineString')
    ].reset_index(drop=True)
    
    print(f"🏗️  有效建筑数：{len(bld_gdf)}")
    print(f"🛣️  有效路网数：{len(road_gdf)}")
    
    # 4. 建筑三角化
    print("\n🔨 建筑三角化处理...")
    triangles = buildings_to_triangles(bld_gdf)
    if not triangles:
        print("❌ 无有效建筑三角面，退出")
        return
    print(f"🔺 生成建筑三角面数：{len(triangles)}")
    
    # 5. 构建建筑三角形空间索引
    print("\n📇 构建建筑空间索引...")
    tri_centers_2d = [((t[0] + t[1] + t[2]) / 3.0)[:2] for t in triangles]
    tri_kdt = cKDTree(tri_centers_2d)
    
    # 6. 生成视点
    print("\n📍 生成路网视点...")
    vp_list = generate_viewpoints(road_gdf, height=EYE_HEIGHT)
    if not vp_list:
        print("❌ 无有效视点，退出")
        return
    
    # 7. 分批计算深度指标
    print("\n🚀 开始深度指标计算...")
    process_depth_batches(vp_list, triangles, tri_kdt)
    
    # 8. 程序结束
    elapsed_time = time.time() - start_time
    print("\n" + "="*70)
    print(f"✅ 深度指标计算完成！总耗时：{elapsed_time:.2f}秒")
    print(f"📁 结果保存至：{OUT_DIR}")
    print("\n📊 计算的深度指标：")
    print("  1. 局部深度（Local Depth）: 半径R内视点到建筑被看点的深度均值")
    print("  2. 全局深度（Global Depth）: 所有视点到建筑被看点的深度均值")
    print("  3. 深度标准差（Depth Std）: 深度值的离散程度")
    print("  4. 局部平均深度（Local Mean Depth R）: 半径R内深度值的总和均值")
    print("="*70)

if __name__ == '__main__':
    main()