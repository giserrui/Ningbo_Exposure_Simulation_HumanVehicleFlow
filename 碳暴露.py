#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
宁波市10米分辨率逐小时碳暴露计算系统 - KDE方法
使用高斯核密度估计生成5天120小时逐小时碳暴露
=============================================================================
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from rasterio.mask import mask
from rasterio.warp import reproject, Resampling
from shapely.geometry import LineString, Point
from shapely.ops import unary_union
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("🌍 宁波市10米分辨率逐小时碳暴露计算系统 (KDE方法)")
print("   5天 × 24小时 = 120小时")
print("=" * 70)

# ============================================================================
# 1. 配置
# ============================================================================
class Config:
    BOUNDARY_SHP = r"D:\宁波数据\宁波市边界合并.shp"
    POPULATION_TIF = r"D:\BaiduNetdiskDownload\【立方数据学社】宁波市\【立方数据学社】宁波市\chn_pop_2025_CN_100m_R2025A_v1.tif"
    CARBON_OUTPUT_DIR = r"D:\宁波市输出\Full_Road_Prediction_Optimized\Daily_Output"
    OUTPUT_DIR = r"C:\宁波市碳暴露成果\Carbon_Exposure\KDE_5days"
    
    TARGET_RESOLUTION = 10  # 10米分辨率
    DECAY_RADIUS = 300  # 扩散半径（米）
    GAUSSIAN_SIGMA = 50  # 高斯平滑sigma
    
    # 5天数据
    USE_DAYS = ["2025-11-05", "2025-11-06", "2025-11-07", "2025-11-08", "2025-11-09"]
    HOURS = list(range(24))


os.makedirs(Config.OUTPUT_DIR, exist_ok=True)


# ============================================================================
# 2. 数据加载
# ============================================================================
def load_boundary():
    print("\n📂 加载宁波市行政区边界...")
    boundary = gpd.read_file(Config.BOUNDARY_SHP, encoding='utf-8')
    if boundary.crs is None:
        boundary.set_crs("EPSG:4326", inplace=True)
    boundary_wgs84 = boundary.copy()
    boundary_proj = boundary.to_crs("EPSG:3857")
    print(f"  ✅ 边界加载完成")
    return boundary_wgs84, boundary_proj


def resample_population(boundary_wgs84, boundary_proj):
    """重采样人口到10米 - 只做一次"""
    print("\n📂 加载并重采样人口栅格...")
    
    with rasterio.open(Config.POPULATION_TIF) as src:
        # 裁剪
        boundary_4326 = boundary_wgs84.to_crs(src.crs)
        out_image, out_transform = mask(src, boundary_4326.geometry, crop=True, nodata=0)
        pop_data = out_image[0]
        pop_data = np.nan_to_num(pop_data, nan=0)
        pop_data = np.maximum(pop_data, 0)
        src_crs = src.crs
    
    # 获取投影边界
    bounds = boundary_proj.total_bounds
    minx, miny, maxx, maxy = bounds
    
    width = int(np.ceil((maxx - minx) / Config.TARGET_RESOLUTION))
    height = int(np.ceil((maxy - miny) / Config.TARGET_RESOLUTION))
    
    dst_transform = from_origin(minx, maxy, Config.TARGET_RESOLUTION, Config.TARGET_RESOLUTION)
    
    # 重采样
    pop_10m = np.zeros((height, width), dtype=np.float32)
    reproject(
        source=pop_data,
        destination=pop_10m,
        src_transform=out_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs="EPSG:3857",
        dst_nodata=0,
        resampling=Resampling.bilinear
    )
    
    pop_10m = np.nan_to_num(pop_10m, nan=0)
    pop_10m = np.maximum(pop_10m, 0)
    
    print(f"  ✅ 人口重采样完成")
    print(f"     形状: {pop_10m.shape}")
    print(f"     最大值: {pop_10m.max():.2f}")
    print(f"     非零像素: {np.sum(pop_10m > 0):,}")
    
    return pop_10m, dst_transform, height, width


def load_all_carbon_data(boundary_proj):
    """
    加载所有天、所有小时的碳排放数据
    返回: dict {date: {hour: {'geometries': [], 'carbon': []}}}
    """
    print(f"\n📂 加载5天逐小时碳排放数据...")
    
    all_data = {}
    
    for date_str in Config.USE_DAYS:
        shp_path = os.path.join(
            Config.CARBON_OUTPUT_DIR, 
            date_str, 
            f"完整路网碳排放_{date_str}.shp"
        )
        
        if not os.path.exists(shp_path):
            print(f"  ⚠️ 文件不存在: {shp_path}")
            continue
        
        try:
            gdf = gpd.read_file(shp_path, encoding='gbk')
            
            if gdf.crs is None:
                gdf.set_crs("EPSG:3857", inplace=True)
            elif str(gdf.crs) != "EPSG:3857":
                gdf = gdf.to_crs("EPSG:3857")
            
            # 裁剪到宁波市范围
            boundary_union = unary_union(boundary_proj.geometry)
            gdf = gdf[gdf.intersects(boundary_union)]
            
            if len(gdf) == 0:
                print(f"  ⚠️ {date_str}: 裁剪后无数据")
                continue
            
            # 提取所有小时的碳排放
            all_data[date_str] = {}
            
            for h in Config.HOURS:
                field = f'carbon_{h:02d}'
                if field in gdf.columns:
                    carbon_values = gdf[field].values
                    valid_mask = carbon_values > 0
                    
                    if np.any(valid_mask):
                        all_data[date_str][h] = {
                            'geometries': gdf.geometry.values[valid_mask],
                            'carbon': carbon_values[valid_mask]
                        }
            
            print(f"  ✅ {date_str}: {len(gdf)} 条路段, {len(all_data[date_str])} 个小时有数据")
            
        except Exception as e:
            print(f"  ⚠️ {date_str}: 加载失败 - {e}")
            continue
    
    # 统计
    total_hours = sum(len(data) for data in all_data.values())
    print(f"\n  📊 总计: {len(all_data)} 天, {total_hours} 个小时有数据")
    
    return all_data


# ============================================================================
# 3. KDE碳栅格生成
# ============================================================================
def generate_kde_carbon(geometries, carbon_values, transform, height, width):
    """
    使用高斯核密度估计生成碳栅格
    """
    if len(geometries) == 0:
        return np.zeros((height, width), dtype=np.float32)
    
    # 采样路段点
    points = []
    weights = []
    
    for geom, carbon in zip(geometries, carbon_values):
        if geom.is_empty or carbon <= 0:
            continue
        
        if isinstance(geom, LineString):
            length = geom.length
            # 采样间隔5米
            num_points = max(3, int(length / 5))
            distances = np.linspace(0, length, num_points)
            carbon_per_point = carbon / num_points
            
            for d in distances:
                pt = geom.interpolate(d)
                points.append((pt.x, pt.y))
                weights.append(carbon_per_point)
    
    points = np.array(points)
    weights = np.array(weights)
    
    if len(points) == 0:
        return np.zeros((height, width), dtype=np.float32)
    
    # 创建网格坐标
    x_coords = transform[2] + (np.arange(width) + 0.5) * transform[0]
    y_coords = transform[5] + (np.arange(height) + 0.5) * transform[4]
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    
    # 使用KD树进行核密度估计
    tree = cKDTree(points)
    sigma = Config.GAUSSIAN_SIGMA
    decay_radius = Config.DECAY_RADIUS
    
    # 分批处理
    batch_size = 5000
    n_grid = len(grid_points)
    carbon_kde = np.zeros(n_grid, dtype=np.float32)
    
    for i in tqdm(range(0, n_grid, batch_size), desc="  KDE进度", leave=False):
        batch_end = min(i + batch_size, n_grid)
        batch_points = grid_points[i:batch_end]
        
        # 查找附近采样点
        distances, indices = tree.query(
            batch_points,
            k=min(30, len(points)),
            distance_upper_bound=decay_radius
        )
        
        for j in range(len(batch_points)):
            valid_mask = distances[j] < np.inf
            if not np.any(valid_mask):
                continue
            
            valid_indices = indices[j][valid_mask]
            valid_distances = distances[j][valid_mask]
            
            # 高斯核权重
            kernel_weights = np.exp(-valid_distances ** 2 / (2 * sigma ** 2))
            total_weight = kernel_weights.sum() + 1e-8
            
            # 计算该点的碳密度
            carbon_kde[i + j] = np.sum(weights[valid_indices] * kernel_weights) / total_weight
    
    # 重塑为2D
    carbon_raster = carbon_kde.reshape(height, width)
    carbon_raster = np.nan_to_num(carbon_raster, nan=0)
    carbon_raster = np.maximum(carbon_raster, 0)
    
    return carbon_raster


# ============================================================================
# 4. 计算碳暴露
# ============================================================================
def calculate_exposure(carbon_raster, pop_10m):
    """计算碳暴露"""
    if carbon_raster.shape != pop_10m.shape:
        # 调整形状
        h, w = pop_10m.shape
        if carbon_raster.shape[0] > h:
            carbon_raster = carbon_raster[:h, :]
        elif carbon_raster.shape[0] < h:
            pad_h = h - carbon_raster.shape[0]
            carbon_raster = np.pad(carbon_raster, ((0, pad_h), (0, 0)), mode='constant')
        
        if carbon_raster.shape[1] > w:
            carbon_raster = carbon_raster[:, :w]
        elif carbon_raster.shape[1] < w:
            pad_w = w - carbon_raster.shape[1]
            carbon_raster = np.pad(carbon_raster, ((0, 0), (0, pad_w)), mode='constant')
    
    # 归一化碳值
    carbon_max = carbon_raster.max()
    if carbon_max > 0:
        carbon_normalized = carbon_raster / carbon_max
    else:
        carbon_normalized = carbon_raster
    
    # 碳暴露 = 人口密度 × 归一化碳值
    exposure = pop_10m * carbon_normalized
    
    # 轻微平滑
    exposure = gaussian_filter(exposure, sigma=0.5)
    exposure = np.maximum(exposure, 0)
    
    return exposure


# ============================================================================
# 5. 保存结果
# ============================================================================
def save_raster(data, output_path, transform, height, width):
    """保存栅格"""
    # 确保形状匹配
    if data.shape != (height, width):
        if data.shape[0] > height or data.shape[1] > width:
            data = data[:height, :width]
        else:
            temp = np.zeros((height, width), dtype=np.float32)
            temp[:data.shape[0], :data.shape[1]] = data
            data = temp
    
    meta = {
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'count': 1,
        'dtype': 'float32',
        'crs': "EPSG:3857",
        'transform': transform,
        'compress': 'lzw',
        'nodata': -9999,
        'tiled': True,
        'blockxsize': 256,
        'blockysize': 256
    }
    
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(data, 1)
    
    file_size = os.path.getsize(output_path) / 1024 / 1024
    return file_size


# ============================================================================
# 6. 处理单个小时
# ============================================================================
def process_hour(date_str, hour, data, pop_10m, transform, height, width):
    """处理单个小时的碳暴露"""
    output_dir_hour = os.path.join(Config.OUTPUT_DIR, date_str)
    os.makedirs(output_dir_hour, exist_ok=True)
    
    # 生成KDE碳栅格
    carbon_raster = generate_kde_carbon(
        data['geometries'], 
        data['carbon'], 
        transform, height, width
    )
    
    # 保存碳栅格
    carbon_path = os.path.join(output_dir_hour, f"carbon_kde_h{hour:02d}.tif")
    save_raster(carbon_raster, carbon_path, transform, height, width)
    
    # 计算碳暴露
    exposure = calculate_exposure(carbon_raster, pop_10m)
    
    # 保存碳暴露
    exposure_path = os.path.join(output_dir_hour, f"exposure_h{hour:02d}.tif")
    file_size = save_raster(exposure, exposure_path, transform, height, width)
    
    # 统计信息
    stats = {
        'carbon_max': carbon_raster.max(),
        'carbon_mean': carbon_raster.mean(),
        'carbon_nonzero': np.sum(carbon_raster > 0),
        'exposure_max': exposure.max(),
        'exposure_mean': exposure.mean(),
        'exposure_nonzero': np.sum(exposure > 0),
        'file_size': file_size
    }
    
    return stats


# ============================================================================
# 7. 生成报告
# ============================================================================
def generate_report(all_stats, pop_10m):
    """生成详细报告"""
    print("\n📊 生成统计报告...")
    
    report_path = os.path.join(Config.OUTPUT_DIR, "carbon_exposure_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("宁波市10米分辨率逐小时碳暴露计算报告 (KDE方法)\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("参数设置:\n")
        f.write(f"  空间分辨率: {Config.TARGET_RESOLUTION}米\n")
        f.write(f"  扩散半径: {Config.DECAY_RADIUS}米\n")
        f.write(f"  高斯平滑sigma: {Config.GAUSSIAN_SIGMA}\n")
        f.write(f"  处理天数: {', '.join(Config.USE_DAYS)}\n")
        f.write(f"  小时数: 24小时\n\n")
        
        f.write("人口统计:\n")
        f.write(f"  总人口: {pop_10m.sum():.0f}\n")
        f.write(f"  最大人口密度: {pop_10m.max():.2f}\n")
        f.write(f"  平均人口密度: {pop_10m.mean():.4f}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("逐日逐小时统计\n")
        f.write("=" * 80 + "\n\n")
        
        for date_str in sorted(all_stats.keys()):
            f.write(f"\n📅 {date_str}\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'小时':<6} {'碳最大值':<14} {'碳平均值':<14} {'暴露最大值':<14} {'暴露平均值':<14} {'文件大小'}\n")
            f.write("-" * 80 + "\n")
            
            for hour in sorted(all_stats[date_str].keys()):
                s = all_stats[date_str][hour]
                f.write(f"{hour:02d}时  {s['carbon_max']:>12.4f} {s['carbon_mean']:>12.4f} "
                       f"{s['exposure_max']:>12.4f} {s['exposure_mean']:>12.4f} {s['file_size']:>8.2f}MB\n")
            
            f.write("-" * 80 + "\n")
    
    print(f"  ✅ 报告已保存: {report_path}")


# ============================================================================
# 8. 主程序
# ============================================================================
def main():
    import time
    total_start = time.time()
    
    print("=" * 70)
    print("🌍 宁波市10米分辨率逐小时碳暴露计算系统 (KDE方法)")
    print("   5天 × 24小时 = 120小时")
    print("=" * 70)
    
    try:
        # 1. 加载边界
        boundary_wgs84, boundary_proj = load_boundary()
        
        # 2. 重采样人口（只做一次）
        pop_10m, transform, height, width = resample_population(boundary_wgs84, boundary_proj)
        
        # 保存人口栅格
        pop_path = os.path.join(Config.OUTPUT_DIR, "population_10m.tif")
        save_raster(pop_10m, pop_path, transform, height, width)
        print(f"  ✅ 人口栅格已保存: population_10m.tif")
        
        # 3. 加载所有碳排放数据
        all_data = load_all_carbon_data(boundary_proj)
        
        # 4. 处理每个小时
        print("\n" + "=" * 70)
        print("🚀 开始逐小时碳暴露计算 (KDE方法)")
        print("=" * 70)
        
        all_stats = {}
        total_hours = 0
        
        for date_str in Config.USE_DAYS:
            if date_str not in all_data:
                print(f"\n⚠️ {date_str}: 无数据，跳过")
                continue
            
            print(f"\n📅 处理日期: {date_str}")
            all_stats[date_str] = {}
            
            for hour in Config.HOURS:
                if hour not in all_data[date_str]:
                    print(f"  ⚠️ {date_str} {hour:02d}时: 无数据")
                    continue
                
                start_time = time.time()
                
                stats = process_hour(
                    date_str, hour, all_data[date_str][hour],
                    pop_10m, transform, height, width
                )
                
                elapsed = time.time() - start_time
                all_stats[date_str][hour] = stats
                total_hours += 1
                
                print(f"  ✅ {date_str} {hour:02d}时: "
                      f"碳max={stats['carbon_max']:.2f}, "
                      f"暴露max={stats['exposure_max']:.2f}, "
                      f"耗时={elapsed:.1f}s")
        
        # 5. 生成报告
        generate_report(all_stats, pop_10m)
        
        # 6. 总结
        total_time = time.time() - total_start
        print("\n" + "=" * 70)
        print("🎉 逐小时碳暴露计算完成!")
        print("=" * 70)
        print(f"📁 输出目录: {Config.OUTPUT_DIR}")
        print(f"⏱️ 总耗时: {total_time:.1f} 秒 ({total_time/60:.1f} 分钟)")
        print(f"📊 处理小时数: {total_hours}")
        
        # 列出输出文件结构
        print("\n📄 输出文件结构:")
        for date_str in sorted(all_stats.keys()):
            print(f"  📁 {date_str}/")
            for hour in sorted(all_stats[date_str].keys()):
                print(f"      - carbon_kde_h{hour:02d}.tif (碳栅格)")
                print(f"      - exposure_h{hour:02d}.tif (碳暴露)")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ 执行失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())