"""
完整流程: 路网SHP → 30米栅格 → XGBoost训练 → 全区域预测
"""

import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
import geopandas as gpd
from shapely.geometry import box
from shapely.ops import unary_union
import json
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 1. 配置
# ============================================

class Config:
    # ===== 输入数据 =====
    # 路网碳排放SHP目录 (步骤1的输入)
    ROAD_CO2_DIR = r"D:\深圳市数据\宁波市\公共路网_全天碳排放有效"
    
    # 宁波市边界
    BOUNDARY_PATH = r"D:\宁波数据\宁波市边界3857.shp"
    
    # 30米特征数据 (步骤2的特征输入)
    FEATURE_PATHS = {
        'dem': r"C:\Users\13227\Downloads\input_30m\dem_30m (1).tif",
        'slope': r"C:\Users\13227\Downloads\input_30m\slope_30m (1).tif",
        'ndvi': r"C:\Users\13227\Downloads\input_30m\NDVI_30m (1).tif",
        'ndbi': r"C:\Users\13227\Downloads\input_30m\NDBI_30m (1).tif",
        'ndmi': r"C:\Users\13227\Downloads\input_30m\NDMI_30m (1).tif",
        'ndwi': r"C:\Users\13227\Downloads\input_30m\NDWI_30m (1).tif",
        'alcc': r"C:\Users\13227\Downloads\input_30m\ALCC_30m (1).tif",
        'ahe': r"C:\Users\13227\Downloads\input_30m\ahe_30m (1).tif",
        'road': r"C:\Users\13227\Downloads\input_30m\road_eucdist_log (1).tif",
        'urban': r"C:\Users\13227\Downloads\input_30m\LandUseFraction\Urban_ratio.tif"
    }
    
    # ===== 中间输出 =====
    RASTER_OUTPUT_DIR = r"D:\宁波数据\road_carbon_rasters"
    os.makedirs(RASTER_OUTPUT_DIR, exist_ok=True)
    
    # ===== 最终输出 =====
    PREDICTION_OUTPUT_DIR = r"D:\宁波数据\prediction_results_road_carbon"
    os.makedirs(PREDICTION_OUTPUT_DIR, exist_ok=True)
    
    MODEL_DIR = r"D:\宁波数据\model_output"
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # ===== 参数 =====
    TARGET_CRS = 'EPSG:3857'
    TARGET_RESOLUTION = 30
    BLOCK_SIZE = 500
    
    # 日期列表
    DATE_LIST = ["2025-11-05", "2025-11-06", "2025-11-07", "2025-11-08", "2025-11-09"]
    N_HOURS = 24
    TARGET_HOUR = None  # None表示平均值
    
    # 训练参数
    TEST_SIZE = 0.25
    RANDOM_STATE = 42

config = Config()

# ============================================
# 2. 步骤1: 路网SHP → 30米栅格
# ============================================

def step1_shp_to_raster():
    """
    步骤1: 将路网碳排放SHP转换为30米栅格
    输入: D:\深圳市数据\宁波市\公共路网_全天碳排放有效\*.shp
    输出: D:\宁波数据\road_carbon_rasters\road_carbon_all_days.tif
    """
    print("\n" + "=" * 60)
    print("步骤1: 路网SHP → 30米栅格")
    print("=" * 60)
    
    # 加载边界
    gdf_boundary = gpd.read_file(config.BOUNDARY_PATH)
    if gdf_boundary.crs is None:
        gdf_boundary = gdf_boundary.set_crs('EPSG:4326')
    if gdf_boundary.crs != config.TARGET_CRS:
        gdf_boundary = gdf_boundary.to_crs(config.TARGET_CRS)
    geometry = unary_union(gdf_boundary.geometry)
    bounds = geometry.bounds
    
    # 创建栅格模板
    ref_path = list(config.FEATURE_PATHS.values())[0]
    with rasterio.open(ref_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, config.TARGET_CRS, src.width, src.height,
            bounds[0], bounds[1], bounds[2], bounds[3],
            resolution=config.TARGET_RESOLUTION
        )
    
    print(f"栅格尺寸: {width} x {height}")
    print(f"日期列表: {config.DATE_LIST}")
    
    # 逐天处理
    n_total_bands = len(config.DATE_LIST) * config.N_HOURS
    all_bands = np.zeros((n_total_bands, height, width), dtype=np.float32)
    
    band_idx = 0
    for date in config.DATE_LIST:
        shp_path = os.path.join(config.ROAD_CO2_DIR, f"公共路网_线段_{date}.shp")
        
        if not os.path.exists(shp_path):
            print(f"⚠️ 文件不存在: {shp_path}")
            continue
        
        # 读取SHP
        try:
            gdf = gpd.read_file(shp_path, encoding='utf-8')
        except:
            try:
                gdf = gpd.read_file(shp_path, encoding='gbk')
            except:
                gdf = gpd.read_file(shp_path)
        
        if gdf.crs != config.TARGET_CRS:
            if gdf.crs is None:
                gdf = gdf.set_crs('EPSG:4326')
            gdf = gdf.to_crs(config.TARGET_CRS)
        
        gdf_clipped = gdf.clip(geometry)
        print(f"  {date}: {len(gdf_clipped)} 条路段在边界内")
        
        # 逐小时栅格化
        for hour in range(config.N_HOURS):
            col_name = f'co2_{hour:02d}'
            
            if col_name in gdf_clipped.columns:
                # 准备栅格化数据
                shapes = []
                for idx, row in gdf_clipped.iterrows():
                    val = row[col_name]
                    if pd.notna(val) and val > 0:
                        shapes.append((row.geometry, float(val)))
                
                if shapes:
                    rasterized = rasterize(
                        shapes,
                        out_shape=(height, width),
                        transform=transform,
                        fill=-9999,
                        dtype=np.float32,
                        all_touched=True
                    )
                    all_bands[band_idx, :, :] = rasterized
                else:
                    all_bands[band_idx, :, :] = -9999
            else:
                all_bands[band_idx, :, :] = -9999
            
            band_idx += 1
    
    # 保存多波段栅格
    output_path = os.path.join(config.RASTER_OUTPUT_DIR, "road_carbon_all_days.tif")
    
    meta = {
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'count': n_total_bands,
        'dtype': 'float32',
        'transform': transform,
        'crs': config.TARGET_CRS,
        'compress': 'lzw',
        'nodata': -9999
    }
    
    with rasterio.open(output_path, 'w', **meta) as dst:
        for band in range(n_total_bands):
            dst.write(all_bands[band, :, :], band + 1)
    
    print(f"\n✅ 步骤1完成: {output_path}")
    print(f"   总波段数: {n_total_bands} ({len(config.DATE_LIST)}天 × {config.N_HOURS}小时)")
    print(f"   文件大小: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    
    return output_path

# ============================================
# 3. 步骤2-4: 训练和预测
# ============================================

def load_raster_data(carbon_path, target_hour=None):
    """加载栅格数据"""
    with rasterio.open(carbon_path) as src:
        n_bands = src.count
        height = src.height
        width = src.width
        transform = src.transform
        crs = src.crs
        
        if target_hour is not None:
            # 提取特定小时
            hour_indices = list(range(target_hour, n_bands, 24))
            hour_data = []
            for idx in hour_indices:
                band_data = src.read(idx + 1)
                hour_data.append(band_data)
            carbon_raster = np.mean(hour_data, axis=0)
        else:
            # 所有小时平均
            all_data = src.read()
            carbon_raster = np.mean(all_data, axis=0)
        
        carbon_raster = np.nan_to_num(carbon_raster, nan=0.0)
        carbon_raster = np.maximum(carbon_raster, 0)
        
        return carbon_raster, transform, crs

def load_features(feature_paths, target_shape):
    """加载特征数据"""
    features = {}
    for name, path in feature_paths.items():
        with rasterio.open(path) as src:
            data = src.read(1)
            if data.shape != target_shape:
                from scipy.ndimage import zoom
                zoom_factor = (target_shape[0] / data.shape[0], 
                              target_shape[1] / data.shape[1])
                data = zoom(data, zoom_factor, order=1)
            data = np.nan_to_num(data, nan=0.0)
            features[name] = data
    return features

def clip_to_boundary(raster_data, transform, geometry):
    """裁剪到边界"""
    from rasterio.io import MemoryFile
    
    with MemoryFile() as memfile:
        with memfile.open(
            driver='GTiff',
            height=raster_data.shape[0],
            width=raster_data.shape[1],
            count=1,
            dtype=raster_data.dtype,
            crs=config.TARGET_CRS,
            transform=transform,
        ) as src:
            src.write(raster_data, 1)
            out_image, out_transform = mask(src, [geometry], crop=True)
    
    return out_image[0], out_transform

def train_xgboost_model(carbon_raster, features):
    """步骤3: 训练XGBoost模型"""
    print("\n" + "=" * 60)
    print("步骤3: 训练XGBoost模型")
    print("=" * 60)
    
    # 提取有效像素 (路网覆盖区域)
    valid_mask = carbon_raster > 0
    valid_indices = np.where(valid_mask)
    n_valid = len(valid_indices[0])
    print(f"有效像素数: {n_valid:,}")
    
    # 构建训练数据
    feature_names = list(features.keys())
    rows = valid_indices[0]
    cols = valid_indices[1]
    
    X = np.zeros((n_valid, len(feature_names)), dtype=np.float32)
    y = np.zeros(n_valid, dtype=np.float32)
    
    for i, (row, col) in enumerate(tqdm(zip(rows, cols), desc="构建训练数据", total=n_valid)):
        for j, name in enumerate(feature_names):
            X[i, j] = features[name][row, col]
        y[i] = carbon_raster[row, col]
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    
    print(f"训练集: {len(X_train):,} 样本")
    print(f"测试集: {len(X_test):,} 样本")
    
    # 训练模型
    print("\n训练XGBoost模型...")
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=config.RANDOM_STATE,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train, verbose=False)
    
    # 评估
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    metrics = {
        'train_r2': float(r2_score(y_train, y_pred_train)),
        'test_r2': float(r2_score(y_test, y_pred_test)),
        'train_rmse': float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
        'test_rmse': float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
        'train_mae': float(mean_absolute_error(y_train, y_pred_train)),
        'test_mae': float(mean_absolute_error(y_test, y_pred_test))
    }
    
    print("\n=== 模型评估结果 ===")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # 特征重要性
    importance = model.feature_importances_
    feature_importance = {}
    for name, imp in zip(feature_names, importance):
        feature_importance[name] = float(imp)
        print(f"  {name}: {imp:.4f}")
    
    # 保存模型
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'metrics': metrics,
        'feature_importance': feature_importance
    }
    
    model_path = os.path.join(config.MODEL_DIR, 'model_road_carbon.pkl')
    joblib.dump(model_data, model_path)
    print(f"\n模型已保存: {model_path}")
    
    return model, scaler, feature_names, metrics, feature_importance

def predict_full_raster(model, scaler, feature_names, features, transform, geometry):
    """步骤4: 全区域预测"""
    print("\n" + "=" * 60)
    print("步骤4: 全区域预测")
    print("=" * 60)
    
    height, width = features[feature_names[0]].shape
    prediction = np.full((height, width), np.nan, dtype=np.float32)
    block_size = config.BLOCK_SIZE
    
    n_blocks_row = (height + block_size - 1) // block_size
    n_blocks_col = (width + block_size - 1) // block_size
    print(f"总块数: {n_blocks_row * n_blocks_col} ({n_blocks_row} x {n_blocks_col})")
    
    for row_start in tqdm(range(0, height, block_size), desc="预测进度"):
        row_end = min(row_start + block_size, height)
        for col_start in range(0, width, block_size):
            col_end = min(col_start + block_size, width)
            
            h = row_end - row_start
            w = col_end - col_start
            n_pixels = h * w
            
            X_block = np.zeros((n_pixels, len(feature_names)), dtype=np.float32)
            
            for i, name in enumerate(feature_names):
                block_data = features[name][row_start:row_end, col_start:col_end].flatten()
                X_block[:, i] = block_data
            
            X_block = np.nan_to_num(X_block, nan=0.0)
            
            try:
                X_scaled = scaler.transform(X_block)
            except:
                X_scaled = X_block
            
            try:
                block_pred = model.predict(X_scaled)
                block_pred = np.maximum(block_pred, 0)
                prediction[row_start:row_end, col_start:col_end] = block_pred.reshape(h, w)
            except:
                prediction[row_start:row_end, col_start:col_end] = 0
    
    # 裁剪到边界
    prediction_clipped, transform_clipped = clip_to_boundary(
        prediction, transform, geometry
    )
    
    valid_pixels = prediction_clipped[prediction_clipped > 0]
    print(f"\n预测完成!")
    print(f"  尺寸: {prediction_clipped.shape}")
    if len(valid_pixels) > 0:
        print(f"  范围: {valid_pixels.min():.2f} - {valid_pixels.max():.2f}")
        print(f"  均值: {valid_pixels.mean():.2f}")
    
    return prediction_clipped, transform_clipped

# ============================================
# 4. 主程序
# ============================================

def main():
    print("=" * 60)
    print("完整预测流程: 路网SHP → 30米栅格 → XGBoost训练 → 全区域预测")
    print("=" * 60)
    
    # ===== 步骤1: 路网SHP → 30米栅格 =====
    carbon_tif_path = step1_shp_to_raster()
    
    # ===== 加载边界 =====
    gdf_boundary = gpd.read_file(config.BOUNDARY_PATH)
    if gdf_boundary.crs is None:
        gdf_boundary = gdf_boundary.set_crs('EPSG:4326')
    if gdf_boundary.crs != config.TARGET_CRS:
        gdf_boundary = gdf_boundary.to_crs(config.TARGET_CRS)
    geometry = unary_union(gdf_boundary.geometry)
    
    # ===== 步骤2: 加载数据 =====
    print("\n" + "=" * 60)
    print("步骤2: 加载数据")
    print("=" * 60)
    
    # 加载碳排放栅格
    carbon_raster, carbon_transform, carbon_crs = load_raster_data(
        carbon_tif_path, config.TARGET_HOUR
    )
    
    # 裁剪到边界
    carbon_clipped, carbon_transform = clip_to_boundary(
        carbon_raster, carbon_transform, geometry
    )
    print(f"碳排放数据: {carbon_clipped.shape}, 有效像素: {np.sum(carbon_clipped > 0):,}")
    
    # 加载特征
    features = load_features(config.FEATURE_PATHS, carbon_clipped.shape)
    
    # ===== 步骤3: 训练模型 =====
    model, scaler, feature_names, metrics, feature_importance = train_xgboost_model(
        carbon_clipped, features
    )
    
    # ===== 步骤4: 全区域预测 =====
    prediction, transform = predict_full_raster(
        model, scaler, feature_names, features, carbon_transform, geometry
    )
    
    # ===== 保存结果 =====
    print("\n" + "=" * 60)
    print("保存结果")
    print("=" * 60)
    
    output_tif = os.path.join(config.PREDICTION_OUTPUT_DIR, "ningbo_carbon_30m_prediction.tif")
    meta = {
        'driver': 'GTiff',
        'height': prediction.shape[0],
        'width': prediction.shape[1],
        'count': 1,
        'dtype': 'float32',
        'transform': transform,
        'crs': config.TARGET_CRS,
        'compress': 'lzw',
        'nodata': -9999
    }
    with rasterio.open(output_tif, 'w', **meta) as dst:
        dst.write(prediction.astype('float32'), 1)
    print(f"预测结果已保存: {output_tif}")
    
    print("\n" + "=" * 60)
    print("✅ 所有步骤完成!")
    print("=" * 60)

if __name__ == "__main__":
    main()
