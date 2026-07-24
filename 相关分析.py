#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
公共路网碳排放与空间句法、建成环境指标相关性分析
分别绘制空间句法指标和建成环境指标的相关性热力图
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import lightgbm as lgb
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# ===================== 路径配置 =====================
BASE_DIR = r"D:\宁波数据\宁波市\公共路网_全天碳排放有效"
SYNTAX_CSV = r"C:\宁波市成果\完整路网空间句法指标.csv"
BUILT_ENV_CSV = r"D:\宁波数据\宁波市\宁波市道路路网_10核心指标.csv"
OUTPUT_DIR = r"D:\宁波市输出\相关性分析结果_公共路网"

os.makedirs(OUTPUT_DIR, exist_ok=True)

N_HOURS = 24
DATE_LIST = ["2025-11-05", "2025-11-06", "2025-11-07", "2025-11-08", "2025-11-09"]

# 建成环境字段
BUILT_ENV_FIELDS = ['res_ratio', 'com_ratio', 'mix_build', 'mix_land', 
                    'gen_idx', 'attr_idx', 'cpi', 'ti_score', 'risk_lvl', 'bld_cnt']

# 空间句法字段映射
SYNTAX_FIELD_MAPPING = {
    'local_mean_depth_R': 'local_mean_depth_R',
    'local_integration_R': 'local_integration_R',
    'visibility_ratio': 'visibility_ratio',
    'transparency_visibility': 'transparency_visibility',
    'neighbor_count': 'neighbor_count',
    'depth_sample_count': 'depth_sample_count',
    'Angular_Co': 'Angular Co',
    'Connectivi': 'Connectivi',
    'T1024_Choi': 'T1024 Choi',
    'T1024_Inte': 'T1024 Inte',
    'T1024_Tota': 'T1024 Tota',
    'Choice': 'Choice',
    'Harmonic_M': 'Harmonic M',
    'Integratio': 'Integratio',
    'Intensity': 'Intensity',
    'Mean_Depth': 'Mean Depth',
    'Node_Count': 'Node Count'
}

# 空间句法字段中文名称
SYNTAX_CHINESE = {
    'local_mean_depth_R': '局部平均深度',
    'local_integration_R': '局部整合度',
    'visibility_ratio': '可视比率',
    'transparency_visibility': '透明可视性',
    'neighbor_count': '邻域节点数',
    'depth_sample_count': '深度样本数',
    'Angular_Co': '角度连接度',
    'Connectivi': '连接度',
    'T1024_Choi': '1024尺度选择度',
    'T1024_Inte': '1024尺度整合度',
    'T1024_Tota': '1024尺度总整合度',
    'Choice': '选择度',
    'Harmonic_M': '调和均值',
    'Integratio': '整合度',
    'Intensity': '强度',
    'Mean_Depth': '平均深度',
    'Node_Count': '节点数量'
}

# 建成环境字段中文名称
BUILT_ENV_CHINESE = {
    'res_ratio': '居住用地占比',
    'com_ratio': '商业用地占比',
    'mix_build': '建筑混合度',
    'mix_land': '土地利用混合度',
    'gen_idx': '综合指数',
    'attr_idx': '吸引力指数',
    'cpi': '紧凑度指数',
    'ti_score': '交通影响评分',
    'risk_lvl': '风险等级',
    'bld_cnt': '建筑数量'
}


def load_public_road_data():
    """加载公共路网数据"""
    print("=" * 60)
    print("📂 加载公共路网数据")
    print("=" * 60)
    
    # 加载公共路网
    shp_files = [f for f in os.listdir(BASE_DIR) if f.endswith('.shp')]
    base_shp = os.path.join(BASE_DIR, shp_files[0])
    gdf = gpd.read_file(base_shp, encoding='utf-8')
    
    fid_col = None
    for col in ['FID', 'fid', 'ID', 'id']:
        if col in gdf.columns:
            fid_col = col
            break
    if fid_col is None:
        gdf['FID'] = range(len(gdf))
        fid_col = 'FID'
    
    public_fids = set(gdf[fid_col].values)
    print(f"   ✅ 公共路网记录: {len(public_fids)} 条")
    
    # 加载空间句法
    syntax_df = pd.read_csv(SYNTAX_CSV, encoding='utf-8')
    syntax_public = syntax_df[syntax_df['fid'].isin(public_fids)].copy()
    
    syntax_features = pd.DataFrame()
    syntax_features['fid'] = syntax_public['fid']
    for field, csv_field in SYNTAX_FIELD_MAPPING.items():
        if csv_field in syntax_public.columns:
            syntax_features[field] = syntax_public[csv_field]
    
    print(f"   ✅ 空间句法特征: {len(syntax_features.columns) - 1} 个")
    
    # 加载建成环境
    built_df = pd.read_csv(BUILT_ENV_CSV, encoding='utf-8')
    fid_col_built = 'FID' if 'FID' in built_df.columns else 'fid'
    built_public = built_df[built_df[fid_col_built].isin(public_fids)].copy()
    
    built_features = pd.DataFrame()
    built_features['fid'] = built_public[fid_col_built]
    for field in BUILT_ENV_FIELDS:
        if field in built_public.columns:
            built_features[field] = built_public[field]
    
    print(f"   ✅ 建成环境特征: {len(built_features.columns) - 1} 个")
    
    # 加载碳排放
    carbon_dicts = []
    for date in DATE_LIST:
        shp_path = os.path.join(BASE_DIR, f"公共路网_线段_{date}.shp")
        if os.path.exists(shp_path):
            day_gdf = gpd.read_file(shp_path, encoding='utf-8')
            day_fid_col = [c for c in ['FID', 'fid', 'ID', 'id'] if c in day_gdf.columns][0]
            co2_cols = [f'co2_{h:02d}' for h in range(N_HOURS)]
            available_co2 = [c for c in co2_cols if c in day_gdf.columns]
            if available_co2:
                day_gdf['daily_co2'] = day_gdf[available_co2].mean(axis=1)
                day_public = day_gdf[day_gdf[day_fid_col].isin(public_fids)]
                carbon_dicts.append(day_public.set_index(day_fid_col)['daily_co2'].to_dict())
    
    carbon_data = pd.DataFrame()
    carbon_data['fid'] = list(public_fids)
    carbon_avg = []
    for fid in public_fids:
        values = [d.get(fid, np.nan) for d in carbon_dicts if d.get(fid) is not None]
        if values:
            carbon_avg.append(np.mean(values))
        else:
            carbon_avg.append(np.nan)
    carbon_data['carbon_avg'] = carbon_avg
    carbon_data = carbon_data.dropna()
    
    print(f"   ✅ 碳排放数据: {len(carbon_data)} 条记录")
    print(f"   碳排放均值: {carbon_data['carbon_avg'].mean():.2f} g")
    
    # 合并数据
    merged = pd.merge(syntax_features, built_features, on='fid', how='inner')
    merged = pd.merge(merged, carbon_data, on='fid', how='inner')
    merged = merged.set_index('fid')
    
    for col in merged.columns:
        if merged[col].dtype in ['float64', 'int64']:
            merged[col] = merged[col].fillna(merged[col].median())
    
    print(f"\n📊 最终数据: {len(merged)} 条, {len(merged.columns)} 个特征")
    
    return merged


def plot_individual_heatmaps(data_df):
    """分别绘制碳排放与空间句法、建成环境指标的相关性热力图"""
    print("\n" + "=" * 60)
    print("📈 生成相关性热力图")
    print("=" * 60)
    
    # 获取各类型特征
    syntax_cols = [c for c in data_df.columns if c in SYNTAX_FIELD_MAPPING.keys()]
    built_cols = [c for c in data_df.columns if c in BUILT_ENV_FIELDS]
    target_col = 'carbon_avg'
    
    # ========== 图1：碳排放 vs 空间句法指标 ==========
    print("\n🔹 生成碳排放 vs 空间句法指标热力图...")
    
    # 准备数据
    syntax_data = data_df[syntax_cols + [target_col]].copy()
    
    # 计算相关系数
    corr_syntax = syntax_data.corr(method='spearman')
    
    # 提取碳排放与其他指标的相关性
    corr_with_carbon_syntax = corr_syntax[target_col].drop(target_col).sort_values(ascending=False)
    
    # 按相关系数排序
    sorted_cols = corr_with_carbon_syntax.index.tolist()
    corr_syntax_sorted = corr_syntax.loc[sorted_cols + [target_col], sorted_cols + [target_col]]
    
    # 创建图形
    fig1, ax1 = plt.subplots(figsize=(12, 10), dpi=300)
    
    # 绘制热力图
    mask1 = np.triu(np.ones_like(corr_syntax_sorted, dtype=bool), k=1)
    
    # 使用更明显的颜色映射
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    
    heatmap1 = sns.heatmap(
        corr_syntax_sorted,
        mask=mask1,
        cmap=cmap,
        center=0,
        annot=True,
        fmt='.3f',
        annot_kws={'size': 9},
        square=True,
        linewidths=0.5,
        cbar_kws={'shrink': 0.8, 'label': 'Spearman相关系数 (ρ)', 'aspect': 30},
        ax=ax1
    )
    
    # 修改x轴和y轴标签为中文
    labels_syntax = [SYNTAX_CHINESE.get(col, col) for col in corr_syntax_sorted.columns]
    ax1.set_xticklabels(labels_syntax, rotation=45, ha='right', fontsize=10)
    ax1.set_yticklabels(labels_syntax, rotation=0, fontsize=10)
    
    # 高亮碳排放列/行
    carbon_idx = list(corr_syntax_sorted.columns).index('carbon_avg')
    for text in ax1.texts:
        # 高亮碳排放行的标注
        pass
    
    ax1.set_title('公共路网碳排放与空间句法指标相关性热力图', fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlabel('空间句法指标', fontsize=12)
    ax1.set_ylabel('空间句法指标', fontsize=12)
    
    plt.tight_layout()
    output_path1 = os.path.join(OUTPUT_DIR, 'fig1_correlation_syntax_heatmap.png')
    plt.savefig(output_path1, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   ✅ 保存: {output_path1}")
    plt.close()
    
    # 打印排序结果
    print("\n   📊 空间句法指标相关性排序:")
    for col, rho in corr_with_carbon_syntax.items():
        chinese_name = SYNTAX_CHINESE.get(col, col)
        sig = "***" if abs(rho) > 0.2 else "**" if abs(rho) > 0.15 else "*" if abs(rho) > 0.1 else ""
        print(f"      {chinese_name:15s}: ρ={rho:8.4f} {sig}")
    
    # ========== 图2：碳排放 vs 建成环境指标 ==========
    print("\n🔹 生成碳排放 vs 建成环境指标热力图...")
    
    # 准备数据
    built_data = data_df[built_cols + [target_col]].copy()
    
    # 计算相关系数
    corr_built = built_data.corr(method='spearman')
    
    # 提取碳排放与其他指标的相关性
    corr_with_carbon_built = corr_built[target_col].drop(target_col).sort_values(ascending=False)
    
    # 按相关系数排序
    sorted_cols_built = corr_with_carbon_built.index.tolist()
    corr_built_sorted = corr_built.loc[sorted_cols_built + [target_col], sorted_cols_built + [target_col]]
    
    # 创建图形
    fig2, ax2 = plt.subplots(figsize=(12, 10), dpi=300)
    
    # 绘制热力图
    mask2 = np.triu(np.ones_like(corr_built_sorted, dtype=bool), k=1)
    
    heatmap2 = sns.heatmap(
        corr_built_sorted,
        mask=mask2,
        cmap=cmap,
        center=0,
        annot=True,
        fmt='.3f',
        annot_kws={'size': 10},
        square=True,
        linewidths=0.5,
        cbar_kws={'shrink': 0.8, 'label': 'Spearman相关系数 (ρ)', 'aspect': 30},
        ax=ax2
    )
    
    # 修改x轴和y轴标签为中文
    labels_built = [BUILT_ENV_CHINESE.get(col, col) for col in corr_built_sorted.columns]
    ax2.set_xticklabels(labels_built, rotation=45, ha='right', fontsize=10)
    ax2.set_yticklabels(labels_built, rotation=0, fontsize=10)
    
    ax2.set_title('公共路网碳排放与建成环境指标相关性热力图', fontsize=14, fontweight='bold', pad=20)
    ax2.set_xlabel('建成环境指标', fontsize=12)
    ax2.set_ylabel('建成环境指标', fontsize=12)
    
    plt.tight_layout()
    output_path2 = os.path.join(OUTPUT_DIR, 'fig2_correlation_built_heatmap.png')
    plt.savefig(output_path2, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   ✅ 保存: {output_path2}")
    plt.close()
    
    # 打印排序结果
    print("\n   📊 建成环境指标相关性排序:")
    for col, rho in corr_with_carbon_built.items():
        chinese_name = BUILT_ENV_CHINESE.get(col, col)
        sig = "***" if abs(rho) > 0.2 else "**" if abs(rho) > 0.15 else "*" if abs(rho) > 0.1 else ""
        print(f"      {chinese_name:15s}: ρ={rho:8.4f} {sig}")
    
    # ========== 图3：碳排放相关性对比条形图 ==========
    print("\n🔹 生成相关性对比条形图...")
    
    fig3, ax3 = plt.subplots(figsize=(14, 10), dpi=300)
    
    # 合并两种类型的相关性结果
    syntax_corr = corr_with_carbon_syntax.to_dict()
    built_corr = corr_with_carbon_built.to_dict()
    all_corr = {**syntax_corr, **built_corr}
    all_corr_sorted = dict(sorted(all_corr.items(), key=lambda x: x[1]))
    
    # 准备绘图数据
    features = list(all_corr_sorted.keys())
    rhos = list(all_corr_sorted.values())
    
    # 设置颜色：空间句法为蓝色系，建成环境为红色系
    colors = []
    for f in features:
        if f in syntax_cols:
            colors.append('#2E86AB')  # 空间句法 - 蓝色
        else:
            colors.append('#D64933')  # 建成环境 - 红色
    
    bars = ax3.barh(features, rhos, color=colors, edgecolor='black', linewidth=0.5)
    
    # 添加显著性标记
    for i, (f, rho) in enumerate(zip(features, rhos)):
        if abs(rho) > 0.2:
            sig = '***'
        elif abs(rho) > 0.15:
            sig = '**'
        elif abs(rho) > 0.1:
            sig = '*'
        else:
            sig = ''
        if sig:
            x_pos = rho + (0.02 if rho > 0 else -0.02)
            ax3.text(x_pos, i, sig, va='center', ha='left' if rho > 0 else 'right',
                    fontsize=11, fontweight='bold')
    
    ax3.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax3.set_xlabel('Spearman相关系数 (ρ)', fontsize=12)
    ax3.set_ylabel('特征指标', fontsize=12)
    ax3.set_title('公共路网碳排放与各指标相关性对比', fontsize=14, fontweight='bold')
    
    # 设置y轴标签为中文
    y_labels = []
    for f in features:
        if f in SYNTAX_CHINESE:
            y_labels.append(SYNTAX_CHINESE[f])
        elif f in BUILT_ENV_CHINESE:
            y_labels.append(BUILT_ENV_CHINESE[f])
        else:
            y_labels.append(f)
    ax3.set_yticklabels(y_labels, fontsize=9)
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2E86AB', label='空间句法指标', edgecolor='black'),
        Patch(facecolor='#D64933', label='建成环境指标', edgecolor='black')
    ]
    ax3.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    ax3.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_path3 = os.path.join(OUTPUT_DIR, 'fig3_correlation_comparison.png')
    plt.savefig(output_path3, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   ✅ 保存: {output_path3}")
    plt.close()
    
    # ========== 图4：碳排放 vs 两类指标综合热力图 ==========
    print("\n🔹 生成综合热力图...")
    
    # 选择相关性最强的Top特征
    all_corr_sorted_items = sorted(all_corr.items(), key=lambda x: abs(x[1]), reverse=True)
    top_features = [item[0] for item in all_corr_sorted_items[:20]]
    selected_cols = [target_col] + top_features
    
    corr_selected = data_df[selected_cols].corr(method='spearman')
    
    # 按相关性排序
    corr_with_target = corr_selected[target_col].drop(target_col).sort_values(ascending=False)
    sorted_cols_all = corr_with_target.index.tolist()
    corr_selected_sorted = corr_selected.loc[sorted_cols_all + [target_col], sorted_cols_all + [target_col]]
    
    fig4, ax4 = plt.subplots(figsize=(14, 12), dpi=300)
    
    mask4 = np.triu(np.ones_like(corr_selected_sorted, dtype=bool), k=1)
    
    heatmap4 = sns.heatmap(
        corr_selected_sorted,
        mask=mask4,
        cmap='RdBu_r',
        center=0,
        annot=True,
        fmt='.3f',
        annot_kws={'size': 8},
        square=True,
        linewidths=0.5,
        cbar_kws={'shrink': 0.8, 'label': 'Spearman相关系数 (ρ)'},
        ax=ax4
    )
    
    # 设置标签
    labels_all = []
    for col in corr_selected_sorted.columns:
        if col in SYNTAX_CHINESE:
            labels_all.append(SYNTAX_CHINESE[col])
        elif col in BUILT_ENV_CHINESE:
            labels_all.append(BUILT_ENV_CHINESE[col])
        else:
            labels_all.append(col)
    
    ax4.set_xticklabels(labels_all, rotation=45, ha='right', fontsize=8)
    ax4.set_yticklabels(labels_all, rotation=0, fontsize=8)
    
    ax4.set_title('公共路网碳排放与关键特征综合相关性热力图', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    output_path4 = os.path.join(OUTPUT_DIR, 'fig4_correlation_comprehensive.png')
    plt.savefig(output_path4, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   ✅ 保存: {output_path4}")
    plt.close()
    
    print("\n✅ 所有热力图生成完成！")
    
    return corr_syntax_sorted, corr_built_sorted


def main():
    """主函数"""
    print("=" * 60)
    print("🔬 公共路网碳排放相关性分析 - 分类型热力图")
    print("=" * 60)
    
    # 加载数据
    data_df = load_public_road_data()
    
    # 生成热力图
    corr_syntax, corr_built = plot_individual_heatmaps(data_df)
    
    print("\n" + "=" * 60)
    print("📋 分析完成")
    print("=" * 60)
    print(f"\n📁 输出目录: {OUTPUT_DIR}")
    print("\n生成的文件:")
    print("   fig1_correlation_syntax_heatmap.png - 空间句法指标热力图")
    print("   fig2_correlation_built_heatmap.png - 建成环境指标热力图")
    print("   fig3_correlation_comparison.png - 相关性对比条形图")
    print("   fig4_correlation_comprehensive.png - 综合热力图")


if __name__ == "__main__":
    main()