#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
✅ 输出：每小时带 速度 / 车流量 / 碳排放 的 SHP 轨迹文件
✅ 无路网 · 保留 r_id · 每条轨迹独立计算
✅ 输出 SHP 可直接 ArcGIS / QGIS 打开画图
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import warnings
warnings.filterwarnings('ignore')

# ===================== MOVES 参数 =====================
A = 0.134
C = 0.000302
IDLE_SPEED_THRESHOLD = 0.2

EF_CO2 = {
    0: 185, 1: 240, 2: 225, 3: 210, 4: 195, 5: 180, 6: 170, 7: 165,
    8: 160, 9: 158, 10: 160, 11: 165, 12: 175, 13: 195, 14: 230
}

# ===================== 路径 =====================
TRAJECTORY_BASE = r"D:\深圳市数据\宁波市\宁波市202511\宁波市202511"
DATE_FOLDERS = ["宁波2025.11.05","宁波2025.11.06","宁波2025.11.07","宁波2025.11.08","宁波2025.11.09"]
OUTPUT_FOLDER = r"D:\深圳市数据\宁波市\出租车碳排放_每小时SHP"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ===================== 单值计算 VSP BIN =====================
def get_vsp_bin(speed_ms, accel):
    speed_ms = float(speed_ms)
    accel = float(accel)
    if speed_ms < IDLE_SPEED_THRESHOLD:
        return 1
    vsp = speed_ms * accel + A * speed_ms + C * (speed_ms ** 3)
    bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    for i, threshold in enumerate(bins):
        if vsp < threshold:
            return i
    return 14

# ===================== 单条计算 CO2 =====================
def calc_co2_single(speed_ms, accel, length_m, status):
    b = get_vsp_bin(speed_ms, accel)
    ef = EF_CO2.get(b, 160)
    weight = 1.1 if status == 1 else 1.0
    co2 = ef * (length_m / 1000.0) * weight
    return round(co2, 2)

# ===================== 处理文件（输出 SHP） =====================
def process_one_file(shp_path):
    try:
        gdf = gpd.read_file(shp_path, encoding="utf-8")
    except:
        return None

    # 保存原始几何
    geom = gdf.geometry.copy()
    out_df = pd.DataFrame(gdf.drop(columns="geometry"))

    # 长度
    gdf_proj = gdf.to_crs("EPSG:3857")
    out_df["length_m"] = gdf_proj.geometry.length.round(2)

    # 速度
    speed_kmh = []
    for s in out_df["speed"]:
        try:
            v = float(str(s).strip())
            speed_kmh.append(round(np.clip(v, 0, 120), 2))
        except:
            speed_kmh.append(0.0)
    out_df["speed_kmh"] = speed_kmh
    out_df["speed_ms"] = (out_df["speed_kmh"] / 3.6).round(4)

    # 加速度（按车辆 r_id）
    out_df["accel"] = 0.0
    if "r_id" in out_df.columns:
        for rid, group in out_df.groupby("r_id"):
            if len(group) < 2:
                continue
            spd = group["speed_ms"].values
            accel = np.diff(spd, prepend=spd[0])
            accel = np.clip(accel, -10, 10)
            out_df.loc[group.index, "accel"] = accel.round(4)

    # 逐条算碳排放
    co2_list = []
    for i in range(len(out_df)):
        co2 = calc_co2_single(
            out_df["speed_ms"].iloc[i],
            out_df["accel"].iloc[i],
            out_df["length_m"].iloc[i],
            out_df["status"].iloc[i] if "status" in out_df else 0
        )
        co2_list.append(co2)
    out_df["co2_g"] = co2_list

    # 单位里程排放强度
    out_df["co2_perkm"] = np.where(
        out_df["length_m"] > 2,
        out_df["co2_g"] / (out_df["length_m"] / 1000),
        0
    ).round(2)

    # 车流量标记：每条就是1辆车
    out_df["flow_cnt"] = 1

    # 重新组成 GeoDataFrame
    out_gdf = gpd.GeoDataFrame(out_df, geometry=geom, crs=gdf.crs)

    # 保留需要输出的字段
    keep_fields = [
        "r_id", "speed_kmh", "length_m", "accel", "co2_g", "co2_perkm", "flow_cnt",
        "status", "speed", "direction", "angle", "geometry"
    ]
    keep_fields = [f for f in keep_fields if f in out_gdf.columns]
    out_gdf = out_gdf[keep_fields]

    return out_gdf

# ===================== 主程序 =====================
def run_all():
    print("="*60)
    print("🚕 每小时出租车轨迹 + 碳排放 + 车速 + 车流量 → 输出 SHP")
    print("="*60)

    for folder in DATE_FOLDERS:
        date_str = folder.replace("宁波","").replace(".","-")
        folder_path = os.path.join(TRAJECTORY_BASE, folder)
        if not os.path.exists(folder_path):
            continue

        print(f"\n📅 处理日期：{date_str}")
        files = sorted([f for f in os.listdir(folder_path) if f.endswith(".shp")])

        for fname in files:
            hour = fname.split("_")[-2]
            print(f"   {hour}时", end=" → ")
            full_path = os.path.join(folder_path, fname)
            res = process_one_file(full_path)
            if res is None or len(res) == 0:
                print("无数据")
                continue

            # 输出 SHP
            out_shp = os.path.join(OUTPUT_FOLDER, f"出租车_{date_str}_{hour}时.shp")
            res.to_file(out_shp, encoding="utf-8")
            print(f"完成 {len(res)} 条轨迹")

    print("\n🎉 全部 SHP 输出完成！")
    print("输出文件夹：", OUTPUT_FOLDER)

if __name__ == "__main__":
    run_all()
