import geopandas as gpd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 1. 数据加载 ====================
print("正在加载路网数据...")
file_path = r"D:\宁波数据\宁波市\宁波市道路路网_投影统一3857.shp"
gdf = gpd.read_file(file_path, encoding='GBK')

print(f"原始数据包含 {len(gdf)} 条道路段")
print(f"坐标系: {gdf.crs}")

# ==================== 2. 构建拓扑网络 ====================
def build_axial_network(gdf_input):
    """构建轴线网络"""
    print("正在构建轴线网络...")
    
    endpoint_to_edges = {}
    all_endpoints = []
    
    print("  提取线段端点...")
    for idx, row in tqdm(gdf_input.iterrows(), total=len(gdf_input), desc="处理线段"):
        geom = row.geometry
        if geom.geom_type == 'LineString':
            coords = list(geom.coords)
            start_key = (round(coords[0][0], 8), round(coords[0][1], 8))
            end_key = (round(coords[-1][0], 8), round(coords[-1][1], 8))
            
            if start_key not in endpoint_to_edges:
                endpoint_to_edges[start_key] = []
            if end_key not in endpoint_to_edges:
                endpoint_to_edges[end_key] = []
            
            endpoint_to_edges[start_key].append(idx)
            endpoint_to_edges[end_key].append(idx)
            all_endpoints.extend([start_key, end_key])
    
    unique_points = list(set(all_endpoints))
    print(f"  识别到 {len(unique_points)} 个唯一节点")
    
    point_to_id = {pt: i for i, pt in enumerate(unique_points)}
    
    print("  创建轴线网络图...")
    G = nx.Graph()
    
    for pt, idx in tqdm(point_to_id.items(), desc="添加节点"):
        G.add_node(idx, pos=(pt[0], pt[1]), x=pt[0], y=pt[1])
    
    edge_id = 0
    for idx, row in tqdm(gdf_input.iterrows(), total=len(gdf_input), desc="添加轴线"):
        geom = row.geometry
        if geom.geom_type == 'LineString':
            coords = list(geom.coords)
            start_key = (round(coords[0][0], 8), round(coords[0][1], 8))
            end_key = (round(coords[-1][0], 8), round(coords[-1][1], 8))
            
            if start_key in point_to_id and end_key in point_to_id:
                length = geom.length
                
                G.add_edge(point_to_id[start_key], point_to_id[end_key], 
                          length=length,
                          road_id=idx, 
                          edge_id=edge_id,
                          geom=geom)
                edge_id += 1
    
    print(f"  网络包含 {G.number_of_nodes():,} 个节点和 {G.number_of_edges():,} 条轴线")
    return G, point_to_id

# ==================== 3. DepthmapX风格指标计算 ====================

def calculate_connectivity(G):
    """连接值：直接连接的轴线数量"""
    print("计算连接值...")
    connectivity = dict(G.degree())
    return connectivity

def calculate_integration(G):
    """整合度：基于拓扑距离的RA公式"""
    print("计算整合度...")
    integration = {}
    nodes = list(G.nodes())
    n_nodes = len(nodes)
    
    print(f"  节点总数: {n_nodes:,}")
    
    batch_size = 20
    total_batches = (n_nodes + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(total_batches), desc="计算整合度"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_nodes)
        batch_nodes = nodes[start_idx:end_idx]
        
        for node in batch_nodes:
            try:
                lengths = nx.single_source_shortest_path_length(G, node)
                
                if len(lengths) <= 1:
                    integration[node] = 0.0
                    continue
                
                total_depth = sum(lengths.values())
                md = total_depth / (len(lengths) - 1)
                
                if n_nodes > 2 and md > 1:
                    ra = 2 * (md - 1) / (n_nodes - 2)
                    integration[node] = 1.0 / ra if ra > 0 else 0.0
                else:
                    integration[node] = 0.0
            except Exception:
                integration[node] = 0.0
    
    return integration

def calculate_choice(G, radius=3):
    """选择度：基于最短拓扑路径的介数"""
    print(f"计算拓扑选择度...")
    choice = {node: 0 for node in G.nodes()}
    nodes = list(G.nodes())
    n_nodes = len(nodes)
    
    print(f"  节点总数: {n_nodes:,}")
    
    batch_size = 10
    total_batches = (n_nodes + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(total_batches), desc="计算选择度"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_nodes)
        batch_sources = nodes[start_idx:end_idx]
        
        for source in batch_sources:
            try:
                paths = nx.single_source_shortest_path(G, source)
                for target in nodes:
                    if target == source:
                        continue
                    if target in paths:
                        path = paths[target]
                        if len(path) - 1 <= radius:
                            for node in path[1:-1]:
                                if node in choice:
                                    choice[node] += 1
            except Exception:
                continue
    
    return choice

def calculate_metric_choice(G, radius=500):
    """度量选择度：基于几何距离"""
    print(f"计算度量选择度...")
    metric_choice = {node: 0 for node in G.nodes()}
    
    pos = {node: (G.nodes[node]['x'], G.nodes[node]['y']) for node in G.nodes()}
    nodes = list(G.nodes())
    n_nodes = len(nodes)
    
    print(f"  节点总数: {n_nodes:,}")
    
    batch_size = 10
    total_batches = (n_nodes + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(total_batches), desc="计算度量选择度"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_nodes)
        batch_sources = nodes[start_idx:end_idx]
        
        for source in batch_sources:
            pos_source = pos[source]
            for target in nodes:
                if target == source:
                    continue
                
                dist = np.sqrt((pos_source[0] - pos[target][0])**2 + 
                              (pos_source[1] - pos[target][1])**2)
                if dist > radius:
                    continue
                
                try:
                    path = nx.dijkstra_path(G, source, target, weight='length')
                    for node in path[1:-1]:
                        if node in metric_choice:
                            weight = 1.0 / (len(path) + 1)
                            metric_choice[node] += weight
                except Exception:
                    continue
    
    return metric_choice

def calculate_angular_connectivity(G):
    """角度连接值：基于角度变化"""
    print("计算角度连接值...")
    angular_connectivity = {}
    
    for node in tqdm(G.nodes(), desc="计算角度连接值"):
        neighbors = list(G.neighbors(node))
        n_neighbors = len(neighbors)
        
        if n_neighbors == 0:
            angular_connectivity[node] = 0.0
            continue
        
        pos_node = (G.nodes[node]['x'], G.nodes[node]['y'])
        
        angles = []
        for neighbor in neighbors:
            pos_neighbor = (G.nodes[neighbor]['x'], G.nodes[neighbor]['y'])
            angle = np.arctan2(pos_neighbor[1] - pos_node[1], 
                             pos_neighbor[0] - pos_node[0])
            angles.append(angle)
        
        if n_neighbors == 1:
            angular_connectivity[node] = 1.0
        else:
            angles_sorted = sorted(angles)
            angle_diffs = []
            for i in range(len(angles_sorted)):
                diff = angles_sorted[(i+1) % len(angles_sorted)] - angles_sorted[i]
                if i == len(angles_sorted) - 1:
                    diff = diff + 2 * np.pi
                angle_diffs.append(abs(diff))
            
            mean_diff = 2 * np.pi / n_neighbors
            if mean_diff > 0:
                cv = np.std(angle_diffs) / mean_diff
                angular_connectivity[node] = 1.0 / (1.0 + cv)
            else:
                angular_connectivity[node] = 1.0
    
    return angular_connectivity

def calculate_mean_depth(G):
    """平均深度：到所有节点的平均拓扑步数"""
    print("计算平均深度...")
    mean_depth = {}
    nodes = list(G.nodes())
    n_nodes = len(nodes)
    
    print(f"  节点总数: {n_nodes:,}")
    
    batch_size = 20
    total_batches = (n_nodes + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(total_batches), desc="计算平均深度"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_nodes)
        batch_nodes = nodes[start_idx:end_idx]
        
        for node in batch_nodes:
            try:
                lengths = nx.single_source_shortest_path_length(G, node)
                if len(lengths) > 1:
                    total_depth = sum(lengths.values())
                    mean_depth[node] = total_depth / (len(lengths) - 1)
                else:
                    mean_depth[node] = 0.0
            except Exception:
                mean_depth[node] = 0.0
    
    return mean_depth

# ==================== 4. 映射函数 ====================

def map_indicators_to_edges(G, gdf_input, indicators):
    """将节点指标映射到线段（取两端点的平均值）"""
    print("  映射指标到线段...")
    
    edge_indicators = {}
    
    for u, v, data in G.edges(data=True):
        road_id = data.get('road_id', -1)
        if road_id == -1:
            continue
        
        values = []
        for node in [u, v]:
            if node in indicators:
                values.append(indicators[node])
        
        if values:
            edge_indicators[road_id] = np.mean(values)
        else:
            edge_indicators[road_id] = 0.0
    
    return edge_indicators

# ==================== 5. 主程序 ====================
def main():
    # 构建网络
    G, point_to_id = build_axial_network(gdf)
    
    if G.number_of_nodes() == 0:
        print("错误：网络中没有节点")
        return
    
    # 计算各项指标
    print("\n" + "="*60)
    print("开始计算空间句法指标...")
    print("="*60)
    
    # 1. 连接值
    connectivity_raw = calculate_connectivity(G)
    connectivity_edges = map_indicators_to_edges(G, gdf, connectivity_raw)
    
    # 2. T1024整合度
    integration_raw = calculate_integration(G)
    integration_edges = map_indicators_to_edges(G, gdf, integration_raw)
    
    # 3. 拓扑选择度
    choice_raw = calculate_choice(G, radius=3)
    choice_edges = map_indicators_to_edges(G, gdf, choice_raw)
    
    # 4. 度量选择度
    metric_choice_raw = calculate_metric_choice(G, radius=500)
    metric_choice_edges = map_indicators_to_edges(G, gdf, metric_choice_raw)
    
    # 5. 角度连接值
    angular_raw = calculate_angular_connectivity(G)
    angular_edges = map_indicators_to_edges(G, gdf, angular_raw)
    
    # 6. 平均深度
    mean_depth_raw = calculate_mean_depth(G)
    mean_depth_edges = map_indicators_to_edges(G, gdf, mean_depth_raw)
    
    # 将指标添加到原始GeoDataFrame
    print("\n正在添加指标到数据...")
    result_gdf = gdf.copy()
    
    result_gdf['Connectivity'] = 0.0
    result_gdf['Integration'] = 0.0
    result_gdf['Choice'] = 0.0
    result_gdf['MetricChoice'] = 0.0
    result_gdf['AngularConn'] = 0.0
    result_gdf['MeanDepth'] = 0.0
    
    for road_id, value in connectivity_edges.items():
        if road_id < len(result_gdf):
            result_gdf.at[road_id, 'Connectivity'] = value
    
    for road_id, value in integration_edges.items():
        if road_id < len(result_gdf):
            result_gdf.at[road_id, 'Integration'] = value
    
    for road_id, value in choice_edges.items():
        if road_id < len(result_gdf):
            result_gdf.at[road_id, 'Choice'] = value
    
    for road_id, value in metric_choice_edges.items():
        if road_id < len(result_gdf):
            result_gdf.at[road_id, 'MetricChoice'] = value
    
    for road_id, value in angular_edges.items():
        if road_id < len(result_gdf):
            result_gdf.at[road_id, 'AngularConn'] = value
    
    for road_id, value in mean_depth_edges.items():
        if road_id < len(result_gdf):
            result_gdf.at[road_id, 'MeanDepth'] = value
    
    # 保存结果
    output_path = r"D:\宁波数据\宁波市\宁波市道路路网_空间句法指标_原始值.shp"
    result_gdf.to_file(output_path, encoding='GBK')
    print(f"结果已保存至: {output_path}")
    
    # ==================== 6. 统一双色渐变可视化 ====================
    print("\n生成双色渐变可视化地图...")
    
    fig = plt.figure(figsize=(32, 24))
    gs = fig.add_gridspec(2, 3, hspace=0.08, wspace=0.08)
    
    # 双色渐变方案：从浅色到深色 (统一风格)
    # 每个指标使用不同的色系但结构相同
    color_schemes = {
        'Connectivity': ['#f7fcf5', '#41ab5d', '#00441b'],      # 绿色系
        'Integration': ['#f7fbff', '#6baed6', '#08306b'],       # 蓝色系
        'Choice': ['#fff7f3', '#fc9272', '#67000d'],            # 红色系
        'MetricChoice': ['#fff5eb', '#fd8d3c', '#7f2704'],      # 橙色系
        'AngularConn': ['#fcfbfd', '#756bb1', '#3f007d'],       # 紫色系
        'MeanDepth': ['#f7fcfd', '#2ca25f', '#00441b']          # 蓝绿色系
    }
    
    # 指标配置
    indicators = [
        ('Connectivity', '连接值 (原始度)', 0, 0),
        ('Integration', 'T1024整合度 (原始值)', 0, 1),
        ('Choice', '拓扑选择度 (原始介数)', 0, 2),
        ('MetricChoice', '度量选择度 (原始值)', 1, 0),
        ('AngularConn', '角度连接值 (原始值)', 1, 1),
        ('MeanDepth', '平均深度 (原始步数)', 1, 2)
    ]
    
    for col, title, row, col_idx in indicators:
        ax = fig.add_subplot(gs[row, col_idx])
        
        # 创建双色渐变映射
        colors = color_schemes[col]
        cmap = LinearSegmentedColormap.from_list(f'{col}_cmap', colors, N=256)
        
        # 获取数值
        values = result_gdf[col].values
        valid_values = values[values > 0]
        
        if len(valid_values) > 10:
            # 根据数据分布特点调整百分位数范围
            if col == 'Integration':
                # 整合度数值范围大，使用更宽的百分位数范围
                p_low, p_high = 2, 98
            elif col == 'Connectivity' or col == 'AngularConn':
                # 连接值和角度连接值分布集中，使用更宽范围
                p_low, p_high = 1, 99
            else:
                p_low, p_high = 5, 95
            
            p5 = np.percentile(valid_values, p_low)
            p95 = np.percentile(valid_values, p_high)
            
            # 确保范围有效
            if p5 == p95:
                p5 = valid_values.min()
                p95 = valid_values.max()
            
            # 绘制
            result_gdf.plot(column=col, ax=ax, cmap=cmap,
                           linewidth=0.6,
                           vmin=p5,
                           vmax=p95,
                           legend=True,
                           legend_kwds={
                               'label': '',
                               'orientation': 'horizontal',
                               'shrink': 0.3,
                               'pad': 0.02,
                               'aspect': 20
                           })
        else:
            result_gdf.plot(ax=ax, linewidth=0.3, color='#cccccc')
        
        # 设置标题
        ax.set_title(title, fontsize=18, fontweight='bold', pad=10)
        ax.set_axis_off()
        
        # 设置地图范围
        bounds = result_gdf.total_bounds
        x_range = bounds[2] - bounds[0]
        y_range = bounds[3] - bounds[1]
        margin = 0.005
        
        ax.set_xlim(bounds[0] - x_range * margin, bounds[2] + x_range * margin)
        ax.set_ylim(bounds[1] - y_range * margin, bounds[3] + y_range * margin)
        
        # 在右上角显示颜色映射范围
        if len(valid_values) > 10:
            ax.text(0.98, 0.98, f'[{p5:.2f}, {p95:.2f}]', 
                    transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # 添加总标题
    plt.suptitle('宁波市路网空间句法指标分析\n(双色渐变映射 · 原始值)', 
                fontsize=26, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # 保存图片
    output_img = r"D:\宁波数据\宁波市\空间句法指标_双色渐变.png"
    plt.savefig(output_img, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"可视化图片已保存至: {output_img}")
    
    plt.show()
    
    print("\n" + "="*60)
    print("所有计算完成！")
    print("="*60)
    
    return result_gdf

if __name__ == "__main__":
    result = main()