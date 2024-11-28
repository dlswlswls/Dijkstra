import geopandas as gpd
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from entire_code_ import flooded_cells_with_H
from collections import defaultdict
import heapq

nodes_path = r"C:\Users\USER\PYTHON\Dijkstra\node_edge\node.shp"
edges_path = r"C:\Users\USER\PYTHON\Dijkstra\node_edge\edges.shp"
dem_grid_path = r"C:\Users\USER\PYTHON\Dijkstra\DEM_GRID\DEM_GRID.shp"
geo_path = r"C:\Users\USER\PYTHON\Dijkstra\geo\geo.shp"

nodes = gpd.read_file(nodes_path)
edges = gpd.read_file(edges_path)
dem_grid = gpd.read_file(dem_grid_path)
geo_data = gpd.read_file(geo_path)

# dem_grid data -> cell_id: {flood_depth, edge_id} 구축
# 동일한 cell_id를 기준으로 edge_id 병합
filtered_dem_grid = (
    dem_grid.groupby("cell_id")
    .agg(
        {
            "edge_id": lambda x: [edge for edge in x if pd.notna(edge)],  # edge_id 병합, NaN 값 제거
            "col_index": "first",
            "row_index": "first",
        }
    )
    .reset_index() # 병합된 cell_id와 인덱스 일치
)

filtered_dem_grid_dict = {
    row["cell_id"]: {
        "col": int(row["col_index"]),
        "row": int(row["row_index"]),
        "edges_id": row["edge_id"]}
    for _, row in filtered_dem_grid.iterrows()
}

# 침수심과 매핑
flooded_filtered_dem_grid = {}
for cell_id, dem_info in filtered_dem_grid_dict.items():
    col = dem_info["col"]
    row = dem_info["row"]
    edges_id = dem_info["edges_id"]

    for (cx, cy), flood_depth in flooded_cells_with_H.items():
        if (col == cx) and (row == cy):
            flooded_filtered_dem_grid[cell_id] = {
                "flood_depth": float(flood_depth),
                "edges_id": [edges_id] if isinstance(edges_id, str) else edges_id,
                "col": col, "row": row
            }

# GeoDataFrame을 리스트로 변환
edges_list = edges.to_dict(orient="records")

# 다른 셀을 가진 동일한 간선끼리 결합
merged_edges = defaultdict(lambda: {"node_1": None, "node_2": None, "length": None, "cells": []})
for edge in edges_list:
    edge_id = edge["edge_id"]
    if merged_edges[edge_id]["node_1"] is None: # node_1, node_2, length는 첫 번째 값으로 고정
        merged_edges[edge_id]["node_1"] = edge["node_1"]
        merged_edges[edge_id]["node_2"] = edge["node_2"]
        merged_edges[edge_id]["length"] = edge["length"]
    merged_edges[edge_id]["cells"].append(edge["cell_id"])

merged_edges_dict = dict(merged_edges)

def calculate_weight(flood_depth):
    if flood_depth is None:
        return 1
    elif 0.0 < flood_depth < 0.2:
        return 2
    elif 0.2 <= flood_depth < 0.3:
        return 3
    elif 0.3 <= flood_depth < 0.5:
        return 4
    elif flood_depth >= 0.5:
        return 100
    else: return 1

def calculate_edge_weight(edge_cells, dem_grid):
    total_weight = 0
    flood_depth_list = []
    # 간선과 dem_grid의 셀이 일치하는 부분의 침수심 리스트
    for cell_id in edge_cells:
        flood_depth = dem_grid[cell_id]['flood_depth']
        flood_depth_list.append(flood_depth)
    
    if flood_depth_list:
        depth_max = max(flood_depth_list)
    else:
        depth_max = 0
    
    weight = calculate_weight(depth_max)
    if weight is not None:
        total_weight += weight

    return total_weight

def build_graph(edges, dem_grid):
    graph = {}
    for edge_id, edge_data  in edges.items():
        start, end = edge_data['node_1'], edge_data['node_2']
        length = edge_data['length']
        edge_cells = edge_data['cells']
        weight = length
        weight = calculate_edge_weight(edge_cells, dem_grid) * length

        if start not in graph:
            graph[start] = []
        if end not in graph:
            graph[end] = []

        graph[start].append((end, weight))
        graph[end].append((start, weight)) # 양방향

    return graph

def dijkstra(graph, source, target):
    pq = [] # 우선순위 큐
    heapq.heappush(pq, (0, source)) # (거리, 노드) 저장
    distances = {node: float('inf') for node in graph} # 초기 모든 노드 거리 무한대
    distances[source] = 0 
    prev_nodes = {node: None for node in graph} # 이전 노드 추적

    while pq:
        current_distance, current_node = heapq.heappop(pq) # 가장 짧은 거리 노드 꺼냄

        if current_node == target:
            break

        for neighbor, weight in graph[current_node]:
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance # 거리 업데이트
                prev_nodes[neighbor] = current_node # (떠나온) 이전 노드 업데이트
                heapq.heappush(pq, (distance, neighbor))

        # 경로 추적
        path = []
        current = target
        while current is not None:
            path.append(current)
            current = prev_nodes[current]
        path.reverse()
    return path, distances[target]

# 그래프 생성
graph = build_graph(merged_edges_dict, flooded_filtered_dem_grid)

# 임의 설정
source_node = '32'
target_node = '141'

shortest_path, final_weight = dijkstra(graph, source_node, target_node)

print("최단 경로: ", shortest_path)
print("최종 가중치: ", final_weight)

# # 시각화
# 출발노드, 도착노드 GeoDataFrame로 변환
path_gdf = nodes[nodes['node_id'].isin(shortest_path)] # 경로 GeoDataFrame
start_node = shortest_path[0]
end_node = shortest_path[-1]
start_end_nodes = path_gdf[path_gdf['node_id'].isin([start_node, end_node])]

# 최종 경로 라인
final_edges_gdf = edges[
    (edges['node_1'].isin(shortest_path)) & (edges['node_2'].isin(shortest_path))
]

# 지형, 건물 
terrain = geo_data[geo_data['type'] == '지형']
buildings = geo_data[geo_data['type'] == '건물']
# 침수 지도 좌표계 확인

fig, ax = plt.subplots(figsize=(10, 10))

terrain.plot(ax=plt.gca(), color='lightgray', edgecolor='darkgray')
buildings.plot(ax=plt.gca(), color='lightgray', edgecolor='darkgray')

final_edges_gdf.plot(ax=ax, color='red', linewidth=2, label="Shortest Path Edges")
# edges.plot(ax=ax, color='gray', linewidth=0.5)

start_end_nodes.plot(ax=ax, color='red', markersize=30, label='Shortest Path')

# nodes.plot(ax=ax, color='black', markersize=5)

plt.legend()
plt.show()
