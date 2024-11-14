import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from entire_code import flooded_cells_with_H

# Shapefile 불러오기
shapefile_path = r"C:\Users\USER\Desktop\DEM_GRID\DEM_GRID.shp"
gdf = gpd.read_file(shapefile_path)

# 도로인 셀 필터링
road_nodes = gdf[gdf['Type'] == '도로']

G = nx.Graph()

for _, row in road_nodes.iterrows():
    node_id = row['id']  # 'id' 컬럼에서 ID 가져오기
    col_index = row['col_index']
    row_index = row['row_index']
    G.add_node(node_id, pos=(col_index, row_index))  # 좌표를 (col_index, row_index)로 설정

# 간선 생성
for _, row in road_nodes.iterrows():
    current_node_id = row['id']
    current_col = row['col_index']
    current_row = row['row_index']
    
    for _, neighbor_row in road_nodes.iterrows():
        if neighbor_row['id'] != current_node_id:
            neighbor_node_id = neighbor_row['id']
            neighbor_col_ = neighbor_row['col_index']
            neighbor_row_ = neighbor_row['row_index']
            
            # 셀 간의 차이를 계산
            col_diff = abs(current_col - neighbor_col_)
            row_diff = abs(current_row - neighbor_row_)
            
            weight=1 # 가중치 초기화

            # 상하좌우인지 대각선인지 확인
            if (col_diff == 1 and row_diff == 0) or (col_diff == 0 and row_diff == 1):
                weight = 1  # 상하좌우
            elif col_diff == 1 and row_diff == 1:
                weight = 1  # 대각선
            else: continue

            # 수심에 따른 가중치
            if(current_col, current_row) in flooded_cells_with_H:
                 H_value = flooded_cells_with_H[(current_col, current_row)]
                 if 0.4<=H_value<0.8:
                    weight = 2 * weight
                 elif 1.2<=H_value<1.6:
                    weight = 3 * weight
                 elif 1.6<=H_value<2:
                    weight = 4 * weight
                 elif 2<=H_value<2.4:
                    weight = 5 * weight
                 elif H_value >= 2.4:
                    weight += 999 

            # 간선 추가
            G.add_edge(current_node_id, neighbor_node_id, weight=weight)

# dijkstra 실행
source = road_nodes[road_nodes['id'] == 372]['id'].values[0]
target = road_nodes[road_nodes['id'] == 2323]['id'].values[0]
path = nx.dijkstra_path(G, source, target)

print("최단경로: ", path)

## 시각화
plt.figure(figsize=(10, 10))

# 노드 위치 설정
pos = nx.get_node_attributes(G, 'pos')

# 전체 그래프 그리기 (노드 및 간선)
nx.draw(G, pos, node_size=10, node_color='black', edge_color='gray', with_labels=False)

# 최종 경로 그리기 (빨간색)
nx.draw_networkx_edges(G, pos, edgelist=list(zip(path, path[1:])), edge_color='red', width=2)

# 침수 지점 표시 (파란색 포인트)
for (x, y), _ in flooded_cells_with_H.items():
    for node_id, (px, py) in pos.items():
         if abs(px - x) < 1e-2 and abs(py - y) < 1e-2:  # 비교 기준을 조정하여 좌표 매칭을 보장
            plt.scatter(px, py, color='blue', s=20)  # 침수 지점 표시
            
plt.gca().invert_yaxis()
plt.title("도로 네트워크, 침수 지점 및 최종 경로 시각화")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")

plt.show()