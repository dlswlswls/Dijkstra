# import geopandas as gpd
# import networkx as nx
# import matplotlib.pyplot as plt
# import numpy as np
# from entire_code import flooded_cells_with_H

# # Shapefile 불러오기
# dem_grid_path = r"C:\Users\USER\PYTHON\Dijkstra\DEM_GRID\DEM_GRID.shp"
# geo_path = r"C:\Users\USER\PYTHON\Dijkstra\geo.shp"

# gdf = gpd.read_file(dem_grid_path)
# geo_data = gpd.read_file(geo_path)


# # 도로인 셀 필터링
# road_nodes = gdf[gdf['Type'] == '도로']

# G = nx.Graph()

# for _, row in road_nodes.iterrows():
#     node_id = row['id']  # 'id' 컬럼에서 ID 가져오기
#     col_index = row['col_index']
#     row_index = row['row_index']
#     G.add_node(node_id, pos=(col_index, row_index))  # 좌표를 (col_index, row_index)로 설정

# # 간선 생성
# for _, row in road_nodes.iterrows():
#     current_node_id = row['id']
#     current_col = row['col_index']
#     current_row = row['row_index']
    
#     for _, neighbor_row in road_nodes.iterrows():
#         if neighbor_row['id'] != current_node_id:
#             neighbor_node_id = neighbor_row['id']
#             neighbor_col_ = neighbor_row['col_index']
#             neighbor_row_ = neighbor_row['row_index']
            
#             # 셀 간의 차이를 계산
#             col_diff = abs(current_col - neighbor_col_)
#             row_diff = abs(current_row - neighbor_row_)
            
#             weight=1 # 가중치 초기화

#             # 상하좌우인지 대각선인지 확인
#             if (col_diff == 1 and row_diff == 0) or (col_diff == 0 and row_diff == 1):
#                 weight = 1  # 상하좌우
#             elif col_diff == 1 and row_diff == 1:
#                 weight = 1  # 대각선
#             else: continue

#             # 수심에 따른 가중치
#             if(current_col, current_row) in flooded_cells_with_H:
#                  H_value = flooded_cells_with_H[(current_col, current_row)]
#                  if 0.4<=H_value<0.8:
#                     weight = 2 * weight
#                  elif 1.2<=H_value<1.6:
#                     weight = 3 * weight
#                  elif 1.6<=H_value<2:
#                     weight = 4 * weight
#                  elif 2<=H_value<2.4:
#                     weight = 5 * weight
#                  elif H_value >= 2.4:
#                     weight += 999 

#             # 간선 추가
#             G.add_edge(current_node_id, neighbor_node_id, weight=weight)

# # dijkstra 실행
# source = road_nodes[road_nodes['id'] == 372]['id'].values[0]
# target = road_nodes[road_nodes['id'] == 2323]['id'].values[0]
# path = nx.dijkstra_path(G, source, target)

# # print("최단경로: ", path)

# # ## 시각화
# # plt.figure(figsize=(10, 10))

# # # 노드 위치 설정
# # pos = nx.get_node_attributes(G, 'pos')

# # # 전체 그래프 그리기 (노드 및 간선)
# # nx.draw(G, pos, node_size=10, node_color='black', edge_color='gray', with_labels=False)

# # # 최종 경로 그리기 (빨간색)
# # nx.draw_networkx_edges(G, pos, edgelist=list(zip(path, path[1:])), edge_color='red', width=2)

# # # 침수 지점 표시 (파란색 포인트)
# # for (x, y), _ in flooded_cells_with_H.items():
# #     for node_id, (px, py) in pos.items():
# #          if abs(px - x) < 1e-2 and abs(py - y) < 1e-2:  # 비교 기준을 조정하여 좌표 매칭을 보장
# #             plt.scatter(px, py, color='blue', s=20)  # 침수 지점 표시
            


# # # 노드 위치 설정
# # pos = nx.get_node_attributes(G, 'pos')

# # # # 그래프의 edge만 그리기 (회색)
# # # nx.draw_networkx_edges(G, pos, edge_color='gray', width=0.5)

# # # 최종 경로 그리기 (빨간색)
# # nx.draw_networkx_edges(G, pos, edgelist=list(zip(path, path[1:])), edge_color='red', width=2)

# # # 침수 지점 표시 (파란색 포인트)
# # for (x, y), _ in flooded_cells_with_H.items():
# #     for node_id, (px, py) in pos.items():
# #         if abs(px - x) < 1e-2 and abs(py - y) < 1e-2:  # 비교 기준을 조정하여 좌표 매칭을 보장
# #             plt.scatter(px, py, color='blue', s=20)  # 침수 지점 표시

# # plt.gca().invert_yaxis()
# # plt.title('Terrain and Buildings Visualization')
# # plt.show()




# # plt.title("도로 네트워크, 침수 지점 및 최종 경로 시각화")
# # plt.xlabel("X Coordinate")
# # plt.ylabel("Y Coordinate")

# # plt.show()


# from shapely.affinity import scale
# import matplotlib.pyplot as plt
# import networkx as nx

# # geo_data의 경계 상자 가져오기
# minx, miny, maxx, maxy = geo_data.total_bounds
# #print(minx, miny, maxx, maxy)

# # 64x64 그리드 크기 설정
# grid_size = 64

# # 좌표 변환 함수 정의
# def transform_to_grid(geometry, minx, miny, maxx, maxy, grid_size):
#     xfact = grid_size / (maxx - minx)
#     yfact = grid_size / (maxy - miny)
#     return scale(geometry, xfact=xfact, yfact=yfact, origin=(minx, miny))

# # geo_data의 geometry 변환
# geo_data['geometry'] = geo_data['geometry'].apply(lambda geom: transform_to_grid(geom, minx, miny, maxx, maxy, grid_size))
# # print(geo_data.head())

# print("Terrain bounds:", geo_data.total_bounds)
# print("Raster bounds:", gdf.bounds)



# # 시각화 설정
# plt.figure(figsize=(10, 10))

# # 지형과 건물 시각화 (변환된 좌표계)
# terrain = geo_data[geo_data['type'] == '지형']
# buildings = geo_data[geo_data['type'] == '건물']

# terrain.plot(ax=plt.gca(), color='lightgray', edgecolor='none')
# buildings.plot(ax=plt.gca(), color='lightgray', edgecolor='darkgray')

# # 노드 위치 설정 (64x64 그리드)
# pos = nx.get_node_attributes(G, 'pos')

# # 그래프의 edge만 그리기
# nx.draw_networkx_edges(G, pos, edge_color='gray', width=0.5)

# # 침수 지점 표시 (64x64 그리드 좌표)
# for (x, y), _ in flooded_cells_with_H.items():
#     plt.scatter(x, y, color='blue', s=20)

# plt.title('Aligned Coordinates Visualization')
# plt.xlim(0, grid_size)
# plt.ylim(0, grid_size)
# plt.gca().invert_yaxis()  # y축 반전 (그리드의 좌표계가 위에서 아래로 내려가는 경우)

# plt.show()



import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from entire_code import flooded_cells_with_H

# Shapefile 불러오기
dem_grid_path = r"C:\Users\USER\PYTHON\Dijkstra\DEM_GRID\DEM_GRID.shp"
geo_path = r"C:\Users\USER\PYTHON\Dijkstra\geo\geo.shp"

gdf = gpd.read_file(dem_grid_path)
geo_data = gpd.read_file(geo_path)

# 도로인 셀 필터링
road_nodes = gdf[gdf['Type'] == '도로']

G = nx.Graph()

# 노드 추가 (좌표는 (col_index, row_index) 형태로 저장)
for _, row in road_nodes.iterrows():
    node_id = row['id']
    col_index = row['col_index']
    row_index = row['row_index']
    G.add_node(node_id, pos=(col_index, row_index))

# 간선 생성 및 가중치 계산
for _, row in road_nodes.iterrows():
    current_node_id = row['id']
    current_col = row['col_index']
    current_row = row['row_index']
    
    for _, neighbor_row in road_nodes.iterrows():
        if neighbor_row['id'] != current_node_id:
            neighbor_node_id = neighbor_row['id']
            neighbor_col_ = neighbor_row['col_index']
            neighbor_row_ = neighbor_row['row_index']
            
            col_diff = abs(current_col - neighbor_col_)
            row_diff = abs(current_row - neighbor_row_)
            
            weight = 1  # 기본 가중치 설정

            if (col_diff == 1 and row_diff == 0) or (col_diff == 0 and row_diff == 1):
                weight = 1  # 상하좌우
            elif col_diff == 1 and row_diff == 1:
                weight = 1  # 대각선
            else:
                continue

            if (current_col, current_row) in flooded_cells_with_H:
                H_value = flooded_cells_with_H[(current_col, current_row)]
                if 0.4 <= H_value < 0.8:
                    weight = 2 * weight
                elif 1.2 <= H_value < 1.6:
                    weight = 3 * weight
                elif 1.6 <= H_value < 2:
                    weight = 4 * weight
                elif 2 <= H_value < 2.4:
                    weight = 5 * weight
                elif H_value >= 2.4:
                    weight += 999

            G.add_edge(current_node_id, neighbor_node_id, weight=weight)

# 다익스트라 경로 탐색
source = road_nodes[road_nodes['id'] == 372]['id'].values[0]
target = road_nodes[road_nodes['id'] == 2323]['id'].values[0]
path = nx.dijkstra_path(G, source, target)

# 지도 좌표 변환 함수
affine_transform = (np.min(gdf.geometry.bounds.minx), np.min(gdf.geometry.bounds.miny),
                    (np.max(gdf.geometry.bounds.maxx) - np.min(gdf.geometry.bounds.minx)) / gdf['col_index'].max(),
                    (np.max(gdf.geometry.bounds.maxy) - np.min(gdf.geometry.bounds.miny)) / gdf['row_index'].max())

def grid_to_map(col, row, transform, y_center=None):
    x_origin, y_origin, pixel_width, pixel_height = transform
    x_map = x_origin + col * pixel_width
    y_map = y_origin + row * pixel_height

     # 중심축 기준으로 Y좌표 반전
    if y_center is not None:
        y_map = y_center - (y_map - y_center)

    return (x_map, y_map)

# 시각화 전에 중심축 계산
y_min, y_max = gdf.total_bounds[1], gdf.total_bounds[3]  # GeoDataFrame의 Y축 최소/최대 값
y_center = (y_min + y_max) / 2  # 중심축 계산

# 지형과 건물 
terrain = geo_data[geo_data['type'] == '지형']
buildings = geo_data[geo_data['type'] == '건물']

# 시각화
fig, ax = plt.subplots(figsize=(10, 10))
# geo_data.plot(ax=ax, color='lightgrey')
# gdf.plot(ax=ax, color='none', edgecolor='black')

terrain.plot(ax=plt.gca(), color='lightgray', edgecolor='darkgray')
buildings.plot(ax=plt.gca(), color='lightgray', edgecolor='darkgray')

# 변환된 경로 그리기
path_coords = [grid_to_map(*G.nodes[node]['pos'], affine_transform, y_center) for node in path]
x_coords, y_coords = zip(*path_coords)
plt.plot(x_coords, y_coords, color='red', linewidth=1, label='Dijkstra Path')
# ax.set_ylim(ax.get_ylim()[::-1])
# plt.gca().invert_yaxis()
plt.show()
