import streamlit as st

st.title("🎈 My new Streamlit app")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import heapq

def parse_edges(input_text):
    graph = {}
    lines = input_text.strip().split('\n')
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 3:
            continue
        u, v, w = parts
        w = float(w)
        if u not in graph:
            graph[u] = {}
        graph[u][v] = w
    return graph

def dijkstra(graph, start):
    distances = {node: float('inf') for node in graph}
    previous_nodes = {node: None for node in graph}
    distances[start] = 0
    queue = [(0, start)]

    while queue:
        current_dist, current_node = heapq.heappop(queue)

        if current_dist > distances[current_node]:
            continue

        for neighbor, weight in graph.get(current_node, {}).items():
            distance = current_dist + weight
            if distance < distances.get(neighbor, float('inf')):
                distances[neighbor] = distance
                previous_nodes[neighbor] = current_node
                heapq.heappush(queue, (distance, neighbor))

    return distances, previous_nodes

def reconstruct_path(previous_nodes, start, end):
    path = []
    current = end
    while current != start:
        if current is None:
            return None
        path.append(current)
        current = previous_nodes[current]
    path.append(start)
    path.reverse()
    return path

def draw_graph(graph, path=None):
    G = nx.DiGraph()
    for u in graph:
        for v, w in graph[u].items():
            G.add_edge(u, v, weight=w)

    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=1500, font_size=12)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    if path:
        edge_list = list(zip(path, path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=edge_list, edge_color='red', width=3)

    st.pyplot(plt)

def graph_theory_info(graph):
    st.subheader("그래프 이론 정보")
    nodes = set(graph.keys())
    for adj in graph.values():
        nodes.update(adj.keys())
    num_nodes = len(nodes)
    num_edges = sum(len(adj) for adj in graph.values())
    st.markdown(f"- 정점 수: {num_nodes}")
    st.markdown(f"- 간선 수: {num_edges}")

    st.markdown("- 각 정점의 차수:")
    for node in sorted(nodes):
        out_deg = len(graph.get(node, {}))
        in_deg = sum(1 for adj in graph.values() if node in adj)
        st.markdown(f"  - {node}: 진입 차수 {in_deg}, 진출 차수 {out_deg}")

st.title("🔗 다익스트라 최단 경로 & 그래프 이론 웹 앱")

st.markdown("""
**사용 방법:** 아래에 간선을 한 줄씩 입력하세요. 예: `A B 3` (A에서 B로 가는 가중치 3인 간선)
""")

edge_input = st.text_area("간선 입력", """A B 4
A C 2
B C 5
B D 10
C D 3""")

graph = parse_edges(edge_input)

graph_theory_info(graph)

if graph:
    all_nodes = sorted(set(graph.keys()) | {v for adj in graph.values() for v in adj})
    start = st.selectbox("출발 노드", all_nodes)
    end = st.selectbox("도착 노드", all_nodes)

    if st.button("최단 경로 찾기"):
        distances, previous_nodes = dijkstra(graph, start)
        path = reconstruct_path(previous_nodes, start, end)
        if path:
            st.success(f"'{start}'에서 '{end}'까지의 최단 거리: {distances[end]}")
            st.info(f"최단 경로: {' → '.join(path)}")
        else:
            st.error("경로를 찾을 수 없습니다.")
        draw_graph(graph, path)
else:
    st.warning("그래프를 먼저 입력해주세요.")
