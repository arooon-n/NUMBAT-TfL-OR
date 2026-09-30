from __future__ import annotations
import networkx as nx
import pandas as pd


def build_graph(frame: pd.DataFrame) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    for row in frame.dropna(subset=["from_station", "to_station"]).itertuples(index=False):
        graph.add_edge(str(row.from_station), str(row.to_station), line=getattr(row, "line", None), direction=getattr(row, "direction", None), time_period=getattr(row, "time_period", None), link_load=getattr(row, "link_load", None), link_frequency=getattr(row, "link_frequency", None))
    return graph


def statistics(graph: nx.MultiDiGraph) -> dict[str, object]:
    return {"stations": graph.number_of_nodes(), "links": graph.number_of_edges(), "connected_components": nx.number_weakly_connected_components(graph), "is_weakly_connected": nx.is_weakly_connected(graph) if graph else False, "branch_nodes": [n for n in graph if graph.out_degree(n) > 1 or graph.in_degree(n) > 1]}
