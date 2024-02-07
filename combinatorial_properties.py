
import numpy as np
import networkx as nx

def laplacian_after_vertex_removal(tree_edges, vertex_to_remove):
    modified_edges = [edge for edge in tree_edges if vertex_to_remove not in edge]
    G_modified = nx.Graph(modified_edges)
    L_modified = nx.laplacian_matrix(G_modified).toarray()
    return L_modified

def laplacian_after_edge_removal(tree_edges, edge_to_remove):
    modified_edges = [edge for edge in tree_edges if edge != edge_to_remove and edge[::-1] != edge_to_remove]
    G_modified = nx.Graph(modified_edges)
    components = list(nx.connected_components(G_modified))
    L_blocks = [nx.laplacian_matrix(G_modified.subgraph(component)).toarray() for component in components]
    L_modified = np.block([[L if i == j else np.zeros_like(L) for j, L in enumerate(L_blocks)] for i, _ in enumerate(L_blocks)])
    return L_modified

# Example Usage:
# L_vertex_removed = laplacian_after_vertex_removal([('a', 'b'), ('b', 'c'), ('b', 'd')], 'a')
# L_edge_removed = laplacian_after_edge_removal([('a', 'b'), ('b', 'c'), ('b', 'd')], ('a', 'b'))
