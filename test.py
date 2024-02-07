import numpy as np
import networkx as nx

class DirectedTreeFactorization:
    def __init__(self, directed_edges):
        self.graph = nx.DiGraph()
        self.graph.add_edges_from(directed_edges)
        self.laplacian = None
        self.C = None
        self.node_to_idx = {node: idx for idx, node in enumerate(self.graph.nodes())}

    def compute_laplacian(self):
        n = len(self.graph.nodes())
        self.laplacian = np.zeros((n, n))
        for edge in self.graph.edges():
            i, j = self.node_to_idx[edge[0]], self.node_to_idx[edge[1]]
            self.laplacian[i, i] += 1
            self.laplacian[i, j] -= 1
            
    def construct_matrix_C(self):
        n = len(self.graph.nodes())
        self.C = np.zeros((n, n - 1))
        for col, edge in enumerate(self.graph.edges()):
            i, j = self.node_to_idx[edge[0]], self.node_to_idx[edge[1]]
            self.C[i, col] = 1
            self.C[j, col] = -1

    def factorize_laplacian(self):
        if self.laplacian is None:
            self.compute_laplacian()
        if self.C is None:
            self.construct_matrix_C()
        return np.allclose(self.laplacian, np.dot(self.C, self.C.T))

# Testing the class
directed_edges = [(0, 1), (0, 2), (2, 3)]

# Initialize the class
dtf = DirectedTreeFactorization(directed_edges)

# Factorize the Laplacian
factorization_holds = dtf.factorize_laplacian()

# Display results
print(f"Laplacian Matrix:\n{dtf.laplacian}")
print(f"\nMatrix C:\n{dtf.C}")
print(f"\nFactorization holds: {factorization_holds}")