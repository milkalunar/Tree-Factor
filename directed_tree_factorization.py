
import numpy as np
import networkx as nx

class DirectedTreeFactorization:
    def __init__(self, directed_edges):
        self.graph = nx.DiGraph()
        self.graph.add_edges_from(directed_edges)
        self.laplacian = None
        self.C = None

    def compute_laplacian(self):
        n = len(self.graph.nodes())
        self.laplacian = np.zeros((n, n))
        for edge in self.graph.edges():
            i, j = edge
            self.laplacian[i, i] += 1
            self.laplacian[i, j] -= 1

    def construct_matrix_C(self):
        n = len(self.graph.nodes())
        self.C = np.zeros((n, n - 1))
        for col, edge in enumerate(self.graph.edges()):
            i, j = edge
            self.C[i, col] = 1
            self.C[j, col] = -1

    def factorize_laplacian(self):
        if self.laplacian is None:
            self.compute_laplacian()
        if self.C is None:
            self.construct_matrix_C()
        return np.allclose(self.laplacian, np.dot(self.C, self.C.T))

    def run(self):
        self.compute_laplacian()
        self.construct_matrix_C()
        return self.factorize_laplacian()

# Example Usage:
# tree = DirectedTreeFactorization([('a', 'b'), ('b', 'c'), ('b', 'd')])
# result = tree.run()
