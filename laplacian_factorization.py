
import numpy as np
import networkx as nx

class TreeFactorization:
    def __init__(self, edges):
        self.graph = nx.Graph()
        self.graph.add_edges_from(edges)
        self.laplacian = None
        self.laplacian_perturbed = None
        self.lower_triangular = None

    def compute_laplacian(self):
        self.laplacian = nx.laplacian_matrix(self.graph).toarray()

    def perturb_laplacian(self, epsilon=1e-10):
        if self.laplacian is None:
            self.compute_laplacian()
        self.laplacian_perturbed = self.laplacian + epsilon * np.eye(self.laplacian.shape[0])

    def factorize_laplacian(self):
        if self.laplacian_perturbed is None:
            self.perturb_laplacian()
        self.lower_triangular = np.linalg.cholesky(self.laplacian_perturbed)

    def reconstruct_laplacian(self):
        if self.lower_triangular is None:
            self.factorize_laplacian()
        reconstructed = np.dot(self.lower_triangular, self.lower_triangular.T)
        return reconstructed

    def run(self):
        self.compute_laplacian()
        self.perturb_laplacian()
        self.factorize_laplacian()
        return self.reconstruct_laplacian()

# Example Usage:
# tree = TreeFactorization([('a', 'b'), ('b', 'c'), ('b', 'd')])
# reconstructed_laplacian = tree.run()
