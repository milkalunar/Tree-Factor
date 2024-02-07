import numpy as np
import networkx as nx

def laplacian_after_vertex_removal(tree_edges, vertex_to_remove):
    """
    Compute the Laplacian matrix of a tree after removing a specified vertex.

    Parameters:
    - tree_edges (list of tuple): List of edges in the tree represented as tuples (u, v).
    - vertex_to_remove (any hashable type): The vertex to be removed from the tree.

    Returns:
    - numpy.ndarray: The Laplacian matrix of the modified tree.
    """
    
    # Filter out edges that include the vertex to be removed
    modified_edges = [edge for edge in tree_edges if vertex_to_remove not in edge]
    
    # Create a graph from the modified edges
    G_modified = nx.Graph(modified_edges)
    
    # Compute the Laplacian matrix of the modified graph
    L_modified = nx.laplacian_matrix(G_modified).toarray()
    
    return L_modified



import numpy as np
import networkx as nx

def laplacian_after_edge_removal(tree_edges, edge_to_remove):
    """
    Compute the block diagonal Laplacian matrix after removing a specified edge from a tree.

    Parameters:
    - tree_edges (list of tuple): List of edges in the tree represented as tuples (u, v).
    - edge_to_remove (tuple): The edge to be removed from the tree.

    Returns:
    - numpy.ndarray: The block diagonal Laplacian matrix of the disjoint subgraphs after edge removal.
    """
    
    # Filter out the specified edge and its reverse
    modified_edges = [edge for edge in tree_edges if edge != edge_to_remove and edge[::-1] != edge_to_remove]
    
    # Create a graph from the modified edges
    G_modified = nx.Graph(modified_edges)
    
    # Identify the connected components
    components = list(nx.connected_components(G_modified))
    
    # Compute the Laplacian matrix for each component
    L_blocks = [nx.laplacian_matrix(G_modified.subgraph(component)).toarray() for component in components]
    
    # Construct a block diagonal matrix from the Laplacian matrices
    L_modified = np.block([[L if i == j else np.zeros_like(L) for j, L in enumerate(L_blocks)] for i, _ in enumerate(L_blocks)])
    
    return L_modified



import numpy as np
import networkx as nx

class DirectedTreeFactorization:
    def __init__(self, directed_edges):
        self.graph = nx.DiGraph()
        self.graph.add_edges_from(directed_edges)
        self.laplacian = None
        self.C = None
        self.node_to_idx = {node: idx for idx, node in enumerate(self.graph.nodes())}  # Mapping nodes to indices

    def compute_laplacian(self):
        n = len(self.graph.nodes())
        self.laplacian = np.zeros((n, n))
        for edge in self.graph.edges():
            i, j = self.node_to_idx[edge[0]], self.node_to_idx[edge[1]]
            self.laplacian[i, i] += 1
            self.laplacian[i, j] -= 1



import numpy as np
import networkx as nx

class DirectedTreeFactorization:
    def __init__(self, directed_edges):
        """
        Initialize the DirectedTreeFactorization class with the provided directed edges.

        Parameters:
        - directed_edges (list of tuple): List of directed edges in the format (source, target).
        """
        
        # Initialize an empty directed graph
        self.graph = nx.DiGraph()
        
        # Add edges to the graph from the provided directed edges
        self.graph.add_edges_from(directed_edges)
        
        # Initialize Laplacian matrix and C matrix to None
        self.laplacian = None
        self.C = None
        
        # Create a mapping from node labels to indices for easy referencing in matrices
        self.node_to_idx = {node: idx for idx, node in enumerate(self.graph.nodes())}

    def compute_laplacian(self):
        """
        Compute the Laplacian matrix for the directed graph.
        This method calculates a variant of the "out-degree Laplacian" for directed graphs.
        """
        
        # Number of nodes in the graph
        n = len(self.graph.nodes())
        
        # Initialize the Laplacian matrix as an n x n zero matrix
        self.laplacian = np.zeros((n, n))
        
        # Iterate over the directed edges in the graph
        for edge in self.graph.edges():
            # Map the nodes of the edge to their corresponding indices
            i, j = self.node_to_idx[edge[0]], self.node_to_idx[edge[1]]
            
            # Update the Laplacian matrix based on the directed edge
            self.laplacian[i, i] += 1
            self.laplacian[i, j] -= 1


def construct_matrix_C(self):
    """
    Construct the auxiliary matrix C based on the directed edges of the graph.
    For each directed edge (i, j), the corresponding column in C will have a 
    value of 1 at row i and a value of -1 at row j.
    """
    
    # Get the number of nodes in the graph
    n = len(self.graph.nodes())
    
    # Initialize matrix C of shape n x (n-1)
    self.C = np.zeros((n, n - 1))
    
    # Populate matrix C based on the directed edges
    for col, edge in enumerate(self.graph.edges()):
        i, j = self.node_to_idx[edge[0]], self.node_to_idx[edge[1]]  # Use node-to-index mapping
        self.C[i, col] = 1
        self.C[j, col] = -1

def factorize_laplacian(self):
    """
    Check if the Laplacian matrix can be factorized as L = C x C^T.
    
    Returns:
    - bool: True if the factorization holds, False otherwise.
    """
    
    # Compute the Laplacian matrix if it hasn't been computed
    if self.laplacian is None:
        self.compute_laplacian()
        
    # Construct matrix C if it hasn't been constructed
    if self.C is None:
        self.construct_matrix_C()
    
    # Check if the Laplacian matrix is close to the product of C and its transpose
    return np.allclose(self.laplacian, np.dot(self.C, self.C.T))

