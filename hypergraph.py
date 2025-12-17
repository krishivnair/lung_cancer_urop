"""
Hypergraph Construction
Multi-type hyperedge generation for lung nodule relationships
"""

from typing import List, Dict, Optional
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity


class HypergraphConstructor:
    """
    Construct hypergraph from lung CT scans with multi-order relationships.
    
    Attributes:
        k_neighbors: Number of nearest neighbors for spatial edges
        spatial_threshold: Maximum distance for spatial hyperedges (mm)
        feature_threshold: Minimum similarity for feature-based hyperedges
    """
    
    def __init__(
        self,
        k_neighbors: int = 8,
        spatial_threshold: float = 50.0,
        feature_threshold: float = 0.7
    ) -> None:
        """
        Initialize hypergraph constructor.
        
        Args:
            k_neighbors: Number of neighbors for k-NN
            spatial_threshold: Spatial proximity threshold in mm
            feature_threshold: Feature similarity threshold [0, 1]
        """
        self.k_neighbors = k_neighbors
        self.spatial_threshold = spatial_threshold
        self.feature_threshold = feature_threshold
    
    def construct_spatial_hyperedges(
        self,
        nodule_coords: np.ndarray,
        nodule_features: np.ndarray
    ) -> List[List[int]]:
        """
        Create hyperedges based on spatial proximity.
        
        Args:
            nodule_coords: Array of shape (N, 3) with (z, y, x) coordinates
            nodule_features: Array of shape (N, F) with node features
            
        Returns:
            List of hyperedges, each containing node indices
        """
        if len(nodule_coords) < 2:
            return []
        
        nbrs = NearestNeighbors(
            n_neighbors=min(self.k_neighbors, len(nodule_coords))
        )
        nbrs.fit(nodule_coords)
        distances, indices = nbrs.kneighbors(nodule_coords)
        
        hyperedges: List[List[int]] = []
        for i, (dist, idx) in enumerate(zip(distances, indices)):
            valid_neighbors = idx[dist < self.spatial_threshold]
            if len(valid_neighbors) >= 2:
                hyperedges.append(valid_neighbors.tolist())
        
        return hyperedges
    
    def construct_feature_hyperedges(
        self,
        nodule_features: np.ndarray,
        similarity_threshold: Optional[float] = None
    ) -> List[List[int]]:
        """
        Create hyperedges based on feature similarity.
        
        Args:
            nodule_features: Array of shape (N, F)
            similarity_threshold: Override default threshold
            
        Returns:
            List of hyperedges
        """
        threshold = similarity_threshold or self.feature_threshold
        similarity_matrix = cosine_similarity(nodule_features)
        
        hyperedges: List[List[int]] = []
        for i in range(len(nodule_features)):
            similar_nodes = np.where(similarity_matrix[i] > threshold)[0]
            if len(similar_nodes) >= 2:
                hyperedges.append(similar_nodes.tolist())
        
        return hyperedges
    
    def construct_anatomical_hyperedges(
        self,
        nodule_coords: np.ndarray,
        lung_mask: Optional[np.ndarray] = None
    ) -> List[List[int]]:
        """
        Create hyperedges based on anatomical regions.
        
        Args:
            nodule_coords: Array of shape (N, 3)
            lung_mask: Optional lung segmentation mask
            
        Returns:
            List of hyperedges grouped by anatomical region
        """
        z_coords = nodule_coords[:, 0]
        z_bins = np.percentile(z_coords, [33, 67])
        
        hyperedges: List[List[int]] = []
        regions = [
            (0, z_bins[0]),           # Lower lobe
            (z_bins[0], z_bins[1]),   # Middle lobe
            (z_bins[1], np.inf)       # Upper lobe
        ]
        
        for lower, upper in regions:
            nodes_in_region = np.where((z_coords >= lower) & (z_coords < upper))[0]
            if len(nodes_in_region) >= 2:
                hyperedges.append(nodes_in_region.tolist())
        
        return hyperedges
    
    def construct_hypergraph(
        self,
        nodule_coords: np.ndarray,
        nodule_features: np.ndarray,
        lung_mask: Optional[np.ndarray] = None
    ) -> Dict[str, List[List[int]]]:
        """
        Construct complete hypergraph with all edge types.
        
        Args:
            nodule_coords: Nodule coordinates (N, 3)
            nodule_features: Nodule features (N, F)
            lung_mask: Optional lung segmentation
            
        Returns:
            Dictionary with keys 'spatial', 'feature', 'anatomical'
        """
        spatial_edges = self.construct_spatial_hyperedges(
            nodule_coords, nodule_features
        )
        feature_edges = self.construct_feature_hyperedges(nodule_features)
        anatomical_edges = self.construct_anatomical_hyperedges(
            nodule_coords, lung_mask
        )
        
        return {
            'spatial': spatial_edges,
            'feature': feature_edges,
            'anatomical': anatomical_edges
        }
    
    def hyperedge_to_incidence_matrix(
        self,
        hyperedges: List[List[int]],
        num_nodes: int
    ) -> np.ndarray:
        """
        Convert hyperedges to incidence matrix.
        
        Args:
            hyperedges: List of hyperedges
            num_nodes: Total number of nodes
            
        Returns:
            Incidence matrix H of shape (num_nodes, num_hyperedges)
        """
        num_edges = len(hyperedges)
        H = np.zeros((num_nodes, num_edges), dtype=np.float32)
        
        for j, edge in enumerate(hyperedges):
            for i in edge:
                H[i, j] = 1.0
        
        return H