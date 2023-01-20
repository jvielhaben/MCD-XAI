import numpy as np

import concept_subspaces
import general_utils.data_utils


class SSCCD(concept_subspaces.ConceptSubspaces):
    """   
    Concept subspace discovery based on sparse subspace clustering of hidden activations in CNN or ViT.
    
   Parameters
    -----------        
    feature_maps: hidden activations from model under consideration, shape (batch_size, n_channels, *feature_map_spatial_shape) 
 
        """
    def __init__(self, feature_maps: np.ndarray) -> None:
        super().__init__(feature_maps)
        
        feature_map_shape = feature_maps.shape
        self.batch_size = feature_map_shape[0]
        self.n_channel = feature_map_shape[1]
        self.spatial_shape = feature_map_shape[2:]
    
    @staticmethod
    def map_to_array(maps: np.ndarray, n_channel:int) -> np.ndarray:
        array = maps.transpose(0,2,3,1)
        array = array.reshape(-1, n_channel)
        return array

    @staticmethod
    def array_to_map(array: np.ndarray, batch_size: int, n_channel: int,  spatial_shape) -> np.ndarray:
        array = array.transpose(1,0)
        array = array.reshape(n_channel, batch_size, *spatial_shape)
        array = array.transpose(1,0,2,3)
        return array


    def get_feature_array(self, sample_ratio: float):
        """"
        Get feature arrays from feature maps by unravelling in spatial dimensions and normalization to unit norm.

        Parameters
        -----------        
        sample_ratio: range [0,1]
            Randomly sample feature arrays by this ratio to reduce total number of feature arrays.
        """
        feature_array = self.map_to_array(self.feature_maps, self.n_channel)

        if sample_ratio<1.0:
            features = general_utils.data_utils.subsample_features(feature_array, sample_ratio)

        # normalization to unit l2 norm for SSC
        norm = (feature_array**2).sum(axis=1, keepdims=True)**0.5
        feature_array = feature_array / norm
        
        self.feature_array = feature_array
