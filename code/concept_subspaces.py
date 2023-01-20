import numpy as np
from scipy import sparse
from scipy.linalg import subspace_angles, norm, orth, solve

from tqdm import tqdm

import  skdim.id
import sklearn.decomposition
import sklearn.cluster

import subspaceclustering.elasticnet_selfrepresentation

def principle_angles_subspace_subspace(A,B):
    """
    Use scipy implementation for subspace angles here which computes SVD for cosine and sine.
    This is good for precision but slow.

    Parameters
    -----------   
    A, B: np.ndarray, matrix with basis vectors for subspace of dimensionality p in R^d of shape (p, n)
    """
    if(len(A.shape)==1):
        A= np.expand_dims(A,axis=0)
    if(len(B.shape)==1):
        B = np.expand_dims(B,axis=0)
    try:
        pas = np.array(subspace_angles(A.T,B.T))
    except:
        pas = None
    return pas


def grassmann_distance_from_principal_angles(pas):
    return np.sqrt(np.sum(np.power(pas,2)))


def subspac_distance_matrix(subspaces):
    distance_matrix = np.empty((len(subspaces), len(subspaces)))
    for i,A in enumerate(subspaces):
        for j,B in enumerate(subspaces):
            pas = principle_angles_subspace_subspace(A,B)
            distance_matrix[i,j] = grassmann_distance_from_principal_angles(pas)
    return distance_matrix


def principle_angle_vector_subspace(vectors: np.ndarray, subspace_basis: np.ndarray):
    """
    Efficient calculation of PA between a single vector and a basis based on projection matrix.

    Parameters
    -----------        
    vectors: array of shape (N, n) where N is the number of samples and n is the dimensionality of the (feature) space
    subspace_basis: array of shape (p, n) where p is the dimensionality of the subspace, orthonormal basis
    """
    # get projections w of all vectors onto subspace basis
    # 1. coefficients
    C_Np = np.einsum("Nn,pn->Np", vectors, subspace_basis)
    # 2. projections
    W_Nn = np.einsum("Np,pn->Nn", C_Np, subspace_basis)

    # get projections of all vectors on all projected vectors w
    cos_phi = np.einsum("Nn,Nn->N", W_Nn, vectors) / (  np.einsum("Nn,Nn->N", W_Nn, W_Nn) * np.einsum("Nn,Nn->N", vectors, vectors) )**0.5

    return np.abs(np.arccos(cos_phi))


def conceptspace_feature_similarity(conceptBases, features, verbose=False):
    res = []
    for a in range(len(conceptBases)):
        if verbose:
            print(f"base {a}")
        pas = principle_angle_vector_subspace(features, conceptBases[a])
        res.append(pas)
    return res


def conceptspace_similarity_thresholds(labels, features, conceptBases):
    '''return list of median inter cluster principal angle distances (feature vs. subspace of the corresponding cluster)'''
    print("Determining conceptspace similarity thresholds...")
    thresholds = []
    concepts = np.unique(labels)
    concepts = concepts[concepts>=0]
    print(concepts)
    for i, c in enumerate(concepts):
        mask = np.equal(labels, c)
        similarities = principle_angle_vector_subspace(features[mask], conceptBases[i])
        threshold = np.median(similarities)
        thresholds.append(threshold)

    return thresholds


class ConceptSubspaces():
    """
    Base class for concept subspace discovery.

    Parameters
    -----------        
    feature_maps: hidden activations from model under consideration, shape (batch_size, n_channels, *feature_map_spatial_shape) 
        """
    def __init__(self, feature_maps: np.ndarray) -> None:
        self.feature_maps = feature_maps


    ##### Preprocessing #####
    def get_feature_array(self):
        """
        specify specialized methods for specific concept discovery class
        """
        pass


    def fit_selfrepresentation(self, selfrepresentation_file=None,
                                n_nonzero=-1,
                                outlier_percentile=0.75,
                                refit=True,
                                gamma=50,
                                tau=1.0,
                                random_state=42, # for kmeans in sparse subspace clustering
                                ) -> None:
        """
        Fits (or loads) a sparse selfrepesentation of the feature vectors. Selfrepresentations is automatically saved.
        Parameters
        -----------        
        selfrepresentation_file: str
            If selfrepresentation exists, specify file to load from
        n_nonzero: int, 
            This is an upper bound on the number of nonzero entries of each representation vector.
        outlier_percentile: float, range (0,1]
            ratio of vectors that are considered outliers and will be removed based on l1 norm of selfrepresentation. 
        refit: bool 
            If True, the selfrepresentation is refitted after outlier removal.
        gamma: float,
            noise-controlling hyperparameter
        tau: float, range [0,1]
            sparsity controlling hyperparameter
        """
                                
        n_nonzero = self.feature_array.shape[0] if n_nonzero==-1 else n_nonzero

        self.ssc = subspaceclustering.elasticnet_selfrepresentation.ElasticNetSubspaceClustering(gamma=gamma, tau=tau, n_nonzero=n_nonzero, random_state=random_state, algorithm="lasso_lars"if tau==1.0 else "spams", active_support_params=None)

        if selfrepresentation_file is None:
            self.ssc._self_representation(self.feature_array)

            self.ssc._outlier_detection(outlier_percentile)
            if refit and outlier_percentile<1.0:
                self.ssc._self_representation(self.feature_array[~self.ssc.outlier_mask])
        else:
                self.ssc.representation_matrix_ = sparse.load_npz(selfrepresentation_file)
                self.ssc.outlier_mask = np.load(selfrepresentation_file.replace(".npz","_outlier.npy"))

        self.ssc._representation_to_affinity()


    def cluster(self, n_clusters: int, kmeans: bool=False) -> None:
        """
        Cluster feature vectors, clusters correpond to concept subspaces.
        Sparse subspace clustering is default and selfrepresentation must be fitted before performing spectral clustering on it.

        Parameters
        ----------- 
        n_clusters: number of clusters
        k_means: kmeans clustering instead of sparse subspace clustering      
        """
        if kmeans:
            cluster = sklearn.cluster.KMeans(n_clusters=n_clusters, random_state=0).fit(self.feature_array)
            self.labels = cluster.labels_
        else:
            self.ssc._spectral_clustering(n_clusters)
            self.labels = self.ssc.labels_
        self.concepts = np.unique(self.labels)

    
    def conceptspace_dimensionalities(self, **lPCAkwargs) -> dict:
        """
        Estimates the instrinsic dimensionality of concept subspaces bases on PCA.
        For hyperparameters refer to https://scikit-dimension.readthedocs.io/en/latest/skdim.id.lPCA.html
        """
        lpca = skdim.id.lPCA(**lPCAkwargs)
        conceptIDs = {}
        for c in self.concepts:
            ID = lpca.fit_transform(self.feature_array[self.labels==c])
            conceptIDs[c] = ID
        return conceptIDs


    def conceptspace_bases(self, one_dim=False, **lPCAkwargs):
        """
        Build bases for concept subspaces based on PCA and estimation of instrinsic dimensionality. 
        """
        if one_dim:
            conceptIDs = {c:1 for c in self.concepts}
        else:
            conceptIDs = self.conceptspace_dimensionalities(**lPCAkwargs)
        bases = []
        for c in self.concepts:
            if c!=-1:
                pca = sklearn.decomposition.PCA(n_components=None, random_state=42)
                pca.fit(self.feature_array[self.labels==c])
                bases.append(pca.components_.copy()[:conceptIDs[c]])
        self.conceptBases = bases


class TestConceptSubspaces():
    """
    Concept subspace characterization, including:
        - similarity measure to feature vectors
        - similarity measure to other concept subspaces
        - concept completeness quantification
        - global relevance quantification
        - local relevance quantification

    Parameters
    -----------    
    conceptBases: list of np.ndarrays, each of shape (p_c, d), where p_c is the dimensionality of concept subspace c
    assert_intersection: bool
        Assert that concept subspaces do not intersect.
    """
    def __init__(self, conceptBases: list, assert_intersection=True) -> None:
        self.n_concepts = len(conceptBases)
        
        #check intersection between subspaces
        conceptspace = np.concatenate(conceptBases, axis=0)
        dim_concat = conceptspace.shape[0]
        self.conceptspace_union = orth(conceptspace.T).T
        dim_union = self.conceptspace_union.shape[0]
        dim_intersection = dim_concat - dim_union
        if assert_intersection:
            assert dim_intersection==0, "Concept subspaces intersect."

        # add conceptspace orthogonal complement as last concept
        complement_basis = self.basis_of_complement(self.conceptspace_union)
        self.conceptBases = conceptBases + [complement_basis]
    

    def conceptspace_conceptspace_similarities(self, conceptBases_b=None):
        '''
        Computes n_concepts x n_concepts matrix of concept similarities.
        '''
        if conceptBases_b is None:
            conceptBases_b = self.conceptBases

        n_concepts_a = len(self.conceptBases)
        n_concepts_b = len(conceptBases_b)

        similarities = np.empty((n_concepts_a,n_concepts_b))
        for a in range(n_concepts_a):
            for b in range(n_concepts_b):
                pas = principle_angles_subspace_subspace(self.conceptBases[a], conceptBases_b[b])
                similarities[a,b] = grassmann_distance_from_principal_angles(pas)
        return similarities


    @staticmethod
    def basis_of_union(bases: list):
        union = np.concatenate(bases)
        union = orth(union.T).T
        return union


    @staticmethod
    def projection_matrix(basis, to_complement=False):
        # TODO check orthonormality
        P =  basis.T@basis
        if to_complement:
            P = np.eye(*P.shape) - P
        return P


    @staticmethod
    def basis_of_complement(basis):
        P_orth = TestConceptSubspaces.projection_matrix(orth(basis.T).T, to_complement=True)
        try:
            complement_basis = orth(P_orth.T, rcond=1e-5).T
            assert (complement_basis.shape[0] + basis.shape[0]) == basis.shape[1], "basis does not cover orhogonal complement"
        except AssertionError:
            complement_basis = orth(P_orth.T, rcond=1e-4).T
            assert (complement_basis.shape[0] + basis.shape[0]) == basis.shape[1], "basis does not cover orhogonal complement"
        return complement_basis


    @staticmethod
    def _concept_completeness(weight_vector, union):
        weight_vector = weight_vector / norm(weight_vector)
        P = TestConceptSubspaces.projection_matrix(union)
        P_c = TestConceptSubspaces.projection_matrix(union, to_complement=True)
        
        return norm(P@weight_vector), norm(P_c@weight_vector)


    def orthogonalize_subspaces(self, weight_vector):
        """
        Optional post-hoc orthogonalization of concept subsoaces.
        Iteratively project subspaces into orthogonal complement of previous subspaces.
        Concept subspaces are ordered by completeness based on weight vector of final linear classification layer.

        Parameters
        -----------
        weight_vector: np.ndarray
            weight vector of final linear classification layer corresponding to class under consideration.
        """
        bases_orth = []
        bases_old = list(self.conceptBases.copy())[:-1]

        for i in range(len(bases_old)): #last basis is orthogonal complement
            completeness = np.empty(len(bases_old ))
            bases_tmp = []
            for j in range(len(bases_old)):
                basis = bases_old[j]
                if i>0:
                    # rotate to orthogonal complement of all resident bases
                    bases_orth_union = self.basis_of_union(bases_orth)
                    Pc_union = self.projection_matrix(bases_orth_union, to_complement=True)
                    basis = orth(Pc_union@basis.T).T

                bases_tmp.append(basis)
                union_temp = self.basis_of_union(bases_orth + [basis])
                completeness[j] = self._concept_completeness(weight_vector, union_temp)[0]

            print(completeness)
            next_idx = completeness.argmax()
            print(completeness.max(), next_idx)
            bases_orth.append(bases_tmp[next_idx])
            bases_old.pop(next_idx)
        
        self.conceptspace_union = self.basis_of_union(bases_orth)
        self.conceptBases = bases_orth + [self.conceptBases[-1]]
        

    @staticmethod
    def _concept_idx(bases):
        d_i = np.array([b.shape[0] for b in bases])
        start_idx = np.insert(d_i.cumsum(), 0, 0)[:-1]
        end_idx = d_i.cumsum()
        return start_idx, end_idx


    @staticmethod
    def _subspace_projection(subspaceBases, vector):
        # decompose vector
        C_ij = np.concatenate(subspaceBases)
        coeff_ij = solve(C_ij.T, vector)
        v_ij = coeff_ij[:,np.newaxis] * C_ij

        # sum over concepts
        start_idx, end_idx = TestConceptSubspaces._concept_idx(subspaceBases)
        v_i = np.array([v_ij[s:e].sum(axis=0) for s,e in zip(start_idx, end_idx)])

        return v_i


    def conceptspace_vector_similarities(self, feature_maps: np.ndarray, global_norm=True):
        """
        Similarity between feature vectors and concept subspaces.
        Based on length of vectors in decomposition into subspaces of (normlized) feature vectors.

        Parameters
        -----------
        feature_maps: np.ndarray, shape (batch_size, n_channel, height, width)
        global_norm: bool
            Normalize all feature vectors corresponding to one sample by the maximum norm among them. 
            Otherwise scale each vector to unit length, seperatly.
        """
        batch_size = feature_maps.shape[0]
        n_channel = feature_maps.shape[1]
        spatial_shape = feature_maps.shape[2:]
        similarities = np.empty((batch_size, len(self.conceptBases), *spatial_shape))
        for i in tqdm(range(batch_size)):
            features = feature_maps[i]
            if global_norm:
                max_norm = norm(features, axis=0).max()
                features = features / max_norm
            else:
                features = features / norm(features, axis=0)
            for row in range(spatial_shape[0]):
                for col in range(spatial_shape[1]):
                    f_i = self._subspace_projection(self.conceptBases, features[:,row,col])
                    similarities[i,:,row,col] = norm(f_i, axis=1)

        return similarities


    def conceptspace_global_importance(self, weight_vector: np.ndarray) -> np.ndarray:
        """
        Global importance of concept subspaces based on decompostion of normalized weight vector from linear classification layer into concept subspaces.

        Parameters
        -----------
        weight_vector: np.ndarray
            weight vector of final linear classification layer corresponding to class under consideration.
        """
        weight_vector = weight_vector / norm(weight_vector)
        
        w_i = self._subspace_projection(self.conceptBases, weight_vector)

        global_importance = np.array([norm(w_i_) for w_i_ in w_i])

        return global_importance


    def conceptspace_global_importance_bounds(self, weight_vector: np.ndarray):
        """
        When concept subspaces are not orthogonal, global importance scores do not sum to length 1 of normalized weight vector.
        This method computes the upper and lower bound of the sum of global importances, that is connected to the principle angles between subspaces.
                
        Parameters
        -----------
        weight_vector: np.ndarray
            weight vector of final linear classification layer corresponding to class under consideration.
        """
        w_i_norm = self.conceptspace_global_importance(weight_vector)
        
        min_distance_matrix = np.empty((self.n_concepts, self.n_concepts))
        max_distance_matrix = np.empty((self.n_concepts, self.n_concepts))
        lower_bound = []
        upper_bound = []
        for i,A in enumerate(self.conceptBases[:-1]): #last basis is orthogonal complement
            for j,B in enumerate(self.conceptBases[:-1]):
                pas = principle_angles_subspace_subspace(A,B)
                min_distance_matrix[i,j] = pas.min()
                max_distance_matrix[i,j] = pas.max()
                if j>i:
                    lower_bound.append(2 * w_i_norm[i]*w_i_norm[j] * np.cos(max_distance_matrix[i,j] ))
                    upper_bound.append( 2 * w_i_norm[i]*w_i_norm[j] * np.cos(min_distance_matrix[i,j]) )
        
        return sum(lower_bound), sum(upper_bound)


    def conceptspace_local_importance(self, weight_vector, features):
        """
        Local concept importance (of each feature) based on dot product between feature vector decomposed into concept subspaces and weight vector from linear classification layer.
        
        Parameters
        -----------
        weight_vector: np.ndarray
            weight vector of final linear classification layer corresponding to class under consideration.
        features: np.ndarray of shape (n_features, d)
        """
        features_i = np.empty((features.shape[0], len(self.conceptBases), features.shape[1]) )
        for n,f in tqdm(enumerate(features)):
            features_i[n] = self._subspace_projection(self.conceptBases, f)

        local_importance = np.einsum("Ncd,d->Nc", features_i, weight_vector)

        return local_importance


    def concept_completeness(self, weight_vector):
        """
        Concept completeness based on length of projection of weight vector of linear classification layer into union of all concept subspaces.
        
        Parameters
        -----------
        weight_vector: np.ndarray
            weight vector of final linear classification layer corresponding to class under consideration.
        """
        completeness = self._concept_completeness(weight_vector, self.conceptspace_union)        
        return completeness
