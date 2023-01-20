import torch
import numpy as np


def subsample_features(features, feature_subsample_ratio):    
    n_features = int(feature_subsample_ratio*len(features))
    rndgen = np.random.default_rng(42)
    idx = rndgen.choice(len(features), size=n_features, replace=False)
    features = features[idx]
    return features
    

def load_features(model, loader, n_batches, feature_subsample_ratio, 
                    device, l2_norm=True, return_data=False, only_data=False, postprocess_features=True):
    """
    model maps input to intermediate feature layer
    
    e.g.
    vgg16 = models.vgg16(pretrained=True)
    model = quantized_models.generic_split_model(vgg16.features, 30)
    """
    batch_size = loader.batch_size
    n_samples = len(loader.dataset) if n_batches==-1 else n_batches*batch_size
    n_batches = (n_samples//batch_size + (n_samples%batch_size!=0)*1) if n_batches==-1 else n_batches 
    if n_samples>len(loader.dataset):
        print("Warning: n_batches*batch_size>n_samples, setting n_batches to n_samples//batch_size")
        n_batches = n_samples//batch_size
        n_samples = n_batches*batch_size
    datalabel = []
    
    # hack from https://stackoverflow.com/questions/59314174/is-pytorch-dataloader-iteration-order-stable
    # to make dataloader iteration order stable
    # this important for instantiating a new ConceptSSC that uses an old selfrepresentation matrix for calculating conceptspace bases
    # without saving the features
    torch.manual_seed("0")
    for i,batch in enumerate(loader):
        if i>0 and i<n_batches:
            if not only_data:
                with torch.no_grad():
                    features[i*batch_size : (i+1)*batch_size] = model(batch[0].to(device)).cpu()
            if return_data:
                data[i*batch_size : (i+1)*batch_size] = batch[0]
            datalabel.append(batch[1])
        elif i==0:
            if not only_data:
                with torch.no_grad():
                    feature = model(batch[0].to(device)).cpu()
                features = torch.zeros(n_samples, *(feature.size()[1:]))
                features[i*batch_size : (i+1)*batch_size] = feature
            if return_data:
                data = torch.zeros(n_samples, *(batch[0].size()[1:]))
                data[i*batch_size : (i+1)*batch_size] = batch[0]
            datalabel.append(batch[1])
        elif i>=n_batches:
            break

    if not only_data:
        if features.ndim==3:
            # reshape transformer feature maps
            features = features.permute((0,2,1))
            spatial_len = int(features.shape[2]**0.5)
            features = features.view((features.shape[0],features.shape[1],spatial_len,spatial_len))

    datalabel = np.concatenate(datalabel,axis=0)

    if not only_data:
        feature_shape = features.shape

    if not only_data and postprocess_features:
        print("feature shape:",feature_shape)
        # flatten features, new shape is (n_images*height*width) x n_channels
        features = features.permute(0,2,3,1)
        features = features.reshape(-1,features.size(-1)).numpy()

        # You et al assume normalization to unit l2 norm
        if l2_norm:
            norm = (features**2).sum(axis=1, keepdims=True)**0.5
            features = features / norm

        # Subsample features
        if feature_subsample_ratio<1.0:
            features = subsample_features(features, feature_subsample_ratio)
            data = None,
            datalabel=None
            feature_shape=None

    if not return_data:
        data = None

    if only_data:
        return data
    else:
        return features, feature_shape, data, datalabel
