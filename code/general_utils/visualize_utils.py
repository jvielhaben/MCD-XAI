import numpy as np
import skimage.transform

import matplotlib.pyplot as plt

import general_utils.imagenet_utils

#import general_utils.init_plt as init_plt
#init_plt.update_rcParams(half_size_image=True, fontsize=10)

def swin_resize(tensor: np.ndarray):
    
    resize_shape = (14,14)
    tensor_resized = skimage.transform.resize(tensor, output_shape=resize_shape)

    # tile ever pixel 16 times
    tensor_tiled = np.empty((224, 224))
    nreps = 16**2
    for i in range(0,224,16):
        for j in range(0,224,16):
            tensor_tiled [i:i+16,j:j+16] = np.tile(tensor_resized[i//16,j//16][np.newaxis], nreps ).reshape(16,16)   

    return tensor_tiled


def visualize_single_concept(heatmap, img, ax, concept_expression=True, swin=False):
    x = np.arange(224)
    y = np.arange(224)
    X, Y = np.meshgrid(x, y)

    img = general_utils.imagenet_utils.denormalize(img)
    ax.axis("off")
    ax.imshow(img)
    
    if swin:
        heatmap_r =swin_resize(heatmap)
    else:
        heatmap_r = skimage.transform.resize(heatmap, output_shape=img[:,:,0].shape)
    
    
    if concept_expression:
        CS = ax.contour(X,Y, heatmap_r, [ 0.5, ], colors="yellow")
        CS = ax.contour(X,Y, heatmap_r, [ 0.4, ], colors="white")
        ax.imshow(np.ones_like(heatmap_r)*0.2,  cmap="binary", vmin=0, vmax=1, alpha=np.clip(1-heatmap_r,0,1))
    else:
        vmax = np.abs(heatmap).max()
        ax.imshow(heatmap_r, alpha=0.9, cmap="seismic", vmin=-vmax, vmax=vmax)


def visualize_concepts(example_idx, similarities, data, concept_order, swin=False):
    n_examples = example_idx.shape[0]
    n_concepts = len(concept_order)#similarities.shape[1]
    print(n_examples, n_concepts, concept_order)
    
    # plot
    nrows = n_concepts 
    ncols = n_examples
    _, ax = plt.subplots(nrows, ncols,  figsize=(3.29, 3.29/ncols*nrows*1.4))
    if ax.ndim==1:
        ax = ax[:,np.newaxis]
    x = np.arange(224)
    y = np.arange(224)
    X, Y = np.meshgrid(x, y)

    for i in range(n_examples):
        for k,concept in enumerate(concept_order):
            image_id = int(example_idx[i, concept] )
            
            visualize_single_concept(similarities[image_id, concept], data[image_id], ax[k,i], swin=swin)
    
    plt.tight_layout(pad=0.1)
    
    return ax

