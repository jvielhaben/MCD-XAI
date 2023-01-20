import os
import numpy as np
import torch

import torchvision
import torchvision.datasets as datasets
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset

def random_seeding(seed_value):    
    if seed_value>0:
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)   
        #torch.random.seed(seed_value)    
        #if cuda: torch.cuda.manual_seed_all(seed_value)

class ImageDataframeDataset(Dataset):
    '''creates a dataset based on a given dataframe'''
    def __init__(self, df, transform=None, target_transform=None,
                     loader=default_loader,col_filename="path", col_target="label",
                     col_target_set=None):
        super(ImageDataframeDataset).__init__()
        self.col_target_set = col_target_set
        if(col_target_set is not None):#predefined set for semi-supervised
            self.samples = list(zip(np.array(df[col_filename]), np.array(df[col_target]), np.array(df[col_target_set],dtype=np.int8)))
        else:
            self.samples = list(zip(np.array(df[col_filename]), np.array(df[col_target])))
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        if(self.col_target_set is not None):
            path, target, subset = self.samples[index]
        else:
            path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if(self.col_target_set is not None):
            return sample, [target, subset]
        else:
            return sample, target

    def __len__(self):
        return len(self.samples)

class ImageFolderFiltered(datasets.DatasetFolder):
    """
    Filter imagedata by classes
    """
    def __init__(
            self,
            root,
            transform = None,
            target_transform = None,
            loader = datasets.folder.default_loader,
            is_valid_file = None,
            filter_list = []
    ):
        self.filter_list = filter_list

        # depending on version of torchvision
        # in version 0.11.3, datasets.DatasetFolder only scans through directories found by self.find_classes() and applies is_valid_file only to filename, not filepath so below line does not work
        if int(torchvision.__version__.split(".")[1])<11:
            super().__init__(root=root, transform=transform,target_transform=target_transform,loader=loader,is_valid_file=(lambda x: (os.path.basename(os.path.dirname(x)) in filter_list and "."+(x.split(".")[-1]).lower() in datasets.folder.IMG_EXTENSIONS))  if len(filter_list)>0 else None,extensions=datasets.folder.IMG_EXTENSIONS if len(filter_list)==0 else None)
        else:
            super().__init__(root=root, transform=transform,target_transform=target_transform,loader=loader,is_valid_file=None, extensions=datasets.folder.IMG_EXTENSIONS)
        
    
    def find_classes(self, directory):#method was renamed in torchvision
        return self._find_classes(directory)
    
    def _find_classes(self, directory: str):
        """Finds the class folders in a dataset.

        See :class:`DatasetFolder` for details.
        """
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
        if(len(self.filter_list)>0):
            classes = [c for c in  classes if c in self.filter_list]
            if(len(self.filter_list)!=len(classes)):
                raise RuntimeError(f"Not all specified classes were found.")
        
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


class OrderedDatasetFolder(datasets.vision.VisionDataset):
    """ Dataloader for dataset with samples in root/img_root and targets as npy file in root/target_filename.
    A sorted list of the sample filenames must have the same order as the targets.

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

     Attributes:

        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
            self,
            root: str,
            img_root: str,
            target_filename: str,
            loader,
            transform,
            target_transform = None) -> None:
        super(OrderedDatasetFolder, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        samples = self.make_dataset(self.root, img_root,target_filename)

        self.loader = loader

        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.classes = [] #TODO for compatibility

    @staticmethod
    def make_dataset(directory: str, subdirectory_img: str, target_filename: str):
        """Generates a list of samples of a form (path_to_sample, target).
        Args:
            directory (str): root dataset directory, corresponding to ``self.root``.

        Returns:
            List[Tuple[str, int]]: samples of a form (path_to_sample, class)
        """
        target = np.load(os.path.join(directory,target_filename))

        instances = []
        
        img_directory = os.path.join(directory, subdirectory_img)
        
        for i,fname in enumerate(sorted(os.listdir(img_directory), key=lambda x: int(x[4:][:-4]))):
            path = os.path.join(img_directory, fname)
            item = path, target[i]
            instances.append(item)
        return instances

    
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


    def __len__(self) -> int:
        return len(self.samples)
