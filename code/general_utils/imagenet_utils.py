import os
import pandas as pd
import numpy as np

from pathlib import Path
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import timm

from .dataloader_utils import ImageFolderFiltered, random_seeding


#predefined imagenet subsets
policevan = ["police van"]
goldenretriever = ["golden retriever"]
tailedfrog = ["tailed frog"]
imagenette_classes = "tench, English springer, cassette player, chain saw, church, French horn, garbage truck, gas pump, golf ball, parachute".lower().split(", ")
#https://www.researchgate.net/figure/Detailed-class-mapping-from-ImageNet-classes-to-CIFAR-10-classes-This-mapping-is-used_tbl1_348486810
cifar10_classes = ["airliner","beach wagon","hummingbird","siamese cat","ox", "golden retriever", 
                    "tailed frog", 
                    "zebra", "container ship", "police van"] #"trailer truck", 
random4_classes = ['trimaran', 'monastery', 'flamingo', 'model t']
random100_classes = [ 'echidna','bathtub', 'tobacco shop', 'white wolf', 'wombat',
            'scorpion', 'paper towel', 'water ouzel', 'kimono', 'king crab',
            'bullet train', 'gordon setter', 
            'wood rabbit', 'brass',
            'african grey', 'baboon', 'goose', 
            'trimaran', 'monastery',
            'flamingo', 'model t', 'fly', 'rule', 'warthog', 'chain saw',
            'assault rifle', 'soccer ball', 'indian elephant', 'airedale',
            'jackfruit', 'hyena', 'afghan hound', 'brassiere', 'walking stick',
            'kite', 'affenpinscher', 'punching bag', 'thresher',
            'football helmet', 'cardigan', 'water snake', 'container ship',
            'notebook', 'car mirror', 'panpipe', 'lakeside', 'screen',
            'lakeland terrier', 'pomegranate', 'oil filter', 'pomegranate',
            'fountain', 'trombone', 'french horn', 'bull mastiff',
            'studio couch', 'coffee mug', 'flamingo', 'catamaran', 'spotlight',
            'maze', 'water ouzel', 'swab', 
            'walker hound', 'dingo',
            'lesser panda', 'hook', 'mantis', 'junco', 'entlebucher', 'sax',
            'ox', 'four-poster', 'washer', 'sorrel', 'coucal', 'gorilla',
            'jackfruit', 'bookshop', 'barrel', 'computer keyboard', 'sarong',
            'butternut squash', 'leatherback turtle', 'flat-coated retriever',
            'black grouse', 'bagel', 'forklift', 'trimaran', 'lesser panda',
            'goldfish', 'barracouta', 'freight car', 'koala', 'safe',
            'steel arch bridge', 'carousel', 'parallel bars', 'accordion',
            'plate rack']
vehicles = ['tow truck', 'beach wagon', 'pickup',  'trailer truck', 'beach wagon', 'minivan', 'racer', 'sports car', 'bookshop']
paper_selection = ["container ship", "golden retriever", 'bullet train', "police van"]
cats_vs_dog = [['n02125311',
                'n02124075',
                'n02127052',
                'n02123045',
                'n02123159',
                'n02123597',
                'n02123394'],
                ['n02104029', 'n02088364', 'n02112706', 'n02091831', 'n02096294', 'n02108551', 'n02086240', 'n02112018', 'n02108089', 'n02093647', 'n02093859', 'n02086910', 'n02091244', 'n02111129', 'n02110185', 'n02101556', 'n02105412', 'n02107312', 'n02085782', 'n02110806', 'n02088094', 'n02101388', 'n02106662', 'n02107908', 'n02100735', 'n02112137', 'n02089867', 'n02091032', 'n02097658', 'n02097474', 'n02099429', 'n02100236', 'n02105056', 'n02087046', 'n02091635', 'n02101006', 'n02090622', 'n02110627', 'n02105505', 'n02100583', 'n02097298', 'n02106166', 'n02086079', 'n02095889', 'n02096051', 'n02093428', 'n02085620', 'n02097047', 'n02092002', 'n02087394', 'n02095314', 'n02102973', 'n02096437', 'n02107574', 'n02088632', 'n02108915', 'n02109525', 'n02102318', 'n02110341', 'n02113186', 'n02094258', 'n02099712', 'n02099267', 'n02113712', 'n02088238', 'n02113799', 'n02090721', 'n02109047', 'n02105162', 'n02092339', 'n02102040', 'n02091467', 'n02095570', 'n02107683', 'n02113023', 'n02112350', 'n02097130', 'n02110958', 'n02105855', 'n02093256', 'n02093991', 'n02106382', 'n02113978', 'n02098286', 'n02094114', 'n02086646', 'n02105251', 'n02093754', 'n02100877', 'n02111500', 'n02111889', 'n02096177', 'n02102480', 'n02089078', 'n02108000', 'n02105641', 'n02104365', 'n02109961', 'n02088466', 'n02089973', 'n02106550', 'n02108422', 'n02085936', 'n02113624', 'n02110063', 'n02094433', 'n02097209', 'n02098413', 'n02099601', 'n02102177', 'n02091134', 'n02106030', 'n02098105', 'n02107142', 'n02096585', 'n02099849', 'n02090379', 'n02111277']]
diverse = ["basketball", "electric locomotive", "barber chair", "mobile home", ] #"beach wagon", "construction crane", "desktop computer", "cab", "traffic light",
uniform = ["golden retriever", "goldfish", "african grey", "indian elephant"]

cat_vs_rest = [cats_vs_dog[0], []]

class_selection_dict = {"policevan": policevan, 
                        "goldenretriever": goldenretriever, 
                        "tailedfrog": tailedfrog,
                        "imagenette":imagenette_classes,
                        "cifar10":cifar10_classes,
                        "random4":random4_classes,
                        "random100":random100_classes,
                        "vehicles":vehicles,
                        "paper_selection":paper_selection,
                        "cats_vs_dog":cats_vs_dog,
                        "diverse":diverse,
                        "uniform":uniform}

def get_class_map(image_data_folder, filter_classes=None):
    # imagenet classes (using the kaggle LOC_synset_mapping)
    class_map = pd.read_fwf(Path(image_data_folder)/"LOC_synset_mapping.txt", header=None).reset_index().rename(columns={"index":"classId", 0:"class",1:"description"})
    description_df = class_map["description"].apply(lambda x:str(x).lower()).str.split(', ', expand=True)
    description_df = description_df.rename(columns={i:"description{}".format(i) for i in description_df.columns})
    class_map = pd.concat([class_map, description_df],sort=True, axis=1)

    #fix duplicates manually
    class_map.loc[264,"description0"] = "cardigan welsh corgi"
    class_map.loc[264,"description1"] = "cardigan"
    class_map.loc[517,"description0"] = "construction crane"
    class_map.loc[517,"description1"] = "crane"
    class_map.loc[134,"description1"] = "crane bird"
    class_map.loc[639,"description0"] = "tank suit" #still 638 and 639 are in fact the same class
    class_map.loc[639,"description1"] = "maillot"

    #filter
    if(filter_classes is not None):
        class_map=class_map[class_map.description0.apply(lambda x: x in filter_classes)]
    
    return class_map

def get_dict_class_to_description(image_data_folder, filter_classes=None):
    class_map = get_class_map(image_data_folder, filter_classes)
    dict_description_to_class={}
    dict_class_to_description={}
    for id,row in class_map.iterrows():
        dict_description_to_class[row["description0"]]=row["class"]
        dict_class_to_description[row["class"]]=row["description0"]
    return dict_description_to_class #dict_class_to_description

def get_pretrained_model(arch="resnet50", use_timm=True):
    if use_timm:
        return timm.create_model(arch, pretrained=True)
    else:
        return models.__dict__[arch](pretrained=True)

def denormalize(img_tensor):
    img = img_tensor.cpu().numpy().transpose(1,2,0)
    imagenet_mean=[0.485, 0.456, 0.406]
    imagenet_std=[0.229, 0.224, 0.225]
    img = img*imagenet_std+imagenet_mean
    return np.clip(img,0,1)


def get_dataloader(data_path, selected_classes, batch_size, train=False, shuffle=False, norm=True, ace_resize=False):
    ds_dir = os.path.join(data_path, 'train' if train else 'val')

    normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])

    if ace_resize:
        resize = [transforms.Resize((224,224)),]
    else:
        resize = [transforms.Resize(256),
                transforms.CenterCrop(224)]
    if norm:
        tfms = transforms.Compose( resize + [transforms.ToTensor(), normalize])
    else:
        tfms = transforms.Compose( resize + [transforms.ToTensor()])


    dataset = ImageFolderFiltered(
            ds_dir,
            tfms,
            filter_list = selected_classes,
            target_transform = None
            )
    print("Samples in the dataset:",len(dataset))    
    if shuffle:
        random_seeding(42)
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=None,
        num_workers=0,
        worker_init_fn = torch.initial_seed())
    return loader
