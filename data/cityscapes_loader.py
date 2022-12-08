import os
import torch
import numpy as np
from PIL import Image

from torch.utils import data
from data.city_utils import recursive_glob
import data.psp_transform as psp_trsform

class cityscapesLoader(data.Dataset):
    """cityscapesLoader

    https://www.cityscapes-dataset.com

    Data is derived from CityScapes, and can be downloaded from here:
    https://www.cityscapes-dataset.com/downloads/

    Many Thanks to @fvisin for the loader repo:
    https://github.com/fvisin/dataset_loaders/blob/master/dataset_loaders/images/cityscapes.py
    """

    colors = [  # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]

    label_colours = dict(zip(range(19), colors))
    mean_rgb = {"cityscapes": [73.15835921, 82.90891754, 72.39239876],} 

    def __init__(
        self,
        root,
        split="train",
        SSL=None,
        lab_ratio=1.0,
        img_size=[256, 512],
        img_norm=True,
        version="cityscapes",):

        self.root = root
        self.split = split
        self.img_norm = img_norm
        self.n_classes = 19
        self.img_size = img_size # as a list
        self.mean = np.array(self.mean_rgb[version])
        self.files = {}

        self.images_base = os.path.join(self.root, "leftImg8bit", self.split)
        self.annotations_base = os.path.join(
            self.root, "gtFine", self.split
        )

        self.files[split] = recursive_glob(rootdir=self.images_base, suffix=".png")
        self.SSL = SSL 
            
        if self.SSL is not None: 
            # for semi-supervised learning
            train_dataset_size = len(self.files[split])
            train_ids = np.arange(train_dataset_size)
            np.random.seed(0)
            np.random.shuffle(train_ids)
            lab_size = int(lab_ratio*train_dataset_size)
            if SSL == 'lab':
                self.files[SSL] = [self.files[split][i] for i in train_ids[:lab_size]]
            if SSL == 'unlab':
                self.files[SSL] = [self.files[split][i] for i in train_ids[lab_size:]]

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33,]
        self.class_names = [
            "unlabelled",
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic_light",
            "traffic_sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
        ]

        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_classes, range(19)))

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))
        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        """__len__"""
        if self.SSL is None:
            return len(self.files[self.split])
        else:
            return len(self.files[self.SSL])

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        if self.SSL is None:
            img_path = self.files[self.split][index].rstrip()  
        else:
            img_path = self.files[self.SSL][index].rstrip()
          
        lbl_path = os.path.join(
            self.annotations_base,
            img_path.split(os.sep)[-2], # temporary for cross validation
            os.path.basename(img_path)[:-15] + "gtFine_labelIds.png",
        )
        img = Image.open(img_path).convert('RGB')
        lbl = Image.open(lbl_path).convert('L')
        lbl = np.array(lbl, dtype=np.uint8)
        lbl = self.encode_segmap(lbl)
        lbl = Image.fromarray(lbl)
       
        if self.split == 'train': 
            img, lbl = self.train_transform(img, lbl, self.img_size)
        elif self.split == 'val':
            img, lbl = self.val_transform(img, lbl, self.img_size)

        img_name = img_path.split('/')[-1]
        return img[0], lbl[0,0].long()

    def train_transform(self, img, lbl, crop_size):
        trs_form = []
        #mean, std, ignore_label = cfg["mean"], cfg["std"], cfg["ignore_label"]
        mean = [123.675, 116.28, 103.53] 
        std = [58.395, 57.12, 57.375]
        ignore_label = 250
        trs_form.append(psp_trsform.ToTensor())
        trs_form.append(psp_trsform.Normalize(mean=mean, std=std))
        trs_form.append(psp_trsform.RandResize([0.5, 2.0]))
        trs_form.append(psp_trsform.RandomHorizontalFlip())
        crop_type ='rand'
        trs_form.append(
            psp_trsform.Crop(crop_size, crop_type=crop_type, ignore_label=ignore_label)
        )
        transform = psp_trsform.Compose(trs_form)
        return transform(img, lbl)

    def val_transform(self, img, lbl, crop_size):
        trs_form = []
        #mean, std, ignore_label = cfg["mean"], cfg["std"], cfg["ignore_label"]
        mean = [123.675, 116.28, 103.53] 
        std = [58.395, 57.12, 57.375]
        ignore_label = 250
        trs_form.append(psp_trsform.ToTensor())
        trs_form.append(psp_trsform.Normalize(mean=mean, std=std))
        crop_type = 'center'
        trs_form.append(
            psp_trsform.Crop(crop_size, crop_type=crop_type, ignore_label=ignore_label)
        )
        transform = psp_trsform.Compose(trs_form)
        return transform(img, lbl)

    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask
