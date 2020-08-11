from torch.utils.data import Dataset

import pandas as pd
import numpy as np

import os
import cv2


class MelanomaDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        imfolder: str,
        train: bool = True,
        transforms=None,
        meta_features=None,
    ):
        """
        Class initialization
        Args:
            df (pd.DataFrame): DataFrame with data description
            imfolder (str): folder with images
            train (bool): flag of whether a training dataset is being initialized or testing one
            transforms: image transformation method to be applied
            meta_features (list): list of features with meta information, such as sex and age        
        """
        self.df = df
        self.imfolder = imfolder
        self.transforms = transforms
        self.train = train
        self.meta_features = meta_features

    def __getitem__(self, index):
        im_path = os.path.join(
            self.imfolder, self.df.iloc[index]["image_name"] + ".jpg"
        )
        x = cv2.imread(im_path)
        meta = np.array(
            self.df.iloc[index][self.meta_features].values, dtype=np.float32
        )

        if self.transforms:
            x = self.transforms(x)

        if self.train:
            y = self.df.iloc[index]["target"]
            return (x, meta), y
        else:
            return (x, meta)

    def __len__(self):
        return len(self.df)
