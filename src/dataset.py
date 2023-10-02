import cv2
import numpy as np
import torch
import json
import os

import torchvision.transforms.functional as TF
import torchvision.transforms as T
import torchvision


class DogDataset(torch.utils.data.Dataset):
    def __init__(self, data_folder, transform=None) -> None:
        super().__init__()

        self._img_paths = [os.path.join(data_folder, f)
                           for f in os.listdir(data_folder)]
        self._target_img_size = 256
        self._transform = transform

        self._to_tensor = T.ToTensor()

    def __len__(self):
        return len(self._img_paths)

    def __getitem__(self, index):
        selected_img_path = self._img_paths[index]

        img = cv2.imread(selected_img_path)

        h, w, _ = img.shape

        r = self._target_img_size / min(h, w)
        s = (round(r * h), round(r*w))
        # print(f'New size: {s}')

        img = cv2.resize(img, s, interpolation=cv2.INTER_CUBIC)

        h, w, _ = img.shape

        # Center crop
        x = w/2 - self._target_img_size/2
        y = h/2 - self._target_img_size/2

        crop_img = img[int(y):int(y+self._target_img_size),
                       int(x):int(x+self._target_img_size)]

        # img = TF.center_crop(img, output_size=2 * [self._target_img_size])
        # img = torch.unsqueeze(T.ToTensor()(crop_img), 0)

        if self._transform:
            img_tensor = self._transform(crop_img)
        else:
            img_tensor = crop_img

        img_tensor = self._to_tensor(img_tensor)

        return img_tensor
