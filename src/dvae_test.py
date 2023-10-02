

import torch

from dall_e.encoder import Encoder
from dall_e.decoder import Decoder

import torchvision.transforms.functional as TF
import torchvision.transforms as T
from torch import nn
import torch.nn.functional as F

from PIL import Image
import requests
import io
from dataset import DewarpingInitialDataset

import cv2


def load_model(path: str, device: torch.device = None) -> nn.Module:
    if path.startswith('http://') or path.startswith('https://'):
        resp = requests.get(path)
        resp.raise_for_status()

        with io.BytesIO(resp.content) as buf:
            return torch.load(buf, map_location=device)
    else:
        with open(path, 'rb') as f:
            return torch.load(f, map_location=device)


def main():

    logit_laplace_eps = 0.1
    target_image_size = 256
    # trial_file = "/workspaces/research-recognition/recognition_research/data/batch_1/train/dewarped_5a81456de671fd126e34a0d6-1605278613949.jpg"

    # img = Image.open(trial_file)

    # r = target_image_size / min(img.size)
    # s = (round(r * img.size[0]), round(r*img.size[1]))
    # print(f'New size: {s}')

    # img = TF.resize(img, s, interpolation=Image.LANCZOS)

    # img = TF.center_crop(img, output_size=2 * [target_image_size])
    # img = torch.unsqueeze(T.ToTensor()(img), 0)

    # print(f'Shape of image tensor: {img.shape}')

    ToTensor = T.ToTensor()

    train_data = DewarpingInitialDataset(
        data_folder="/workspaces/research-recognition/recognition_research/data/batch_pdf_images_qa/train",
    )

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, num_workers=1, shuffle=True)

    i = 0
    for im in train_loader:
        sample_img = im
        i += 1
        if i > 1:
            break

    # print(img)
    print(sample_img.shape)

    # cv2.imshow("sample", sample_img_reconstructed)

    # Modifying the values so that [0, 255] is within the range of [logit_laplace_eps, 1 - logit_laplace_eps]

    x = (1 - 2 * logit_laplace_eps) * sample_img + logit_laplace_eps
    print(x)

    device = torch.device("cpu")
    enc = load_model(
        "/workspaces/research-recognition/recognition_research/models/dvae/encoder.pkl", device)
    dec = load_model(
        "/workspaces/research-recognition/recognition_research/models/dvae/decoder.pkl", device)

    z_logits = enc(x)
    print(dir(enc))
    print(z_logits.shape)
    z = torch.argmax(z_logits, axis=1)
    z = F.one_hot(z, num_classes=enc.vocab_size).permute(0, 3, 1, 2).float()


if __name__ == "__main__":
    main()
