import torch

import torchvision.transforms.functional as TF
import torchvision.transforms as T

import deepspeed

import cv2
from matplotlib import pyplot as plt
import os
from src.discretevae import Encoder, Decoder, DiscreteVAE2

class DogDataset(torch.utils.data.Dataset):
    def __init__(self, data_folder, transform=None) -> None:
        super().__init__()

        self._img_paths = []
        for path, subdirs, files in os.walk(data_folder):
            for name in files:
                self._img_paths.append(os.path.join(path, name))

        self._target_img_size = 256
        self._transform = transform

    def __len__(self):
        return len(self._img_paths)

    def __getitem__(self, index):
        selected_img_path = self._img_paths[index]

        img = cv2.imread(selected_img_path)

        h, w, _ = img.shape

        r = self._target_img_size / min(h, w)
        s = (round(r * h), round(r*w))

        img = cv2.resize(img, s, interpolation=cv2.INTER_CUBIC)

        h, w, _ = img.shape

        # Center crop
        x = w/2 - self._target_img_size/2
        y = h/2 - self._target_img_size/2

        crop_img = img[int(y):int(y+self._target_img_size),
                       int(x):int(x+self._target_img_size)]
        
        # Uncomment, if we need to use fp 16 input
        # crop_img = crop_img.astype(np.float16)

        
        if self._transform:
            img_tensor = self._transform(crop_img)
        else:
            img_tensor = crop_img

        
        return img_tensor


def main():

    # Location of the image data
    data_folder = "data/Images"

    # Parameters of Discrete VAE
    num_tokens=8192
    codebook_dim=2048
    num_groups=4
    hidden_base=128
    num_blocks_per_group=2
    num_layers_per_block=4
    num_decoder_init=64
    temperature=0.9
    reconstrution_loss='smooth_l1_loss'
    kl_div_loss_weight=0.01
    logit_laplace_eps=None

    torch.cuda.empty_cache()

    # Initializing the model
    dvae2 = DiscreteVAE2(num_tokens=num_tokens, codebook_dim=codebook_dim, 
                     num_groups=num_groups, hidden_base=hidden_base, 
                     num_blocks_per_group=num_blocks_per_group, num_layers_per_block=num_layers_per_block, 
                     num_decoder_init=num_decoder_init, temperature=temperature, 
                     reconstrution_loss=reconstrution_loss, kl_div_loss_weight=kl_div_loss_weight, 
                     logit_laplace_eps=None)
    
    total_params = sum(p.numel() for p in dvae2.parameters())
    print(f"Number of parameters: {total_params}")

    # Normailization tensors had been calculated seperately
    mean_tensor = (0.4360, 0.4408, 0.4332)
    std_tensor = (0.2619, 0.2639, 0.2616)

    # Creating the Dataset
    dog_data = DogDataset(data_folder=data_folder, transform=T.Compose([T.ToTensor(), T.Normalize(mean=mean_tensor, std=std_tensor)]))
    print(dog_data[1].shape)
 
    # Initialize the deepspeed Engine
    model_engine, optimizer, data_loader, lr_scheduler = deepspeed.initialize(model=dvae2, training_data=dog_data,config='deepspeed_configs/ds_config_1.json')
    
    # Directory for saving checkpoints
    save_dir = "model_chkpt"

    # In case we are resuming the training, we load the checkpoints from the same saved directory
    load_dir = save_dir
    _, client_sd = model_engine.load_checkpoint(load_dir)
    
    # Training 
    num_epochs = 5
    for epoch in range(num_epochs):
        for step, batch in enumerate(data_loader):
            #forward() method
            optimizer.zero_grad()
            batch = batch.to("cuda")
            loss, out = model_engine(batch)

            #runs backpropagation
            model_engine.backward(loss)


            #weight update
            model_engine.step()

            # Priniting the loss
            if step % 500 == 0:
                print(f"epoch: {epoch}, step: {step}, loss: {loss.item()}")
                
            if step == 0:
                model_engine.save_checkpoint(save_dir)

if __name__ == '__main__':
    main()