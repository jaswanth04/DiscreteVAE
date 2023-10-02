
import torch
from discretevae import DiscreteVAE2
from dataset import DewarpingInitialDataset
import sys
import argparse
import deepspeed


def get_args():
    parser = argparse.ArgumentParser(description='CIFAR')
    parser.add_argument('--num_tokens',
                        type=int,
                        default=8192,
                        help='Number of tokens in the codebook')
    parser.add_argument('--codebook_dim',
                        type=int,
                        default=2048,
                        help='Dimensions of codebook')
    parser.add_argument('--num_groups',
                        type=int,
                        default=4,
                        help='Groups of encoder and decoder')
    parser.add_argument('--hidden_base',
                        type=int,
                        default=256,
                        help='multiplier base of enc and ec')
    parser.add_argument('--num_blocks_per_group',
                        type=int,
                        default=2,
                        help='Number of blocks per group')
    parser.add_argument('--num_layers_per_block',
                        type=int,
                        default=4,
                        help='Number of layers per block')
    parser.add_argument('--num_decoder_init',
                        type=int,
                        default=128,
                        help='Initialize block in decoder')
    parser.add_argument('--temparature',
                        type=float,
                        default=0.9,
                        help='Temparature for Gumbel Softmax layer of dvae')
    parser.add_argument('--reconstruction_loss',
                        type=str,
                        default='smooth_l1_loss',
                        help='Reconstruction Loss method')
    parser.add_argument('--kl_div_loss_weight',
                        type=float,
                        default=0.0,
                        help='Weight for KL divergence loss')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=12,
                        help='Number of epochs for training')
    # parser.add_argument('--learning_rate=',
    #                     type=int,
    #                     default=8192,
    #                     help='Number of tokens in the codebook')
    # parser.add_argument('--num_tokens',
    #                     type=int,
    #                     default=8192,
    #                     help='Number of tokens in the codebook')


    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args

def main(args):
    # Model parameters
    num_tokens=args.num_tokens
    codebook_dim=args.codebook_dim
    num_groups=args.num_groups
    hidden_base=args.hidden_base
    num_blocks_per_group=args.num_blocks_per_group
    num_layers_per_block=args.num_layers_per_block
    num_decoder_init=args.num_decoder_init
    temperature=args.temparature
    reconstrution_loss=args.reconstruction_loss
    kl_div_loss_weight=args.kl_div_loss_weight
    logit_laplace_eps=None

    device = torch.device("cuda")

    # Training Loop
    num_epochs = args.num_epochs
    learning_rate = 1e-3
    lr_decay_rate = 0.9
    train_data_folder = "/home/jaswant/Documents/recognition/data/batch_pdf_images_qa/train"
    val_data_folder = "/home/jaswant/Documents/recognition/data/batch_pdf_images_qa/val"

    dvae2 = DiscreteVAE2(num_tokens=num_tokens, 
                        codebook_dim=codebook_dim,
                        num_groups=num_groups,
                        hidden_base=hidden_base,
                        num_blocks_per_group=num_blocks_per_group,
                        num_layers_per_block=num_layers_per_block,
                        num_decoder_init=num_decoder_init,
                        temperature=temperature,
                        reconstrution_loss=reconstrution_loss,
                        kl_div_loss_weight=kl_div_loss_weight,
                        logit_laplace_eps=logit_laplace_eps)#.to(device)

    optimizer = torch.optim.Adam(params=dvae2.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer = optimizer, gamma = lr_decay_rate)

    train_data = DewarpingInitialDataset(
        data_folder=train_data_folder,
        )

    val_data = DewarpingInitialDataset(
        data_folder=val_data_folder,
        )


    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=2, num_workers=1, shuffle=True)

    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=2, num_workers=1, shuffle=True)

    model_engine, optimizer, scheduler = deepspeed.initialize(model=dvae2, 
                                                              optimizer=optimizer, 
                                                              lr_scheduler=scheduler,dist_init_required=None)

    for epoch in range(num_epochs):
        dvae2.train()
        for img in train_loader:
            # img = img.to(device)
            loss, out = dvae2(img)
            # loss.backward()
            # optimizer.step()
            # scheduler.step()

            model_engine.backward(loss)
            model_engine.step()

            train_loss = loss.item()

        dvae2.eval()
        with torch.no_grad():
            for img in val_loader:
                img = img.to(device)
                loss, out = dvae2(img)
                val_loss = loss.item()

        print(f'Epoch: {epoch+1}, Train Loss: {train_loss}, Validation Loss: {val_loss}')



if __name__ == "__main__":
    args = get_args()
    main(args)