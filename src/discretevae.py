

from torch import nn, einsum
from math import log2, sqrt
import torch.nn.functional as F
import torch
from einops import rearrange
import torch.nn.init as init


class EncoderBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=256, num_layers=4) -> None:
        super().__init__()

        self.res_normalize = 1 / (num_layers**2)
        self.ch_equalizer = nn.Identity() if in_channels == out_channels else nn.Conv2d(
            in_channels, out_channels, kernel_size=1)
        hidden_dim = int(out_channels / 4)

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_dim,
                      out_channels=out_channels, kernel_size=1),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.ch_equalizer(x) + self.res_normalize * self.residual(x)

        return out


class EncoderGroup(nn.Module):
    def __init__(self, multiplier, size, num_blocks=2, num_layers_per_block=4, is_top=False) -> None:
        super().__init__()
        assert log2(multiplier).is_integer()

        if is_top:
            self.block_list = [EncoderBlock(
                in_channels=multiplier*size, out_channels=multiplier*size, num_layers=num_layers_per_block) for i in range(num_blocks)]
        else:
            top_block = EncoderBlock(in_channels=int(
                multiplier*size/2), out_channels=multiplier*size)
            block_list = [EncoderBlock(
                in_channels=multiplier*size, out_channels=multiplier*size, num_layers=num_layers_per_block) for i in range(num_blocks - 1)]
            self.block_list = [top_block, *block_list]

        self._group = nn.Sequential(
            *self.block_list, )

    def forward(self, x):
        return self._group(x)


class Encoder(nn.Module):
    def __init__(self, num_tokens=8192,
                 num_groups=4, hidden_base=256, num_blks_per_grp=2, num_layers_per_blk=4) -> None:
        super().__init__()

        in_ch = 3

        groups_multiplier = [2**i for i in range(num_groups)]

        if max(groups_multiplier)*hidden_base > num_tokens:
            raise ValueError(
                "Max of group multiplier * hidden base is greater than codebook dimension, reduce layers or hidden base")

        input_block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_ch, out_channels=groups_multiplier[0]*hidden_base, kernel_size=3, padding=1),
            nn.ReLU())

        top_encoder_group = EncoderGroup(
            multiplier=groups_multiplier[0], size=hidden_base, num_blocks=num_blks_per_grp, num_layers_per_block=num_layers_per_blk, is_top=True)

        encoder_rest_group = [EncoderGroup(multiplier=m, size=hidden_base, num_blocks=num_blks_per_grp, num_layers_per_block=num_layers_per_blk)
                              for m in groups_multiplier[1:]]

        output_block = nn.Conv2d(
            in_channels=groups_multiplier[-1]*hidden_base, out_channels=num_tokens, kernel_size=3, padding=1)

        self.blocks = nn.Sequential(
            input_block,
            top_encoder_group,
            nn.MaxPool2d(kernel_size=2),
            encoder_rest_group[0],
            nn.MaxPool2d(kernel_size=2),
            encoder_rest_group[1],
            nn.MaxPool2d(kernel_size=2),
            encoder_rest_group[2],
            output_block
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        return self.blocks(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=256, num_layers=4) -> None:
        super().__init__()

        self.res_normalize = 1 / (num_layers**2)
        self.ch_equalizer = nn.Identity() if in_channels == out_channels else nn.Conv2d(
            in_channels, out_channels, kernel_size=1)
        hidden_dim = int(out_channels / 4)

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_dim,
                      out_channels=out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.ch_equalizer(x) + self.res_normalize * self.residual(x)

        return out


class DecoderGroup(nn.Module):
    def __init__(self, multiplier, size, num_blocks=2, num_layers_per_block=4, n_init=None) -> None:
        super().__init__()
        assert log2(multiplier).is_integer()

        if n_init is not None:
            top_block = DecoderBlock(
                in_channels=n_init, out_channels=multiplier*size, num_layers=num_layers_per_block)
            rest_block_list = [DecoderBlock(
                in_channels=multiplier*size, out_channels=multiplier*size, num_layers=num_layers_per_block) for i in range(num_blocks - 1)]

        else:
            decreased_size = int(multiplier*size/2)
            top_block = DecoderBlock(
                in_channels=multiplier*size, out_channels=decreased_size, num_layers=num_layers_per_block)
            rest_block_list = [DecoderBlock(
                in_channels=decreased_size, out_channels=decreased_size, num_layers=num_layers_per_block) for i in range(num_blocks - 1)]

        self._group = nn.Sequential(
            top_block,
            *rest_block_list, )

    def forward(self, x):
        return self._group(x)


class Decoder(nn.Module):
    def __init__(self, hidden_base=256, codebook_dim=1024, num_groups=4, n_init=128, num_blocks_per_group=2, num_layers_per_block=4) -> None:
        super().__init__()

        in_ch = codebook_dim

        groups_multiplier = [2**i for i in reversed(range(num_groups))]
        print

        if max(groups_multiplier)*hidden_base > codebook_dim:
            raise ValueError(
                "Max of group multiplier * hidden base is greater than codebook dimension, reduce layers or hidden base")

        input_block = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=n_init,
                      kernel_size=3, padding=1),
            nn.ReLU())

        top_group = DecoderGroup(
            multiplier=groups_multiplier[0], size=hidden_base, num_blocks=num_blocks_per_group, n_init=n_init, num_layers_per_block=num_layers_per_block)

        rest_group = [DecoderGroup(multiplier=m, size=hidden_base, num_blocks=num_blocks_per_group, num_layers_per_block=num_layers_per_block)
                      for m in groups_multiplier[:-1]]

        output_block = nn.Conv2d(
            in_channels=groups_multiplier[-1]*hidden_base, out_channels=6, kernel_size=1)

        self.blocks = nn.Sequential(
            input_block,
            top_group,
            nn.Upsample(scale_factor=2, mode='nearest'),
            rest_group[0],
            nn.Upsample(scale_factor=2, mode='nearest'),
            rest_group[1],
            nn.Upsample(scale_factor=2, mode='nearest'),
            rest_group[2],
            output_block
        )

    def forward(self, x):
        return self.blocks(x)


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner


class ResBlock(nn.Module):
    def __init__(self, chan):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(chan, chan, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(chan, chan, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(chan, chan, 1)
        )

    def forward(self, x):
        return self.net(x) + x


class DiscreteVAE(nn.Module):
    def __init__(
        self,
        image_size=256,
        num_tokens=512,
        codebook_dim=512,
        num_layers=3,
        num_resnet_blocks=0,
        hidden_dim=64,
        channels=3,
        smooth_l1_loss=False,
        temperature=0.9,
        straight_through=False,
        kl_div_loss_weight=0.,
        normalization=((0.5,) * 3, (0.5,) * 3)
    ):
        super().__init__()
        assert log2(image_size).is_integer(), 'image size must be a power of 2'
        assert num_layers >= 1, 'number of layers must be greater than or equal to 1'
        has_resblocks = num_resnet_blocks > 0

        self.image_size = image_size
        self.num_tokens = num_tokens
        self.num_layers = num_layers
        self.temperature = temperature
        self.straight_through = straight_through
        self.codebook = nn.Embedding(num_tokens, codebook_dim)

        hdim = hidden_dim

        enc_chans = [hidden_dim] * num_layers
        dec_chans = list(reversed(enc_chans))

        enc_chans = [channels, *enc_chans]

        dec_init_chan = codebook_dim if not has_resblocks else dec_chans[0]
        dec_chans = [dec_init_chan, *dec_chans]

        enc_chans_io, dec_chans_io = map(lambda t: list(
            zip(t[:-1], t[1:])), (enc_chans, dec_chans))

        enc_layers = []
        dec_layers = []

        for (enc_in, enc_out), (dec_in, dec_out) in zip(enc_chans_io, dec_chans_io):
            enc_layers.append(nn.Sequential(
                nn.Conv2d(enc_in, enc_out, 4, stride=2, padding=1), nn.ReLU()))
            dec_layers.append(nn.Sequential(nn.ConvTranspose2d(
                dec_in, dec_out, 4, stride=2, padding=1), nn.ReLU()))

        for _ in range(num_resnet_blocks):
            dec_layers.insert(0, ResBlock(dec_chans[1]))
            enc_layers.append(ResBlock(enc_chans[-1]))

        if num_resnet_blocks > 0:
            dec_layers.insert(0, nn.Conv2d(codebook_dim, dec_chans[1], 1))

        enc_layers.append(nn.Conv2d(enc_chans[-1], num_tokens, 1))
        dec_layers.append(nn.Conv2d(dec_chans[-1], channels, 1))

        self.encoder = nn.Sequential(*enc_layers)
        self.decoder = nn.Sequential(*dec_layers)

        self.loss_fn = F.smooth_l1_loss if smooth_l1_loss else F.mse_loss
        self.kl_div_loss_weight = kl_div_loss_weight

        # take care of normalization within class
        self.normalization = normalization

    def norm(self, images):
        if self.normalization is None:
            return images

        means, stds = map(lambda t: torch.as_tensor(
            t).to(images), self.normalization)
        means, stds = map(lambda t: rearrange(
            t, 'c -> () c () ()'), (means, stds))
        images = images.clone()
        images.sub_(means).div_(stds)
        return images

    @torch.no_grad()
    @eval_decorator
    def get_codebook_indices(self, images):
        logits = self.forward(images, return_logits=True)
        codebook_indices = logits.argmax(dim=1).flatten(1)
        return codebook_indices

    def decode(
        self,
        img_seq
    ):
        image_embeds = self.codebook(img_seq)
        b, n, d = image_embeds.shape
        h = w = int(sqrt(n))

        image_embeds = rearrange(
            image_embeds, 'b (h w) d -> b d h w', h=h, w=w)
        images = self.decoder(image_embeds)
        return images

    def forward(
        self,
        img,
        return_loss=False,
        return_recons=False,
        return_logits=False,
        temp=None
    ):
        device, num_tokens, image_size, kl_div_loss_weight = img.device, self.num_tokens, self.image_size, self.kl_div_loss_weight
        assert img.shape[-1] == image_size and img.shape[
            -2] == image_size, f'input must have the correct image size {image_size}'

        img = self.norm(img)

        logits = self.encoder(img)

        if return_logits:
            return logits  # return logits for getting hard image indices for DALL-E training

        temp = default(temp, self.temperature)
        soft_one_hot = F.gumbel_softmax(
            logits, tau=temp, dim=1, hard=self.straight_through)
        sampled = einsum('b n h w, n d -> b d h w',
                         soft_one_hot, self.codebook.weight)
        out = self.decoder(sampled)

        if not return_loss:
            return out

        # reconstruction loss

        recon_loss = self.loss_fn(img, out)

        # kl divergence

        logits = rearrange(logits, 'b n h w -> b (h w) n')
        log_qy = F.log_softmax(logits, dim=-1)
        log_uniform = torch.log(torch.tensor([1. / num_tokens], device=device))
        kl_div = F.kl_div(log_uniform, log_qy, None, None,
                          'batchmean', log_target=True)

        loss = recon_loss + (kl_div * kl_div_loss_weight)

        if not return_recons:
            return loss

        return loss, out


class LogitLaplaceLoss(nn.Module):
    def __init__(self):
        super(LogitLaplaceLoss, self).__init__()

    def forward(self, y_pred, y_true):
        # Extract the reconstructed values and scale factors from the decoder output
        recon_x = y_pred[:, :3, :, :]  # Reconstructed values
        scales = y_pred[:, 3:, :, :] + 1e-6   # Scale factors

        # Compute the absolute difference between the original input and the reconstructed output
        abs_diff = torch.abs(y_true - recon_x)

        # Compute the Logit Laplace reconstruction loss
        loss = torch.mean(2.0 * scales * (torch.log(2.0 * scales) - torch.log(abs_diff + 1e-10)) + abs_diff / scales)

        return loss


class DiscreteVAE2(nn.Module):
    def __init__(self, num_tokens=8192, codebook_dim=2048,
                 num_groups=4, hidden_base=256, num_blocks_per_group=2, num_layers_per_block=4, 
                 num_decoder_init=128, temperature=0.9, reconstrution_loss='smooth_l1_loss', kl_div_loss_weight=0.1, logit_laplace_eps=None) -> None:
        super().__init__()

        self._encoder = Encoder(num_tokens=num_tokens, num_groups=num_groups, hidden_base=hidden_base,
                            num_blks_per_grp=num_blocks_per_group, num_layers_per_blk=num_layers_per_block)
        
        self._decoder = Decoder(hidden_base=hidden_base, codebook_dim=codebook_dim, num_groups=num_groups, 
                                n_init=num_decoder_init, num_blocks_per_group=num_blocks_per_group, num_layers_per_block=num_layers_per_block)
        
        self._num_tokens = num_tokens

        self.codebook = nn.Embedding(num_tokens, codebook_dim)

        self._kl_div_loss_weight = kl_div_loss_weight
        self._temperature = temperature

        self._logit_laplace_eps = logit_laplace_eps

        self._kl_loss = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)

        if logit_laplace_eps is not None:
            self._recon_loss = LogitLaplaceLoss()
        else:
            if reconstrution_loss == 'smooth_l1_loss':
                self._recon_loss = torch.nn.SmoothL1Loss(reduction='mean')
            elif reconstrution_loss == 'mse_loss':
                self._recon_loss = torch.nn.MSELoss(reduction='mean')
            else:
                raise ValueError(f'Loss {reconstrution_loss} is not supported')

    @torch.no_grad()
    @eval_decorator
    def get_codebook_indices(self, images):
        logits = self.forward(images, return_logits=True)
        codebook_indices = logits.argmax(dim=1).flatten(1)
        return codebook_indices    

    def forward(self, x, return_logits=False):
        if self._logit_laplace_eps is not None:
            x = (1 - 2 * self._logit_laplace_eps) * x + self._logit_laplace_eps

        logits = self._encoder(x)

        if return_logits:
            return logits
        
        soft_one_hot = F.gumbel_softmax(logits, tau=self._temperature, dim=1)

        sampled_info  = torch.einsum('b n h w, n d -> b d h w',
                                soft_one_hot, self.codebook.weight)

        out = self._decoder(sampled_info)

        mu = out[:, :3, :, :]

        if self._logit_laplace_eps is not None:
            recon_loss = self._recon_loss(x, out)
        else:
            recon_loss = self._recon_loss(x, mu)

        log_prior = torch.log(torch.tensor(1./self._num_tokens))

        # Posterior is p(z|x), which the softmax output of the encoder
        logits = rearrange(logits, 'b n h w -> b (h w) n')
        log_posterior = F.log_softmax(logits, dim=-1)

        kl_div_loss = self._kl_loss(log_prior, log_posterior)

        overall_loss = recon_loss + self._kl_div_loss_weight * kl_div_loss

        if self._logit_laplace_eps is not None:
            reconstructed_img = torch.clamp((mu - self._logit_laplace_eps) / (1 - 2 * self._logit_laplace_eps), 0, 1)
        else:
            reconstructed_img = mu

        return overall_loss, reconstructed_img
                    