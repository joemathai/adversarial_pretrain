import torch
import numpy as np
import functional_attacks.attacks as attacks


class RandomAdvPerturbationNetwork(torch.nn.Module):
    def __init__(self, image_shape, max_depth=3, random_init=False, bit_depth=8,
                 config=None):
        super().__init__()
        self.bits = 2 ** bit_depth  # range of values
        self.config = config
        attks = [key for key, params in self.config.items() if params is not None]
        mixture_depth = np.random.randint(1, min(max_depth + 1, len(attks) + 1))
        sub_network = np.random.choice(attks, mixture_depth, replace=False)
        # note: split on '_' is done to replicate multiple versions of the same attack with different configurations
        self.perturbation_network = torch.nn.Sequential(
            *[getattr(attacks, attk.split('_')[0])(image_shape, random_init=random_init, **self.config[attk]) for attk in sub_network]
        )
        self.name = '_'.join(sorted([net.replace('Transforms', '') for net in sub_network]))

    def forward(self, imgs):
        pert_img = torch.clamp(self.perturbation_network(imgs), 0.0, 1.0)
        # make sure that image has valid bit depth range
        frac = (pert_img * self.bits) % 1
        frac_f = frac / 255.
        out = torch.clamp(pert_img - frac_f + torch.round(frac) / 255., 0.0, 1.0)
        return out