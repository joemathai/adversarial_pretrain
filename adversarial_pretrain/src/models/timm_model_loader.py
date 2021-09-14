from functools import partial

import timm
import torch
import torch.nn as nn


def to_status(m, status):
    if hasattr(m, 'batch_type'):
        m.batch_type = status


to_clean_status = partial(to_status, status='clean')
to_adv_status = partial(to_status, status='adv')
to_mix_status = partial(to_status, status='mix')


class MixLayerNorm(nn.Module):
    '''
    if the dimensions of the tensors from dataloader is [N, 3, 224, 224]
    that of the inputs of the MixBatchNorm2d should be [2*N, 3, 224, 224].
    If you set batch_type as 'mix', this network will using one batchnorm (main bn) to calculate the features corresponding to[:N, 3, 224, 224],
    while using another batch normalization (auxiliary bn) for the features of [N:, 3, 224, 224].
    During training, the batch_type should be set as 'mix'.
    During validation, we only need the results of the features using some specific batchnormalization.
    if you set batch_type as 'clean', the features are calculated using main bn; if you set it as 'adv', the features are calculated using auxiliary bn.
    Usually, we use to_clean_status, to_adv_status, and to_mix_status to set the batch_type recursively. It should be noticed that the batch_type should be set as 'adv' while attacking.
    '''

    def __init__(self, normalized_shape, eps, elementwise_affine, weight, bias):
        super().__init__()
        self.ln = nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
        self.ln_aux = nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
        self.batch_type = 'clean'
        with torch.no_grad():
            self.ln.weight.copy_(weight)
            self.ln.bias.copy_(bias)
            self.ln_aux.weight.copy_(weight)
            self.ln_aux.bias.copy_(bias)

    def forward(self, input):
        if self.batch_type == 'adv':
            input = self.ln_aux(input)
        elif self.batch_type == 'clean':
            input = self.ln(input)
        else:
            assert self.batch_type == 'mix'
            batch_size = input.shape[0]
            input0 = self.ln(input[:batch_size // 2])
            input1 = self.ln_aux(input[batch_size // 2:])
            input = torch.cat((input0, input1), 0)
        return input


class MixBatchNorm2d(nn.Module):
    '''
    if the dimensions of the tensors from dataloader is [N, 3, 224, 224]
    that of the inputs of the MixBatchNorm2d should be [2*N, 3, 224, 224].
    If you set batch_type as 'mix', this network will using one batchnorm (main bn) to calculate the features corresponding to[:N, 3, 224, 224],
    while using another batch normalization (auxiliary bn) for the features of [N:, 3, 224, 224].
    During training, the batch_type should be set as 'mix'.
    During validation, we only need the results of the features using some specific batchnormalization.
    if you set batch_type as 'clean', the features are calculated using main bn; if you set it as 'adv', the features are calculated using auxiliary bn.
    Usually, we use to_clean_status, to_adv_status, and to_mix_status to set the batch_type recursively. It should be noticed that the batch_type should be set as 'adv' while attacking.
    '''

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True,
                 weight=None, bias=None, running_mean=None, running_var=None, num_batches_tracked=0):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum, affine=affine,
                                 track_running_stats=track_running_stats)
        self.aux_bn = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum, affine=affine,
                                     track_running_stats=track_running_stats)
        self.batch_type = 'clean'
        # note: not copying the num_batches tracked
        with torch.no_grad():
            if all(args is not None for args in (weight, bias, running_mean, running_var)):
                self.bn.weight.copy_(weight)
                self.bn.bias.copy_(bias)
                self.bn.running_mean.copy_(running_mean)
                self.bn.running_var.copy_(running_var)
                self.aux_bn.weight.copy_(weight)
                self.aux_bn.bias.copy_(bias)
                self.aux_bn.running_mean.copy_(running_mean)
                self.aux_bn.running_var.copy_(running_var)

    def forward(self, input):
        if self.batch_type == 'adv':
            input = self.aux_bn(input)
        elif self.batch_type == 'clean':
            input = self.bn(input)
        else:
            assert self.batch_type == 'mix'
            batch_size = input.shape[0]
            input0 = self.bn(input[:batch_size // 2])
            input1 = self.aux_bn(input[batch_size // 2:])
            input = torch.cat((input0, input1), 0)
        return input


class MixBNModelBuilder(torch.nn.Module):
    def __init__(self, model_type, num_classes=2, pretrained=True, mix_bn=False, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        super().__init__()
        self.model = timm.create_model(model_type, pretrained=pretrained, num_classes=num_classes)
        if mix_bn:
            print("replacing the BatchNorm2d with MixBatchNorm2d")
            self.replace_bn_layers(self.model)
        self.register_buffer('mean', torch.tensor(mean).view(-1, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(-1, 1, 1))

    def forward(self, imgs):
        imgs.sub_(self.mean).div_(self.std)  # inplace
        return self.model(imgs)

    @staticmethod
    def replace_bn_layers(model):
        # if the model is adv then use auxillary batchnorm
        for n, module in model.named_children():
            if len(list(module.children())) > 0:
                MixBNModelBuilder.replace_bn_layers(module)
            if isinstance(module, nn.BatchNorm2d):
                setattr(model, n, MixBatchNorm2d(num_features=module.num_features,
                                                 eps=module.eps, momentum=module.momentum,
                                                 affine=module.affine,
                                                 track_running_stats=module.track_running_stats,
                                                 running_mean=module.running_mean,
                                                 running_var=module.running_var))
            elif isinstance(module, nn.LayerNorm):
                setattr(model, n, MixLayerNorm(module.normalized_shape, module.eps,
                                               module.elementwise_affine, module.weight, module.bias))
            else:
                continue
