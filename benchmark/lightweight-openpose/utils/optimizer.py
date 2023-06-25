from torch import nn
from torch import optim


def get_optimizer(model_name, net, base_lr):
    if model_name == 'mobilenetv2':
        optimizer = optim.Adam([
            {'params': get_parameters_conv(net.model, 'weight')},
            {'params': get_parameters_conv_depthwise(net.model, 'weight'), 'weight_decay': 0},
            {'params': get_parameters_bn(net.model, 'weight'), 'weight_decay': 0},
            {'params': get_parameters_bn(net.model, 'bias'), 'lr': base_lr * 2, 'weight_decay': 0},
            {'params': get_parameters_conv(net.cpm, 'weight'), 'lr': base_lr},
            {'params': get_parameters_conv(net.cpm, 'bias'), 'lr': base_lr * 2, 'weight_decay': 0},
            {'params': get_parameters_conv_depthwise(net.cpm, 'weight'), 'weight_decay': 0},
            {'params': get_parameters_conv(net.initial_stage, 'weight'), 'lr': base_lr},
            {'params': get_parameters_conv(net.initial_stage, 'bias'), 'lr': base_lr * 2, 'weight_decay': 0},
            {'params': get_parameters_conv(net.refinement_stages, 'weight'), 'lr': base_lr * 4},
            {'params': get_parameters_conv(net.refinement_stages, 'bias'), 'lr': base_lr * 8, 'weight_decay': 0},
            {'params': get_parameters_bn(net.refinement_stages, 'weight'), 'weight_decay': 0},
            {'params': get_parameters_bn(net.refinement_stages, 'bias'), 'lr': base_lr * 2, 'weight_decay': 0},
        ], lr=base_lr, weight_decay=5e-4)
    else:
        optimizer = optim.Adam(net.parameters(), lr=base_lr, weight_decay=5e-4)
    return optimizer




def get_parameters(model, predicate):
    for module in model.modules():
        for param_name, param in module.named_parameters():
            if predicate(module, param_name):
                yield param


def get_parameters_conv(model, name):
    return get_parameters(model, lambda m, p: isinstance(m, nn.Conv2d) and m.groups == 1 and p == name)


def get_parameters_conv_depthwise(model, name):
    return get_parameters(model, lambda m, p: isinstance(m, nn.Conv2d)
                                              and m.groups == m.in_channels
                                              and m.in_channels == m.out_channels
                                              and p == name)


def get_parameters_bn(model, name):
    return get_parameters(model, lambda m, p: isinstance(m, nn.BatchNorm2d) and p == name)