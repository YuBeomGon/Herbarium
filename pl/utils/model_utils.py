import torch

def trunc_init_(m, std=0.02) :
    torch.nn.init.trunc_normal_(m.weight, mean=0., std=std)
    torch.nn.init.zeros_(m.bias)
    return m