import torch
from torch import nn


def tversky(y_true, y_pred, beta, batch_mean=True):
    tp = torch.sum(y_true * y_pred, axis=(1, 2, 3, 4))
    fn = torch.sum((1. - y_pred) * y_true, axis=(1, 2, 3, 4))
    fp = torch.sum(y_pred * (1. - y_true), axis=(1, 2, 3, 4))
    # tversky = reduce_mean(tp)/(reduce_mean(tp) +
    #                           beta*reduce_mean(fn) +
    #                           (1. - beta)*reduce_mean(fp))
    tversky = tp /\
        (tp + beta * fn + (1. - beta) * fp)

    if batch_mean:
        return torch.mean((1. - tversky))
    else:
        return (1. - tversky)


def kl_loss(mu, sig):
    kl = 0.5 * torch.mean(-1 - sig + torch.square(mu) + torch.exp(sig), axis=-1)
    return torch.mean(kl)


def fc_tversky(y_true, y_pred, beta=0.7, gamma=0.75, batch_mean=True):
    smooth = 1
    tp = torch.sum(y_true * y_pred, axis=(3, 4))
    fn = torch.sum((1. - y_pred) * y_true, axis=(3, 4))
    fp = torch.sum(y_pred * (1. - y_true), axis=(3, 4))
    tversky = (tp + smooth) /\
        (tp + beta * fn + (1. - beta) * fp + smooth)

    focal_tversky_loss = 1 - tversky

    if batch_mean:
        return torch.pow(torch.mean(focal_tversky_loss), gamma)
    else:
        return torch.pow(focal_tversky_loss, gamma)


def focal_loss(y_pred, y_true, gamma=0.75):
    BCE_loss = nn.BCEWithLogitsLoss(reduction='none')(y_pred, y_true)
    pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
    F_loss = (1 - pt)**gamma * BCE_loss

    return torch.mean(F_loss)


def MSE(y_pred, y_true):
    return torch.mean(torch.sum((y_pred - y_true)**2, axis=(1, 2, 3, 4)))


def MAE(y_pred, y_true):
    return torch.mean(torch.sum(torch.abs(y_pred - y_true), axis=(1, 2, 3, 4)))


mink_power = 1


def mink(y_pred, y_true):
    return torch.mean(torch.sum(torch.abs(y_pred - y_true)**mink_power, axis=(1, 2, 3, 4)))  # **(1./mink_power))


# alias
adv_loss = nn.BCELoss()

ce_loss = nn.CrossEntropyLoss()  # mink
