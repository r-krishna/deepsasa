# Focal loss adapted from https://github.com/gokulprasadthekkel/pytorch-multi-class-focal-loss/blob/master/focal_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.constants import train_class_weights


def loss_factory(loss, device):
	""" 
	takes in a loss type and a device to store the weights tensor on and returns a pytorch loss class
	padded values are -1 in labels so do not calculate loss for those values
	"""
	if loss == "cross_entropy":
		return nn.CrossEntropyLoss(ignore_index=-1)
	elif loss == "focal_loss":
		return FocalLoss(weight=train_class_weights.to(device), ignore_index=-1)


class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2, reduction='mean', ignore_index=-100):
        super(FocalLoss, self).__init__(weight, reduction=reduction)
        self.gamma = gamma
        self.weight = weight # weight parameter will act as the alpha parameter to balance class weights
        self.ignore_index = ignore_index

    def forward(self, input, target):
        ce_loss = nn.CrossEntropyLoss(reduction=self.reduction, weight=self.weight, ignore_index=self.ignore_index)
        ce_loss = ce_loss(input, target)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss
