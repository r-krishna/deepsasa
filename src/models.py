import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.ResNet1D import DilatedResNet1D, ResNet1D, ResBlock1D


class DeepSASAResNet(nn.Module):

	def __init__(self, in_planes, num_bins=26, num_blocks=21, dropout_proportion=0.2):
		super(DeepSASAResNet, self).__init__()
		self.resnet1D = ResNet1D(in_planes, ResBlock1D, [num_blocks],
								 init_planes=32, kernel_size=17)
		self.dropout = nn.Dropout(p=dropout_proportion)
		expansion1D = int(math.pow(2, self.resnet1D.num_layers - 1))
		out_planes1D = self.resnet1D.init_planes * expansion1D
		self.conv2 = nn.Conv1d(out_planes1D, num_bins, kernel_size=5, padding=2, bias=False)
		self.bn2 = nn.BatchNorm1d(num_bins)

	def forward(self, x):
		x = self.resnet1D(x)
		x = self.dropout(x)
		x = self.conv2(x)
		return x


class DilatedSASAResNet(nn.Module):

	def __init__(self, in_planes, num_bins=26, num_blocks=21, dropout_proportion=0.2):
		super(DeepSASAResNet, self).__init__()
		self.resnet1D = DilatedResNet1D(in_planes, ResBlock1D, [num_blocks],
								 init_planes=32, kernel_size=17)
		self.dropout = nn.Dropout(p=dropout_proportion)
		expansion1D = int(math.pow(2, self.resnet1D.num_layers - 1))
		out_planes1D = self.resnet1D.init_planes * expansion1D
		self.conv2 = nn.Conv1d(out_planes1D, num_bins, kernel_size=5, padding=2, bias=False)
		self.bn2 = nn.BatchNorm1d(num_bins)

	def forward(self, x):
		x = self.resnet1D(x)
		x = self.dropout(x)
		x = self.conv2(x)
		return x
