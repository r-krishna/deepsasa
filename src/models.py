import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class NaiveResNetBlock(nn.Module):

	def __init__(self, in_channels, num_planes, kernel_size=3, stride=1):
		super(NaiveResNetBlock, self).__init__()
		self.activation = F.relu
		self.conv1 = nn.Conv1d(in_channels, num_planes, kernel_size, stride=stride, bias=False, padding=kernel_size // 2)
		self.bn1 = nn.BatchNorm1d(num_planes)
		self.conv2 = nn.Conv1d(num_planes, num_planes, kernel_size, stride=stride, bias=False, padding=kernel_size // 2)
		self.bn2 = nn.BatchNorm1d(num_planes)

	def forward(self, x):
		out = self.activation(self.bn1(self.conv1(x)))
		out = self.activation(self.bn2(self.conv2(out)))
		return out


class NaiveResNet1D(nn.Module):

	def __init__(self, in_channels, block, num_layers, num_output_bins, init_planes=32, kernel_size=3, stride=1, dropout=0.2):
		super(NaiveResNet1D, self).__init__()

		self.planes = init_planes
		self.activation = F.relu
		self.conv1 = nn.Conv1d(in_channels, self.planes, kernel_size, stride=stride, bias=False, padding=kernel_size // 2)
		self.bn1 = nn.BatchNorm1d(self.planes)
		self.layers = []

		for i in range(num_layers):
			self.layers.append(block(self.planes, int(self.planes * math.pow(2, i))))
			self.planes = int(self.planes * math.pow(2, i))
		self.dropout = nn.Dropout(p=dropout)
		self.conv2 = nn.Conv1d(self.planes, num_output_bins, kernel_size, stride=stride, bias=False, padding=kernel_size // 2)
		self.bn2 = nn.BatchNorm1d(num_output_bins)

	def forward(self, x):
		out = self.activation(self.bn1(self.conv1(x)))
		for layer in self.layers:
			out = layer(out)
		out = self.activation(self.bn2(self.conv2(out)))
		return out









