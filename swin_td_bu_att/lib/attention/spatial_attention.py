import torch.nn as nn


class SpatialAttention(nn.Module):

	def __init__(self, in_channels=1024):
		super(SpatialAttention, self).__init__()

		self.spatial_attention = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=(1, 1))

	def __str__(self):
		return "SpatialAttention"

	def forward(self, features):
		sp_attention = self.spatial_attention(features)
		return features * sp_attention, sp_attention
