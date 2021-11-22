import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class conv_block(nn.Module):
	def __init__(self, in_channels, out_channels):
		super().__init__()

		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

	def forward(self, inputs):
		x = self.conv1(inputs)
		print(x.shape)


if __name__ == '__main__':
	x = torch.randn((2, 64, 128,128))
	b = conv_block(32, 64)
	b(x)