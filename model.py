import torch
import torch.nn as nn



class GOLmodel(nn.Module):
	"""Cell automata model.

	Parameters
	-----------
	n_channels: int 
		Number of channels of the grid

	hidden_channels: int
		Hidden channels that are related to pixel wise 1x1 convolution

	fire_rate: float
		Number between 0 and 1. The lower its is the more likely it is for 
		 cells to be set to zero during the 'stochastic_update' process.

	device: torch.device
		Determines on what device we perform all computations.

	Attributes
	-----------
	"""
	def __init__(self, n_channels = 16, hidden_channels=128, fire_rate =0.5, device = "cuda"):
		super().__init__()

		self.fire_rate = 0.5
		self.n_channels = n_channels
		self.device = device or torch.device("cpu")

		sobel_filter_ = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]])
		scalar = 8.0

		sobel_filter_x = sobel_filter_ / scalar
		sobel_filter_y = sobel_filter_ / scalar

		identity_filter = torch.tensor([0,0,0],[0,1,0],[0,0,0],dtype = torch.float32)
		filters = torch.stack(identitiy_filter, sobel_filter_x,sobel_filter_y)
		filters = filters.repeat((n_channels,1,1))



		self.filters = filters[:, None, ...].to(self.device)

		self.update_module = nn.sequential(
			nn.Conv2d(
				3* n_channels,
				hidden_channels,
				kernel_size = 1,
				),
			nn.ReLU(),
			nn.Conv2d(
				hidden_channels,
				n_channels,
				kernel_size=1,
				bias= False,
				),
			)

		with torch.no_grad():
			self.update_module[2].weight.zero_()

		self.to(self.device)

		def percieve(self, x):
			return nn.functional.conv2d(x, self.filters, padding =1, groups= self.n_channels)

		def update(self, x):
			return self.update_module(x)

		@staticmehod
		def stochastic_update(x, fire_rate):
			device = x.device
			mask = (torch.rand(x[:, :1, :, :].shape) <= fire_rate).to(device, torch.float32)
			return x*mask

		@staticmehod
		def living_mask(self, x):
			return (
				nn.functional.max_pool2d(
				x[:, 3:4, :, :], kernel_size = 3, strides=1, padding=1)
				 > 0.1
				 )

		def forward(self, x):
			pre_life_mask = self.get_living_mask(x)

			y= self.percieve(x)
			dx = self.update(y)
			dx = self.stochastic_update(dx, fire_rate = self.fire_rate)

			x = x+dx

			post_life_mask = self.get_living_mask(x)
			life_mask = (pre_life_mask & post_life_mask).to(torch.float32)

			return x  * life_mask

