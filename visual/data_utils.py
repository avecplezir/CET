import torch
import einx
import torch.nn.functional as  F
import math
from torchvision.transforms import v2

CIFAR_TRANSFORMS = v2.Compose([
	v2.RandomHorizontalFlip(p=0.5),
])

def create_grouped_data(dataset,
						n_group:int,):
	
	dataset.targets = torch.Tensor(dataset.targets)
	grouped = []
	min_len = min([(dataset.targets == i).sum() for i in range(10)])
	min_len = min_len - min_len%n_group

	for i in range(10):
		indices =  dataset.targets == i
		x = dataset.data[indices][:min_len]
		y = dataset.targets[indices][:min_len]
		grouped.append((x,y))

	images = torch.stack([torch.Tensor(i[0]) for i in grouped])
	labels = torch.stack([torch.Tensor(i[1]) for i in grouped])

	print(f"\nImage shape: {images.shape}, label shape: {labels.shape}\n")

	# n_classes x n_images x h x w x n_channels if CIFAR

	images = einx.rearrange("c (n k) ... -> (c n) k ...",images,k=n_group)
	labels = einx.rearrange("c (n k) -> (c n) k",labels,k=n_group)

	return images,labels

def get_batch_idx(n:int,
				  batch_size:int,
				  shuffle:bool=True):
	
	if shuffle:
		permutations = torch.randperm(n)
	else:
		permutations = torch.arange(n)

	permutations = permutations[0:batch_size*(n//batch_size)]
	batches = einx.rearrange("(n b) -> n b",permutations,b=batch_size)

	return batches

def one_hot(x,n=10):

	identity = torch.eye(n,dtype=x.dtype,device=x.device)
	return identity[x]

def linear_kernel(ratio):
	weights = torch.arange(1, ratio + 1) / ratio
	weights = torch.cat([weights, weights.flip(0)[1:]])
	return weights

def interpolate(x,y,ratio,n_classes=10,func=linear_kernel):

	b,c,h,w = x.shape
	dilated_x = torch.zeros((b * ratio, c, h, w), dtype=x.dtype, device=x.device)
	dilated_x[::ratio] = x

	one_hot_y = one_hot(y.to(torch.int),n_classes)

	dilated_one_hot = torch.zeros((b*ratio,n_classes),dtype=x.dtype,device=x.device)
	dilated_one_hot[::ratio] = one_hot_y

	# b x c

	weights = func(ratio)
	kernel_length = weights.shape[0]
	weights = einx.rearrange("l -> 1 1 l",weights)
	weights_x = einx.multiply("1 1 l, d1 d2 -> d1 d2 l",weights,torch.eye(c))

	interp_x = F.conv1d(einx.rearrange("b c h w -> (h w) c b",dilated_x),weights_x,padding=kernel_length//2)
	interp_y = F.conv1d(einx.rearrange("b c -> c 1 b",dilated_one_hot),weights,padding=kernel_length//2)

	interp_x = einx.rearrange("(h w) c b -> b c h w",interp_x,h=h)
	interp_y = einx.rearrange("c 1 b ->b c",interp_y)

	return interp_x,interp_y

def square_cos_window(x,k):
	
	def f(x):
		f = torch.cos(math.pi*x) / torch.sqrt(0.25 + torch.square(torch.cos(math.pi*x)))
		return f/torch.max(f)
	
	y = (1/2)*(1+f(x/k))
	y[torch.abs(x)>=k] = 0
	return y

def hann_kernel(k):
	time = torch.linspace(-k,k,2*k+1)
	return square_cos_window(time,k)

def grouped_inerpolated_prep(x,
							 y,
							 interp_factor,
							 func=hann_kernel,
							 *args,
							 **kwargs):

	if len(x.shape) == 4:
		x = einx.rearrange("n k h w -> (n k) 1 h w",x).to(torch.float)
	else:
		x = einx.rearrange("n k h w c -> (n k) c h w",x).to(torch.float)
	y = einx.rearrange("n k -> (n k)",y)
	x,y = interpolate(x,y,interp_factor,func=func)
	x = einx.rearrange("n c h w -> n (c h w)",x)
	return x,y

def grouped_inerpolated_prep_cnn(x,
							 y,
							 interp_factor,
							 func=hann_kernel,
							 *args,
							 **kwargs):

	if len(x.shape) == 4:
		x = einx.rearrange("n k h w -> (n k) 1 h w",x).to(torch.float)
	else:
		x = einx.rearrange("n k h w c -> (n k) c h w",x).to(torch.float)
	y = einx.rearrange("n k -> (n k)",y)
	x,y = interpolate(x,y,interp_factor,func=func)
	return x,y

def vanilla_prep(x,y,sorted=False):

	if len(x.shape)==4:
		x = einx.rearrange("n h w c -> n (c h w)",x)
	else:
		x = einx.rearrange("n h w -> n (h w)",x)
	return x,y.squeeze().to(torch.int64)

def vanilla_prep_cnn(x,y,sorted=False):

	if len(x.shape)==4:
		x = einx.rearrange("n h w c -> n c h w",x)
	else:
		x = einx.rearrange("n h w -> n 1 h w",x)
	return x,y.squeeze().to(torch.int64)

def random_prep_cnn(x,
							 y,
							 interp_factor,
							 func=hann_kernel,
							 *args,
							 **kwargs):

	if len(x.shape) == 4:
		x = einx.rearrange("n k h w -> (n k) 1 h w",x).to(torch.float)
	else:
		x = einx.rearrange("n k h w c -> (n k) c h w",x).to(torch.float)
	y = einx.rearrange("n k -> (n k)",y)
	x,y = interpolate(x,y,interp_factor,func=func)
	x = CIFAR_TRANSFORMS(x)
	return x,y