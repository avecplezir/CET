import torch
import math
import scipy
import einx
from scipy.special import gammainc

def f_n(t,t_max,n):
	if n == 0:
		return torch.exp(-t/t_max)
	return t**n * torch.exp(-n*t/t_max)

def max_norm(t,n):
	return (t**n)*math.exp(-n)

def max_norm_f_n(t,t_max,n):
	return (1/max_norm(t_max,n)) * f_n(t,t_max,n)

# def l1_norm_f_n(t,t_max,n):
# 	y = f_n(t,t_max,n)
# 	return y / torch.sum(y)

def l1_norm_f_n(t,
				t_max,
				n,
				norm=None,
				harmonics=None):
	if n==0:
		y = torch.exp(-t/t_max)*(t>=0)
	else:
		a = n/(t_max-0.5)
		y = (1/(a**(n+1)))*(gammainc(n+1,a*(t))-gammainc(n+1,a*torch.maximum(torch.tensor([0],device=t.device,dtype=t.dtype),t-1)))
	if norm is not None:
		return (y*(t>=0))/norm
	return (y*(t>=0)) / torch.sum(y)

def perfect_impulse(t,t_max,*kwarg,**kwargs):
	return (torch.abs(t - t_max)<=0.1).to(torch.float32)

def cos_f_n(t,t_max,n,k=0):
	if k>0:
		c = torch.zeros_like(t,dtype=torch.float32)
		for i in range(k):
			c += torch.cos(2*math.pi*t*(i+1)/t_max)
	else:
		c = torch.ones_like(t)

	if n == 0:
		return torch.exp(-t/t_max)*(t>=0)
	return torch.pow(t*torch.exp(-t/t_max),n)*c

def l1_norm_cos_f_n(t,t_max,n,harmonics=0,*kwarg,**kwargs):
	y = cos_f_n(t,t_max,n,harmonics)
	return y / torch.sum(torch.abs(y))

def mat_f_n(t,t_max,n):
	mat = scipy.linalg.toeplitz(l1_norm_f_n(t,t_max,n).numpy())
	return torch.tril(torch.tensor(mat))

def mat_conv(h):
	mat = scipy.linalg.toeplitz(h)
	return torch.tril(torch.tensor(mat))

def conv_grad(x,theta,delta,h_e,h_delta):
	e_trace = einx.dot("t d2,t d1 -> t d2 d1",theta,x)
	e_trace = einx.dot("t_out t_in,t_in d2 d1 -> t_out d2 d1",h_e,e_trace)
	delta = einx.dot("t_out t_in, t_in d2 -> t_out d2",h_delta,delta)
	grad = einx.multiply("t d2, t d2 d1 -> t d2 d1",delta,e_trace)
	return einx.sum("b d_out d_in -> d_out d_in",grad)

class linear:

	def __init__(self,d1,d2):

		self.w1 = np.randn((d1,d2))
		self.b = np.randn((1,d2))

	def forward(self,x):

		return x @ self.w1 + self.b
	
	def backward(self,x,deltas,thetas):

		return deltas * x @ thetas

def relu(x):

	return x * (x>=0)
	
class MLP:

	def __init__(self,input_dim,dimensions):

		self.layers = []

		for d in dimensions:
			self.layers.append(linear(input_dim,d))
			input_dim = d
	
	def forward(self,x):

		for layer in self.layer[:-1]:

			x = relu(layer.forward(x))

def relu_grad(x:torch.Tensor,t=0):
	return (x>=t).to(x.dtype)

def identity(x):
	return x

def thresh(x,t,scale=1):
	return x * (torch.abs(x)>=(t*scale))

def theta_thresh(x,t,scale=1):
	return (torch.abs(x)>=(t*scale)).to(x.dtype)