import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
import torch.nn.functional as F
import einx
from torch.utils import data
import numpy as np
from typing import Callable
import math
import scipy
from typing import Tuple,Callable
import math
from system_grads import perfect_impulse,l1_norm_cos_f_n,thresh,relu_grad,identity,mat_conv

def sparse_mat_conv(kernel,
					input_indices,
					output_indices):

	return mat_conv(kernel.cpu())[output_indices,:][:,input_indices].to(kernel.device)

def fast_sparse_conv(h_func:Callable,
					 input_indices:torch.Tensor,
					 output_indices:torch.Tensor,
					 device="cuda",
					 correct_sum=False):
	time_indices = einx.rearrange("b -> b 1",output_indices) - einx.rearrange("b -> 1 b",input_indices)
	out = (h_func(time_indices*(time_indices>=0))*(time_indices>=0)).to(device)
	if correct_sum:
		out = out*out.shape[0]
	return out

def matrix_angle(mat1,mat2):
	v1 = einx.rearrange("... -> (...)",mat1)
	v2 = einx.rearrange("... -> (...)",mat2)

	norm1 = torch.norm(v1, dim=0)
	norm2 = torch.norm(v2, dim=0)

	if norm1 == 0 or norm2 == 0:
		return torch.tensor(1.0, device=mat1.device)
	
	return einx.dot("n,n -> ()",v1,v2)/((torch.norm(v1,dim=0)*torch.norm(v2,dim=0)))


class SparseDelayedLinearFunction(torch.autograd.Function):

	@staticmethod
	def forward(ctx,
			 input:torch.Tensor,
			 weight:torch.Tensor,
			 bias:torch.Tensor,
			 h_e:Callable,
			 h_theta:Callable,
			 non_lin:Callable,
			 non_lin_grad:Callable,
			 input_history_func:Callable,
			 theta_func:Callable,
			 grad_func:Callable,
			 input_indices:torch.Tensor,
			 output_indices:torch.Tensor):
		
		z = einx.dot("t d1, d2 d1 -> t d2",input,weight)
		if bias is not None:
			z = z + bias
		h = non_lin(z)

		ctx.save_for_backward(input_history_func(input),
						weight,
						bias,
						non_lin_grad(z),
						input_indices,
						output_indices)
		
		ctx.theta_func = theta_func
		ctx.grad_func = grad_func
		ctx.h_e_func = h_e
		ctx.h_theta_func = h_theta

		return h

	@staticmethod
	def backward(ctx, grad_outputs):
		
		inputs,weight,bias,theta,input_indices,output_indices = ctx.saved_tensors

		h_e = fast_sparse_conv(h_func=ctx.h_e_func,input_indices=input_indices,output_indices=output_indices)
		h_theta = fast_sparse_conv(h_func=ctx.h_theta_func,input_indices=input_indices,output_indices=output_indices)

		h_e = h_e.to(inputs.device,dtype=inputs.dtype)
		h_theta = h_theta.to(inputs.device,dtype=inputs.dtype)

		e_trace = einx.dot("t d2, t d1 -> t d2 d1",theta,inputs)
		e_trace = einx.dot("t_out t_in, t_in d2 d1 -> t_out d2 d1",h_e,e_trace)

		theta = ctx.theta_func(einx.dot("t_out t_in, t_in d2 -> t_out d2",h_theta,theta))

		grad_weight = einx.multiply("t d2, t d2 d1 -> t d2 d1",grad_outputs,e_trace)


		grad_input = einx.dot("t d2, d2 d1 -> t d1",grad_outputs*theta,weight)
		grad_bias = einx.sum("b d2 -> 1 d2",grad_outputs*theta) if bias is not None else None

		return grad_input,grad_weight,grad_bias,None,None,None,None,None,None,None,None,None

class SparseDelayedLinear(nn.Module):

	def __init__(self,
			  in_features:int,
			  out_features:int,
			  h_e_func:torch.Tensor,
			  h_theta_func:torch.Tensor,
			  non_lin:Callable,
			  non_lin_grad:Callable,
			  bias=True,
			  input_history_func:Callable = lambda x:x,
			  theta_func:Callable = lambda x:x,
			  grad_func:Callable = lambda x:x,
			  *args,**kwargs):
		
		super().__init__()
		self.weight = nn.Parameter(torch.randn(out_features, in_features))
		nn.init.xavier_uniform_(self.weight)
		if bias:
			self.bias = nn.Parameter(torch.randn(1,out_features))
			nn.init.xavier_uniform_(self.bias)
		else:
			self.bias = None

		self.h_e_func = h_e_func
		self.h_theta_func = h_theta_func
		self.non_lin = non_lin
		self.non_lin_grad = non_lin_grad
		self.input_func = input_history_func
		self.theta_func = theta_func
		self.grad_func = grad_func

	def forward(self,
			 x,
			 input_indices,
			 output_indices):
		
		return SparseDelayedLinearFunction.apply(x,
										   self.weight,
										   self.bias,
										   self.h_e_func,
										   self.h_theta_func,
										   self.non_lin,
										   self.non_lin_grad,
										   self.input_func,
										   self.theta_func,
										   self.grad_func,
										   input_indices,
										   output_indices)
	
class SparseDelayMLP(nn.Module):

	def __init__(self,
			  n_hidden:int,
			  batch_size:int,
			  e_trace_delay_f:Callable,
			  theta_delay_f:Callable,
			  delay:int,
			  n:int,
			  input_func=identity,
			  grad_func=identity,
			  theta_func=identity,
			  cifar_10=False,
			  store_hebbian=True,
			  harmonics=0,):
		
		super().__init__()

		self.time = torch.arange(batch_size)
		self.delay = delay

		in_dim = 3*32*32 if cifar_10 else 28*28

		print(f"Delay of {2*delay} for delta and {store_hebbian*2*delay} for hebbian term")

		nm_1_x = e_trace_delay_f(self.time,store_hebbian*2*delay,n,harmonics=harmonics,norm=1).sum()
		nm_1_theta = theta_delay_f(self.time,store_hebbian*2*delay,n,harmonics=harmonics,norm=1).sum()
		max_1_theta = theta_delay_f(self.time,store_hebbian*2*delay,n,harmonics=harmonics,norm=nm_1_theta).max()

		self.w1 = SparseDelayedLinear(in_features=in_dim,
								out_features=n_hidden,
								h_e_func=lambda x : e_trace_delay_f(x,store_hebbian*2*delay,n,harmonics=harmonics,norm = nm_1_x),
								h_theta_func=lambda x : theta_delay_f(x,store_hebbian*2*delay,n,harmonics=harmonics,norm = nm_1_theta),
								non_lin=F.relu,
								non_lin_grad=relu_grad,
								bias=True,
								input_history_func=input_func,
								theta_func=lambda x : theta_func(x,max_1_theta),
								grad_func=grad_func)

		nm_2_x = e_trace_delay_f(self.time,store_hebbian*1*delay,n,harmonics=harmonics,norm=1).sum()
		nm_2_theta = theta_delay_f(self.time,store_hebbian*1*delay,n,harmonics=harmonics,norm=1).sum()
		max_2_theta = theta_delay_f(self.time,store_hebbian*1*delay,n,harmonics=harmonics,norm=nm_2_theta).max()
		
		print(f"Delay of {1*delay} for delta and {store_hebbian*1*delay} for hebbian term")
		self.w2= SparseDelayedLinear(in_features=n_hidden,
								out_features=n_hidden,
								h_e_func=lambda x : e_trace_delay_f(x,store_hebbian*1*delay,n,harmonics=harmonics,norm=nm_2_x),
								h_theta_func=lambda x : theta_delay_f(x,store_hebbian*1*delay,n,harmonics=harmonics,norm=nm_2_theta),
								non_lin=F.relu,
								non_lin_grad=relu_grad,
								bias=True,
								input_history_func=input_func,
								theta_func=lambda x: theta_func(x,max_2_theta),
								grad_func=grad_func)
		
		# last layer with prefect gradients

		self.w3 = nn.Linear(n_hidden,10)

	def forward(self,
			 x,
			 t_max,):

		T = x.shape[0]

		perm = torch.randperm(t_max)[:T]
		perm, _ = torch.sort(perm)

		x = self.w1(x,perm,perm+2*self.delay)
		x = self.w2(x,perm,perm+self.delay)
		x = self.w3(x)

		return x
	
def unfold(x, ks, pad):
    x = F.pad(x, (pad, pad, pad, pad), "constant", 0)
    x = x.unfold(2, ks, 1)
    x = x.unfold(3, ks, 1)
    x = x.unsqueeze(1)
    return x

def _input_grad(in_size,
				grad,
				weight,
				theta,
				ks,
				pad,
				grad_func):
	
	grad_out_patches = einx.rearrange("b c h w -> b c (h w)",grad_func(grad*theta))
	weight = einx.rearrange("c_out c_in kh kw -> 1 c_out (c_in kh kw) 1",weight)
	grad_input_patches = (weight*grad_out_patches.unsqueeze(2)).sum(dim=1)

	input_grad = F.fold(
		grad_input_patches,
		output_size=in_size,
		kernel_size=(ks, ks),
		dilation=1,
		padding=pad,
		stride=1
	)  # (N, C_in, H_in, W_in)
	return input_grad

class SparseDelayedConvFunction(torch.autograd.Function):

	@staticmethod
	def forward(ctx,
			 input:torch.Tensor,
			 weight:torch.Tensor,
			 bias:torch.Tensor,
			 h_e:Callable,
			 h_theta:Callable,
			 non_lin:Callable,
			 non_lin_grad:Callable,
			 input_history_func:Callable,
			 theta_func:Callable,
			 grad_func:Callable,
			 input_indices:torch.Tensor,
			 output_indices:torch.Tensor):
		
		c_out,c_in,kh,kw = weight.shape
		z = F.conv2d(input,weight,padding=kh//2)
		if bias is not None:
			z = z + einx.rearrange("1 b -> 1 b 1 1",bias)
		h = non_lin(z)

		ctx.save_for_backward(input_history_func(input),
						weight,
						bias,
						non_lin_grad(z),
						input_indices,
						output_indices)
		
		ctx.grad_func = grad_func
		ctx.theta_func = theta_func
		ctx.h_e_func = h_e
		ctx.h_theta_func = h_theta
		return h
	
	def weight_grad(output_grads,
				theta,
				inputs,
				ks,
				pad):
	
		"""
		output_grads: output gradients (b c_out h_out w_out)
		theta: non-linearity derivative (b c_out h_out w_out)
		inputs: inputs (b c_in h_in w_in)
		ks: kernel size
		pad: padding
		"""

		inputs = unfold(inputs,ks,pad)
		output_grads = einx.rearrange("b c h w -> b c 1 h w 1 1",output_grads)
		theta = einx.rearrange("b c h w -> b c 1 h w 1 1",theta)
		eligibility_trace = output_grads*theta
		grads = inputs*eligibility_trace
		return einx.sum("b c_out c_in h w kh kw -> c_out c_in kh kw",grads)
	
	@staticmethod
	def backward(ctx, grad_outputs):
		inputs,weight,bias,theta,input_indices,output_indices = ctx.saved_tensors

		c_out,c_in,k_h,k_w = weight.shape
		b,_,h_in,w_in = inputs.shape

		# preparing matrices

		h_e = fast_sparse_conv(h_func=ctx.h_e_func,input_indices=input_indices,output_indices=output_indices)
		h_theta = fast_sparse_conv(h_func=ctx.h_theta_func,input_indices=input_indices,output_indices=output_indices)

		h_e = h_e.to(inputs.device,dtype=inputs.dtype)
		h_theta = h_theta.to(inputs.device,dtype=inputs.dtype)
		
		inputs = unfold(inputs,k_h,k_h//2)
		e_trace = inputs*einx.rearrange("b c h w -> b c 1 h w 1 1",theta)

		# applying delay

		theta = ctx.theta_func(einx.dot("t_out t_in, t_in ... -> t_out ...",h_theta,theta))
		e_trace = einx.dot("t_out t_in, t_in ... -> t_out ...",h_e,e_trace)
		
		grad_bias = einx.sum("b d_out h w -> 1 d_out",grad_outputs*theta) if bias is not None else None

		grad_input = _input_grad((h_in,w_in),grad_outputs,weight,theta,k_h,k_h//2,ctx.grad_func)
		
		grad_outputs = einx.rearrange("b c h w -> b c 1 h w 1 1",grad_outputs)

		grad_weight = einx.sum("b c_out c_in h w kh kw -> c_out c_in kh kw",e_trace*grad_outputs)

		return grad_input,grad_weight,grad_bias,None,None,None,None,None,None,None,None,None
		
class SparseDelayedConv(nn.Module):

	def __init__(self,
			  in_channels,
			  out_channels,
			  kernel_size,
			  h_e_func,
			  h_theta_func,
			  non_lin,
			  non_lin_grad,
			  bias=True,
			  input_history_func=identity,
			  grad_func=identity,
			  theta_func=identity,
			  *args,**kwargs):
		
		super().__init__()

		self.weight = nn.Parameter(torch.randn(out_channels,in_channels,kernel_size,kernel_size))
		nn.init.xavier_uniform_(self.weight)
		if bias:
			self.bias = nn.Parameter(torch.randn(1,out_channels))
			nn.init.xavier_uniform_(self.bias)
			self.bias
		else:
			self.bias = None
			
		self.h_e_func = h_e_func
		self.h_theta_func = h_theta_func

		self.non_lin = non_lin
		self.non_lin_grad = non_lin_grad
		self.input_func = input_history_func
		self.grad_func = grad_func
		self.theta_func = theta_func

	def forward(self,
			 x,
			 input_indices,
			 output_indices):

		return SparseDelayedConvFunction.apply(x,
										 self.weight,
										 self.bias,
										 self.h_e_func,
										 self.h_theta_func,
										 self.non_lin,
										 self.non_lin_grad,
										 self.input_func,
										 self.theta_func,
										 self.grad_func,
										 input_indices,
										 output_indices)
	
class PoolNorm(nn.Module):

	def __init__(self,norm_op):

		super().__init__()

		self.norm = norm_op
		self.pool = nn.AvgPool2d(2)

	def forward(self,x):
		x = self.pool(x)
		x = self.norm(x)
		return x

class SparseDelayCNN(nn.Module):

	def __init__(self,
			  n_hidden,
			  batch_size,
			  e_trace_delay_f,
			  theta_delay_f,
			  delay,
			  n,
			  input_func=identity,
			  grad_func=identity,
			  theta_func=identity,
			  store_hebbian=True,
			  harmonics=0,
			  cifar_10=False,
			  sparse_inpulse=False,
			  norm="batch",
			  pdrop=0.0,
			  *args,**kwargs):
		
		super().__init__()

		self.time = torch.arange(batch_size)

		in_channel = 3 if cifar_10 else 1

		if norm == "batch":
			print("Using batchnorm")
			norm_op = lambda x:PoolNorm(nn.BatchNorm2d(x,affine=False))
		elif norm == "group":
			print("Using groupnorm")
			norm_op = lambda x:PoolNorm(nn.GroupNorm(1,x))
		elif norm == "instance":
			norm_op = lambda x:PoolNorm(nn.InstanceNorm2d(x,affine=False))
		else:
			norm_op = lambda x: PoolNorm(nn.Identity())

		self.pool = nn.AvgPool2d(2)

		self.delay = delay

		nm_c1_x = e_trace_delay_f(self.time,store_hebbian*4*delay,n,harmonics=harmonics,norm=1).sum()
		nm_c1_theta = theta_delay_f(self.time,store_hebbian*4*delay,n,harmonics=harmonics,norm=1).sum()
		max_c1_theta = theta_delay_f(self.time,store_hebbian*4*delay,n,harmonics=harmonics,norm=nm_c1_theta).max()

		self.c1 = SparseDelayedConv(in_channels=in_channel,
							  out_channels=32,
							  kernel_size=3,
							  h_e_func=lambda x : e_trace_delay_f(x,store_hebbian*4*delay,n,harmonics=harmonics,norm=nm_c1_x),
							  h_theta_func=lambda x : theta_delay_f(x,store_hebbian*4*delay,n,harmonics=harmonics,norm=nm_c1_theta),
							  non_lin=F.relu,
							  non_lin_grad=relu_grad,
							  input_history_func=input_func,
							  grad_func=grad_func,
							  theta_func=lambda x : theta_func(x,max_c1_theta),)
		
		self.bn1 = norm_op(32)

		nm_c2_x = e_trace_delay_f(self.time,store_hebbian*3*delay,n,harmonics=harmonics,norm=1).sum()
		nm_c2_theta = theta_delay_f(self.time,store_hebbian*3*delay,n,harmonics=harmonics,norm=1).sum()
		max_c2_theta = theta_delay_f(self.time,store_hebbian*3*delay,n,harmonics=harmonics,norm=nm_c2_theta).max()
		
		self.c2 = SparseDelayedConv(in_channels=32,
							  out_channels=64,
							  kernel_size=3,
							  h_e_func=lambda x : e_trace_delay_f(x,store_hebbian*3*delay,n,harmonics=harmonics,norm=nm_c2_x),
							  h_theta_func=lambda x : theta_delay_f(x,store_hebbian*3*delay,n,harmonics=harmonics,norm=nm_c2_theta),
							  non_lin=F.relu,
							  non_lin_grad=relu_grad,
							  input_history_func=input_func,
							  grad_func=grad_func,
							  theta_func=lambda x: theta_func(x,max_c2_theta),)
		
		self.bn2 = norm_op(64)

		nm_c3_x = e_trace_delay_f(self.time,store_hebbian*2*delay,n,harmonics=harmonics,norm=1).sum()
		nm_c3_theta = theta_delay_f(self.time,store_hebbian*2*delay,n,harmonics=harmonics,norm=1).sum()
		max_c3_theta = theta_delay_f(self.time,store_hebbian*2*delay,n,harmonics=harmonics,norm=nm_c3_theta).max()
		
		self.c3 = SparseDelayedConv(in_channels=64,
							  out_channels=128,
							  kernel_size=3,
							  h_e_func=lambda x : e_trace_delay_f(x,store_hebbian*2*delay,n,harmonics=harmonics,norm=nm_c3_x),
							  h_theta_func=lambda x : theta_delay_f(x,store_hebbian*2*delay,n,harmonics=harmonics,norm=nm_c3_theta),
							  non_lin=F.relu,
							  non_lin_grad=relu_grad,
							  input_history_func=input_func,
							  grad_func=grad_func,
							  theta_func=lambda x: theta_func(x,max_c3_theta),)
		
		self.bn3 = norm_op(128)
		
		in_linear = 4*4*128 if cifar_10 else 1152

		nm_w1_x = e_trace_delay_f(self.time,store_hebbian*1*delay,n,harmonics=harmonics,norm=1).sum()
		nm_w1_theta = theta_delay_f(self.time,store_hebbian*1*delay,n,harmonics=harmonics,norm=1).sum()
		max_w1_theta = theta_delay_f(self.time,store_hebbian*1*delay,n,harmonics=harmonics,norm=nm_w1_theta).max()

		self.w1 = SparseDelayedLinear(in_features=in_linear,
								out_features=n_hidden,
								h_e_func=lambda x : e_trace_delay_f(x,store_hebbian*1*delay,n,harmonics=harmonics,norm=nm_w1_x),
								h_theta_func=lambda x : theta_delay_f(x,store_hebbian*1*delay,n,harmonics=harmonics,norm=nm_w1_theta),
								non_lin=F.relu,
								non_lin_grad=relu_grad,
								bias=True,
								input_history_func=input_func,
								theta_func=lambda x: theta_func(x,max_w1_theta),
								grad_func=grad_func)

		self.w2 = nn.Linear(n_hidden,10)

		self.drop1 = nn.Dropout(pdrop)
		self.drop2 = nn.Dropout(pdrop)
		self.drop3 = nn.Dropout(pdrop)

		self.flatten = nn.Flatten(1)

	def forward(self,
			 x,
			 t_max,):

		T = x.shape[0]

		perm = torch.randperm(t_max)[:T]
		perm, _ = torch.sort(perm)

		x = self.c1(x,perm,perm+4*self.delay)
		x = self.bn1(x)
		x = self.drop1(x)

		x = self.c2(x,perm,perm+3*self.delay)
		x = self.bn2(x)
		x = self.drop2(x)

		x = self.c3(x,perm,perm+2*self.delay)
		x = self.bn3(x)
		x = self.drop3(x)

		x = self.flatten(x)

		x = self.w1(x,perm,perm+self.delay)
		x = self.w2(x)

		return x

class GenericHook:

	def __init__(self,layer):

		self.forward_hook = layer.register_forward_hook(self._forward_hook)
		self.with_back_hook = False

	def register_backward(self,module):

		self.with_back_hook = True
		self.backward_hook = module.register_full_backward_hook(self._backward_hook)

	def _forward_hook(self,module,inputs,outputs):

		self.x = inputs[0]
		self.y = outputs

	def _backward_hook(self,module,grad_input,grad_output):
		self.output_grad = grad_output[0]
		self.input_grad = grad_input[0]

	def compute_grad(self,module,output_grad):
		
		input_grad = torch.autograd.grad(
			outputs=self.y,
			inputs=self.x,
			grad_outputs=output_grad,
		)[0]

		return input_grad

	def detach(self):

		self.forward_hook.remove()
		if self.with_back_hook:
			self.backward_hook.remove()


class SSMLayerHook:

	def __init__(self,
			  non_lin_grad,
			  delay=0):
		
		self.non_lin_grad = non_lin_grad
		self.output_grad = None
		self.delay=delay

	def hook(self,module,backward_hook = False):

		self.forward_hook = module.register_forward_hook(self._forward_hook)
		if backward_hook:
			self.store_back = True
			self.backward_hook = module.register_full_backward_hook(self._backward_hook)
		else:
			self.store_back = False

	def _forward_hook(self,module,inputs,outputs):

		self.x = inputs[0]
		self.theta = self.non_lin_grad(outputs)

	def _backward_hook(self,module,grad_input,grad_output):

		self.output_grad = grad_output[0]

	def compute_grad(self,
				  module,
				  output_grad):
		pass

	def detach(self):

		self.forward_hook.remove()
		if self.store_back:
			self.backward_hook.remove()

class LinearHook(SSMLayerHook):

	def __init__(self, non_lin_grad, delay=0):
		super().__init__(non_lin_grad, delay)

	def _forward_hook(self, module, inputs, outputs):
		
		self.x = inputs[0]
		with torch.no_grad():
			z = einx.dot("b d1, d2 d1 -> b d2",self.x,module.weight)
			if module.bias is not None:
				z = z + module.bias
			self.theta = module.non_lin_grad(z)

	def compute_grad(self, module, output_grad):
		
		with torch.no_grad():
			output_grad = output_grad*self.theta
			input_grad = einx.dot("d2 d1, b d2 -> b d1",module.weight,output_grad)
			if self.delay > 0:
				output_grad = output_grad[:-self.delay]
				x = self.x[:-self.delay]
			else:
				x = self.x
			weight_grad = einx.dot("b d2, b d1 -> b d2 d1",output_grad,x)
			weight_grad = einx.sum("b d2 d1 -> d2 d1",weight_grad)
			angle = matrix_angle(weight_grad,module.weight.grad).item()
			if torch.isnan(torch.tensor(angle)):
				similarity = 1
			else:
				similarity = min(1, angle)
		self.x = None
		self.theta = None
		self.output_grad = None
		return input_grad,similarity
	
class ConvHook(SSMLayerHook):

	def __init__(self, non_lin_grad, delay=0):
		super().__init__(non_lin_grad, delay)

	def _forward_hook(self, module, inputs, outputs):
		
		self.x = inputs[0]

		with torch.no_grad():
			c_out,c_in,kh,kw = module.weight.shape
			z = F.conv2d(self.x,module.weight,padding=kh//2)
			if module.bias is not None:
				z = z + einx.rearrange("1 b -> 1 b 1 1",module.bias)
			self.theta = module.non_lin_grad(z)

	def compute_grad(self, module, output_grad):

		with torch.no_grad():
			output_grad = output_grad*self.theta
			input_grad = nn.grad.conv2d_input(self.x.shape,module.weight,output_grad,stride=1,padding=1)
			if self.delay > 0:
				output_grad = output_grad[:-self.delay]
				x = self.x[:-self.delay]
			else:
				x = self.x
			weight_grad = nn.grad.conv2d_weight(x,module.weight.shape,output_grad,stride=1,padding=1)
			angle = matrix_angle(weight_grad,module.weight.grad).item()
			if torch.isnan(torch.tensor(angle)):
				similarity = 1
			else:
				similarity = min(1, angle)
		self.x = None
		self.theta = None
		self.output_grad = None
		return input_grad,similarity
	
class CNNHook:

	def __init__(self,
			  model:SparseDelayCNN,
			  delay=0):

		self.c1_hook = ConvHook(relu_grad,delay=delay)
		self.c1_hook.hook(model.c1)
		self.bn1_hook = GenericHook(model.bn1)
		self.drop1_hook = GenericHook(model.drop1)

		self.c2_hook = ConvHook(relu_grad,delay=delay)
		self.c2_hook.hook(model.c2)
		self.bn2_hook = GenericHook(model.bn2)
		self.drop2_hook = GenericHook(model.drop2)

		self.c3_hook = ConvHook(relu_grad,delay=delay)
		self.c3_hook.hook(model.c3)
		self.bn3_hook = GenericHook(model.bn3)
		self.drop3_hook = GenericHook(model.drop3)

		self.flatten_hook = GenericHook(model.flatten)

		self.w1_hook = LinearHook(relu_grad,delay=delay)
		self.w1_hook.hook(model.w1)

		self.w2_hook = LinearHook(torch.ones_like,delay=delay)
		self.w2_hook = GenericHook(model.w2)
		self.w2_hook.register_backward(model.w2)

		self.model = model

	def backward(self,):

		output_grad = self.w2_hook.input_grad

		# output_grad = self.w2_hook.compute_grad(self.model.w2,output_grad)

		output_grad,w1_similarity = self.w1_hook.compute_grad(self.model.w1,output_grad)
		output_grad = self.flatten_hook.compute_grad(self.model.flatten,output_grad)

		output_grad = self.drop3_hook.compute_grad(self.model.drop3,output_grad)
		output_grad = self.bn3_hook.compute_grad(self.model.bn3,output_grad)
		output_grad,c3_similarity = self.c3_hook.compute_grad(self.model.c3,output_grad)

		output_grad = self.drop2_hook.compute_grad(self.model.drop2,output_grad)
		output_grad = self.bn2_hook.compute_grad(self.model.bn2,output_grad)
		output_grad,c2_similarity = self.c2_hook.compute_grad(self.model.c2,output_grad)

		output_grad = self.drop1_hook.compute_grad(self.model.drop1,output_grad)
		output_grad = self.bn1_hook.compute_grad(self.model.bn1,output_grad)
		output_grad,c1_similarity = self.c1_hook.compute_grad(self.model.c1,output_grad)

		return {
			"w1":w1_similarity,
			"c3":c3_similarity,
			"c2":c2_similarity,
			"c1":c1_similarity
		}
	
	def detach(self,):

		self.c1_hook.detach()
		self.bn1_hook.detach()
		self.drop1_hook.detach()

		self.c2_hook.detach()
		self.bn2_hook.detach()
		self.drop2_hook.detach()

		self.c3_hook.detach()
		self.bn3_hook.detach()
		self.drop3_hook.detach()

		self.flatten_hook.detach()

		self.w1_hook.detach()
		self.w2_hook.detach()

class MLPHook:

	def __init__(self,model,delay=0):

		self.w1_hook = LinearHook(relu_grad,delay)
		self.w1_hook.hook(model.w1)

		self.w2_hook = LinearHook(relu_grad,delay)
		self.w2_hook.hook(model.w2)

		self.w3_hook = LinearHook(torch.ones_like,delay=delay)
		self.w3_hook = GenericHook(model.w3)
		self.w3_hook.register_backward(model.w3)

		self.model = model

	def backward(self,):

		output_grad = self.w3_hook.input_grad

		output_grad,w2_similarity = self.w2_hook.compute_grad(self.model.w2,output_grad)

		output_grad,w1_similarity = self.w1_hook.compute_grad(self.model.w1,output_grad)

		return {
			"w2":w2_similarity,
			"w1":w1_similarity
		}

	
	def detach(self,):

		self.w1_hook.detach()
		self.w2_hook.detach()
		self.w3_hook.detach()