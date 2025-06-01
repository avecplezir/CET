import torch
from torch import nn
from torch.nn import functional as F
from typing import Callable
import einx
from system_grads import (
	conv_grad,
	mat_conv,
	perfect_impulse,
	relu_grad,
	identity
)

import numpy as np


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

def sparse_mat_conv(kernel,mask):

	return mat_conv(kernel.cpu())[mask,:][:,mask].to(kernel.device)

def matrix_angle(mat1,mat2):
	v1 = einx.rearrange("... -> (...)",mat1)
	v2 = einx.rearrange("... -> (...)",mat2)

	norm1 = torch.norm(v1, dim=0)
	norm2 = torch.norm(v2, dim=0)

	if norm1 == 0 or norm2 == 0:
		return torch.tensor(1.0, device=mat1.device)
	
	return einx.dot("n,n -> ()",v1,v2)/((torch.norm(v1,dim=0)*torch.norm(v2,dim=0)))

class DelayedConvFunction(torch.autograd.Function):

	@staticmethod
	def forward(ctx,
			 input:torch.Tensor,
			 weight:torch.Tensor,
			 bias:torch.Tensor,
			 h_e:torch.Tensor,
			 h_delta:torch.Tensor,
			 non_lin:Callable,
			 non_lin_grad:Callable,
			 input_history_func:Callable,
			 grad_func:Callable,
			 sparse_inputs:torch.Tensor):
		
		c_out,c_in,kh,kw = weight.shape
		z = F.conv2d(input,weight,padding=kh//2)
		if bias is not None:
			z = z + einx.rearrange("1 b -> 1 b 1 1",bias)
		h = non_lin(z)

		ctx.save_for_backward(input_history_func(input),
						weight,
						bias,
						h_e,
						h_delta,
						sparse_inputs)
		
		ctx.grad_func = grad_func
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
	def backward(ctx,grad_outputs):

		input,weight,bias,h_e,h_delta,sparse_inputs = ctx.saved_tensors

		h_e = h_e.to(input.device,dtype=input.dtype)
		h_delta = h_delta.to(input.device,dtype=input.dtype)

		c_out,c_in, k_h,k_w = weight.shape
		b,_,h_in,w_in = input.shape

		# Apply sparsity mask
		input_mask = einx.sum("b ... -> b",grad_outputs**2)>sparse_inputs.item()
		input = einx.multiply("t, t ... -> t ...",input_mask,input)

		# recompute theta instead of storing

		theta = F.conv2d(input,weight,padding=k_h//2)
		if bias is not None:
			theta = theta + einx.rearrange("1 b -> 1 b 1 1",bias)
		theta = theta>=0

		# input grads
		grad_input = _input_grad((h_in,w_in),grad_outputs,weight,theta,k_h,k_h//2,ctx.grad_func)

		input = unfold(input,k_h,k_h//2)

		# applying delay
		grad_outputs = einx.dot("t_out t_in, t_in ... -> t_out ...",h_delta,grad_outputs)
		grad_bias = einx.sum("b d_out h w -> 1 d_out",grad_outputs) if bias is not None else None
		grad_outputs = einx.rearrange("b c h w -> b c 1 h w 1 1",grad_outputs)

		theta = einx.rearrange("b c h w -> b c 1 h w 1 1",theta)

		# computing eligibility trace and applying delay
		e_trace = input*theta
		e_trace = einx.dot("t_out t_in, t_in ... -> t_out ...",h_e,e_trace)
		
		grad_weight = einx.sum("b c_out c_in h w kh kw -> c_out c_in kh kw",e_trace*grad_outputs)

		return grad_input,grad_weight,grad_bias,None,None,None,None,None,None,None

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


class DelayedLinearFunction(torch.autograd.Function):

	@staticmethod
	def forward(ctx,
			 input:torch.Tensor,
			 weight:torch.Tensor,
			 bias:torch.Tensor,
			 h_e:torch.Tensor,
			 h_delta:torch.Tensor,
			 non_lin:Callable,
			 non_lin_grad:Callable,
			 input_history_func:Callable,
			 grad_func:Callable,
			 sparse_inputs:torch.Tensor):
		z = einx.dot("t d1, d2 d1 -> t d2",input,weight)
		if bias is not None:
			z = z + bias
		h = non_lin(z)
		
		ctx.save_for_backward(input_history_func(input),
						weight,
						bias,
						non_lin_grad(z),
						h_e,
						h_delta,
						sparse_inputs)
		ctx.grad_func = grad_func
		return h
	
	@staticmethod
	def backward(ctx, grad_outputs):
		input,weight,bias,theta,h_e,h_delta,sparse_inputs = ctx.saved_tensors

		h_e = h_e.to(input.device,dtype=input.dtype)
		h_delta = h_delta.to(input.device,dtype=input.dtype)

		input_mask = einx.sum("b d -> b",grad_outputs**2)>sparse_inputs.item()

		input = einx.multiply("t, t d -> t d",input_mask,input)

		grad_outputs = ctx.grad_func(grad_outputs * theta)

		grad_input = einx.dot("t d2, d2 d1 -> t d1",grad_outputs,weight)

		e_trace = einx.dot("t d2,t d1 -> t d2 d1",theta,input)
		e_trace = einx.dot("t_out t_in,t_in d2 d1 -> t_out d2 d1",h_e,e_trace)

		grad_outputs = einx.dot("t_out t_in, t_in d2 -> t_out d2",h_delta,grad_outputs)
		grad_weight = einx.multiply("t d2, t d2 d1 -> t d2 d1",grad_outputs,e_trace)
		grad_bias = einx.sum("b d2 -> 1 d2",grad_outputs) if bias is not None else None

		return grad_input,grad_weight,grad_bias,None,None,None,None,None,None,None

class DelayedLinear(nn.Module):
	def __init__(self,
				in_features,
				out_features,
				e_trace_mat,
				delta_mat,
				non_lin,
				non_lin_grad,
				bias=True,
				input_history_func:Callable = lambda x:x,
				grad_func:Callable = lambda x:x,
				sparse_inputs:bool=False):
		
		super().__init__()
		self.weight = nn.Parameter(torch.randn(out_features, in_features))
		nn.init.xavier_uniform_(self.weight)
		if bias:
			self.bias = nn.Parameter(torch.randn(1,out_features))
			nn.init.xavier_uniform_(self.bias)
		else:
			self.bias = None
		self.register_buffer("e_trace_mat",e_trace_mat)
		self.register_buffer("delta_mat",delta_mat)
		self.non_lin = non_lin
		self.non_lin_grad = non_lin_grad
		self.input_func = input_history_func
		self.grad_func = grad_func
		self.sparse_inputs = torch.Tensor([1E-8 if sparse_inputs else -1])
		print(f"sparse input tensor{self.sparse_inputs}")

	def forward(self, input):
		return DelayedLinearFunction.apply(input,self.weight,self.bias,self.e_trace_mat,
										self.delta_mat,self.non_lin,self.non_lin_grad,self.input_func,
										self.grad_func,self.sparse_inputs)
	
class DelayMLP(nn.Module):

	def __init__(self,
			  n_hidden,
			  batch_size,
			  delta_delay_f,
			  e_trace_delay_f,
			  delay,
			  n,
			  store_hebbian=True,
			  harmonics=0,
			  input_func=identity,
			  grad_func=identity,
			  cifar_10=False,
			  sparse_inputs=False):
		
		print(f"Running with sparse inputs: {sparse_inputs}")
		
		super().__init__()

		time = torch.arange(batch_size)

		in_dim = 3*32*32 if cifar_10 else 28*28

		print(f"Delay of {1*delay} for delta and {store_hebbian*1*delay} for hebbian term")
		self.w1 = DelayedLinear(in_dim,n_hidden,e_trace_mat=mat_conv(e_trace_delay_f(time,store_hebbian*1*delay,n,harmonics=harmonics)),
						  delta_mat=mat_conv(delta_delay_f(time,1*delay,n)),
						  non_lin=F.relu,non_lin_grad=relu_grad,input_history_func=input_func,
						  grad_func=grad_func,sparse_inputs=sparse_inputs)
		
		print(f"Delay of {1*delay} for delta and {store_hebbian*1*delay} for hebbian term")
		self.w2 = DelayedLinear(n_hidden,n_hidden,e_trace_mat=mat_conv(e_trace_delay_f(time,store_hebbian*1*delay,n,harmonics=harmonics)),
						  delta_mat=mat_conv(delta_delay_f(time,1*delay,n)),
						  non_lin=F.relu,non_lin_grad=relu_grad,input_history_func=input_func,
						  grad_func=grad_func,sparse_inputs=sparse_inputs)
		
		# Last layer has access to perfect gradients

		print("Using delay on last layer")

		self.w3 = DelayedLinear(n_hidden,10,e_trace_mat=mat_conv(e_trace_delay_f(time,store_hebbian*1*delay,n,harmonics=harmonics)),
						  delta_mat=mat_conv(delta_delay_f(time,1*delay,n)),
						  non_lin=identity,non_lin_grad=torch.ones_like,input_history_func=input_func,
						  grad_func=grad_func,sparse_inputs=sparse_inputs)
		
	def forward(self,x):
		x = self.w1(x)
		x = self.w2(x)
		x = self.w3(x)
		return x
	
class DelayedConv(nn.Module):

	def __init__(self,
			  in_channels,
			  out_channels,
			  kernel_size,
			  e_trace_mat,
			  delta_mat,
			  non_lin,
			  non_lin_grad,
			  bias=True,
			  input_history_func:Callable = lambda x:x,
			  grad_func:Callable = lambda x:x,
			  sparse_inputs:bool=False):
		
		super().__init__()

		self.weight = nn.Parameter(torch.randn(out_channels,in_channels,kernel_size,kernel_size))
		nn.init.xavier_uniform_(self.weight)
		if bias:
			self.bias = nn.Parameter(torch.randn(1,out_channels))
			nn.init.xavier_uniform_(self.bias)
			self.bias
		else:
			self.bias = None

		self.register_buffer("e_trace_mat",e_trace_mat)
		self.register_buffer("delta_mat",delta_mat)

		# register a buffer for the states and a buffer for the gradients
		# use the buffers inside the forward pass
		# add a function to compute the loss and store the input mask
		# update the buffers after the backward pass

		self.non_lin = non_lin
		self.non_lin_grad = non_lin_grad
		self.input_func = input_history_func
		self.grad_func = grad_func
		self.sparse_inputs = torch.Tensor([1E-8 if sparse_inputs else -1])
		print(f"sparse input tensor{self.sparse_inputs}")

	def forward(self,input):

		return DelayedConvFunction.apply(
			input,
			self.weight,
			self.bias,
			self.e_trace_mat,
			self.delta_mat,
			self.non_lin,
			self.non_lin_grad,
			self.input_func,
			self.grad_func,
			self.sparse_inputs
		)
	
class PoolNorm(nn.Module):

	def __init__(self,norm_op):

		super().__init__()

		self.norm = norm_op
		self.pool = nn.AvgPool2d(2)

	def forward(self,x):
		x = self.pool(x)
		x = self.norm(x)
		return x

class DelayCNN(nn.Module):

	def __init__(self,
			  n_hidden,
			  batch_size,
			  delta_delay_f,
			  e_trace_delay_f,
			  delay,
			  n,
			  store_hebbian=True,
			  harmonics=0,
			  input_func=identity,
			  grad_func=identity,
			  cifar_10=False,
			  sparse_inputs=False,
			  norm="batch",
			  pdrop=0.0,
			  *args,**kwargs):
		
		super().__init__()
		
		print(f"Running with sparse inputs: {sparse_inputs}")

		time = torch.arange(batch_size)

		in_channels = 3 if cifar_10 else 1

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

		self.c1 = DelayedConv(in_channels=in_channels,
						out_channels=32,
						kernel_size=3,
						e_trace_mat=mat_conv(e_trace_delay_f(time,store_hebbian*1*delay,n,harmonics=harmonics)),
						delta_mat=mat_conv(delta_delay_f(time,1*delay,n)),
						non_lin=F.relu,
						non_lin_grad=relu_grad,
						input_history_func=input_func,
						grad_func=grad_func,
						sparse_inputs=sparse_inputs)
		
		self.bn1 = norm_op(32)
		
		self.c2 = DelayedConv(in_channels=32,
						out_channels=64,
						kernel_size=3,
						e_trace_mat=mat_conv(e_trace_delay_f(time,store_hebbian*1*delay,n,harmonics=harmonics)),
						delta_mat=mat_conv(delta_delay_f(time,1*delay,n)),
						non_lin=F.relu,
						non_lin_grad=relu_grad,
						input_history_func=input_func,
						grad_func=grad_func,
						sparse_inputs=sparse_inputs)
		
		self.bn2 = norm_op(64)
		
		self.c3 = DelayedConv(in_channels=64,
						out_channels=128,
						kernel_size=3,
						e_trace_mat=mat_conv(e_trace_delay_f(time,store_hebbian*1*delay,n,harmonics=harmonics)),
						delta_mat=mat_conv(delta_delay_f(time,1*delay,n)),
						non_lin=F.relu,
						non_lin_grad=relu_grad,
						input_history_func=input_func,
						grad_func=grad_func,
						sparse_inputs=sparse_inputs)
		
		self.bn3 = norm_op(128)
		
		in_linear = 4*4*128 if cifar_10 else 1152

		self.w1 = DelayedLinear(in_linear,n_hidden,e_trace_mat=mat_conv(e_trace_delay_f(time,store_hebbian*1*delay,n,harmonics=harmonics)),
						  delta_mat=mat_conv(delta_delay_f(time,1*delay,n)),
						  non_lin=F.relu,non_lin_grad=relu_grad,input_history_func=input_func,
						  grad_func=grad_func,sparse_inputs=sparse_inputs)
		
		print("Using delay on last layer")

		self.w2 = DelayedLinear(512,10,e_trace_mat=mat_conv(e_trace_delay_f(time,store_hebbian*1*delay,n,harmonics=harmonics)),
						  delta_mat=mat_conv(delta_delay_f(time,1*delay,n)),
						  non_lin=identity,non_lin_grad=torch.ones_like,input_history_func=input_func,
						  grad_func=grad_func,sparse_inputs=sparse_inputs)
		
		self.drop1 = nn.Dropout(pdrop)
		self.drop2 = nn.Dropout(pdrop)
		self.drop3 = nn.Dropout(pdrop)

		self.flatten = nn.Flatten(1)

	def forward(self,x):

		x = self.c1(x)
		x = self.bn1(x)
		x = self.drop1(x)

		x = self.c2(x)
		x = self.bn2(x)
		x = self.drop2(x)

		x = self.c3(x)
		x = self.bn3(x)
		x = self.drop3(x)
		
		x = self.flatten(x)
		x = self.w1(x)
		x = self.w2(x)
		return x
	
class LayerNorm2D(nn.Module):

	def __init__(self,*args,**kwargs):

		super().__init__()

	def forward(self,x):

		return F.layer_norm(x,x.shape[1:])

class GenericHook:

	def __init__(self,layer):

		self.forward_hook = layer.register_forward_hook(self._forward_hook)

	def _forward_hook(self,module,inputs,outputs):

		self.x = inputs[0]
		self.y = outputs

	def compute_grad(self,module,output_grad):
		
		input_grad = torch.autograd.grad(
			outputs=self.y,
			inputs=self.x,
			grad_outputs=output_grad,
		)[0]

		return input_grad

	def detach(self):

		self.forward_hook.remove()

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
			  model:DelayCNN,
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
		self.w2_hook.hook(model.w2,backward_hook=True)

		self.model = model

	def backward(self,):

		output_grad = self.w2_hook.output_grad

		output_grad,w2_similarity = self.w2_hook.compute_grad(self.model.w2,output_grad)

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
			"w2":w2_similarity,
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

		self.w3_hook = LinearHook(torch.ones_like,delay)
		self.w3_hook.hook(model.w3,backward_hook=True)

		self.model = model

	def backward(self,):

		output_grad = self.w3_hook.output_grad
		output_grad,w3_similarity = self.w3_hook.compute_grad(self.model.w3,output_grad)

		output_grad,w2_similarity = self.w2_hook.compute_grad(self.model.w2,output_grad)

		output_grad,w1_similarity = self.w1_hook.compute_grad(self.model.w1,output_grad)

		return {
			"w3":w3_similarity,
			"w2":w2_similarity,
			"w1":w1_similarity
		}

	
	def detach(self,):

		self.w1_hook.detach()
		self.w2_hook.detach()
		self.w3_hook.detach()