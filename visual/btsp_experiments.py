import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import v2
import einx
from torch import nn
from torch.nn import functional as F
from btsp_models import DelayCNN, DelayMLP, CNNHook, MLPHook
import numpy as np
from torch.utils import data
import argparse
from system_grads import perfect_impulse,l1_norm_f_n,thresh,relu_grad,identity
from data_utils import (grouped_inerpolated_prep,
						vanilla_prep,
						get_batch_idx,
						create_grouped_data,
						grouped_inerpolated_prep_cnn,
						vanilla_prep_cnn,
						random_prep_cnn)
import os
from datetime import datetime
from typing import Tuple
import math
import wandb
torch.manual_seed(0)

BATCH_SIZE = 128
N_HIDDEN = 128
N_TRAINING_STEPS = 5000

def cosine_warmup(epoch,n_epochs,warmup):
	n_warmup = int(n_epochs*warmup)
	if epoch < n_warmup:
		return epoch/int(n_warmup)
	else:
		return max(0.1,0.5+ 0.5*math.cos(math.pi*(epoch-n_warmup)/(n_epochs-n_warmup)))

def avg_dict(d_list):
	print(len(d_list))
	return {key:np.mean([i[key] for i in d_list]) for key in d_list[0].keys()}

def train(model: DelayMLP,
		  hook_class,
		  x_train,
		  y_train,
		  x_val,
		  y_val,
		  batch_size,
		  val_batch_size,
		  n_groups,
		  interp_factor,
		  n_train_steps: int,
		  val_steps: int,
		  device="cuda",
		  val_prep_batch=grouped_inerpolated_prep,
		  train_prep_batch=vanilla_prep,
		  lr=1E-3,
		  weight_decay=1E-1,
		  optim_alg=torch.optim.SGD,
		  top_quantile=0.0,
		  sparsity_scaling=1,
		  warmup_p=0.05,
		  delay:int=0,
		  track_sim=False):
	
	n_steps = 0

	losses = []
	running_sim = []
	
	loss_fn = nn.CrossEntropyLoss()
	reduced_loss = nn.CrossEntropyLoss()
	non_reduced_loss = nn.CrossEntropyLoss(reduction="none")
	
	def sparse_training_loss(x,y,top_quantile_loss=0.9):

		if top_quantile_loss>0:
			with torch.no_grad():
				y_pred = model(x)
				loss_batch = non_reduced_loss(y_pred,y)

				valid_mask = ~torch.isnan(loss_batch)
				loss_batch = loss_batch[valid_mask]
				x = x[valid_mask]
				y = y[valid_mask]

				if loss_batch.numel() == 0:
					return torch.tensor(0.0, device=loss_batch.device, requires_grad=True)

				loss_th = torch.quantile(loss_batch,top_quantile_loss)
				idx = loss_batch >= loss_th
				# raise exception here to see why it crashes
				x = x[idx]
				y = y[idx]
				if idx.sum() <= 1:
					return torch.tensor(0.0, device=loss_batch.device, requires_grad=True)

		y_pred = model(x)
		loss = reduced_loss(y_pred,y)
		return loss

	model = model.to(device)

	print(f"\ntraining with lr {lr} and weight decay {weight_decay}\n")

	optimizer = optim_alg(model.parameters(), lr=lr, weight_decay=weight_decay)

	lambda_lr = lambda step: cosine_warmup(step,n_train_steps,warmup_p)

	scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lambda_lr)
	
	training_losses = []
	validation_losses = []
	training_accuracies = []
	validation_accuracies = []

	if track_sim:
		hook = hook_class(model,delay)

	def val_loop():

		model.eval()
		val_losses = []
		correct = 0
		total = 0

		batches = get_batch_idx(len(x_val),val_batch_size,True)

		l = 0
		
		for b,indices in enumerate(batches):

			x,y = val_prep_batch(x_val[indices],y_val[indices])
			
			x = x.to(device)
			y = y.to(device)

			with torch.no_grad():

				y_pred = model(x)
				
				# Compute validation loss
	
				loss = loss_fn(y_pred, y.to(torch.int64))
				val_losses.append(loss.item())
				
				# Compute validation accuracy
				_, predicted = torch.max(y_pred, 1)
				correct += (predicted == y).sum().item()
				total += y.size(0)

		model.train()

		val_accuracy = correct / total if total > 0 else 0
		return (np.mean(val_losses)), val_accuracy
	
	while n_steps < n_train_steps:
		correct = 0
		total = 0

		batches = get_batch_idx(len(x_train),batch_size//(n_groups*interp_factor),True)
		
		for b,indices in enumerate(batches):

			x,y = train_prep_batch(x_train[indices],y_train[indices])
			
			x = x.to(device)
			y = y.to(device)

			if (n_steps == 0) & (b == 0):
				print(x.shape)
				print(y.shape)
			
			# Forward pass and compute loss
			loss = sparse_training_loss(x,y,top_quantile)
			loss.backward(retain_graph=track_sim)
			nn.utils.clip_grad_norm_(model.parameters(),0.5)
			losses.append(loss.item())
			
			# Optimizer step
			optimizer.step()
			if track_sim:
				sim = hook.backward()
				running_sim.append(sim)
			else:
				running_sim.append(0.0)

			model.zero_grad()
			scheduler.step()
			
			n_steps += 1

			if n_steps % val_steps == 0:
				if track_sim:
					hook.detach()
				val_loss, val_accuracy = val_loop()
				if track_sim:
					hook = hook_class(model,delay)

				similarity = avg_dict(running_sim)
				
				print(f"Training step: {n_steps}/{n_train_steps} | Loss: {np.mean(losses)} | Val loss: {val_loss} | Val Acc: {val_accuracy} |")

				print(similarity)

				log = {
					"train_loss": np.mean(losses),
					"val_loss": val_loss,
					"val_acc": val_accuracy
				}
				
				training_losses.append(np.mean(losses))
				validation_losses.append(val_loss)
				validation_accuracies.append(val_accuracy)

				wandb.log({**similarity,**log})
				
				losses = []
				accuracies = []
				running_sim = []
				
			if n_steps >= n_train_steps:
				break

	state_dict = {"model_state":model.state_dict(),
				"training_losses":training_losses,
				"validation_losses":validation_losses,
				"training_accuracies":training_accuracies,
				"validation_accuracies":validation_accuracies}

	return state_dict

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-b","--batch_size",help="Batch size (time)",type=int)
	parser.add_argument("-d","--hidden_dim",help="Model hidden dimension",type=int)
	parser.add_argument("-t","--training_steps",help="Number of training steps",type=int)
	parser.add_argument("-c","--cpu",help="Train on CPU",action=argparse.BooleanOptionalAction,default=False)
	parser.add_argument("-l","--e_trace_impulse",help="Impulse response for eligibility trace",action=argparse.BooleanOptionalAction,default=False)
	parser.add_argument("-g","--grad_impulse",help="Impulse response for grads",action=argparse.BooleanOptionalAction,default=True)
	parser.add_argument("-n","--ssm_degree",help="Number of hidden variables",type=int)
	parser.add_argument("-s","--sorted",help="Use sorted MNIST",action=argparse.BooleanOptionalAction,default=False)
	parser.add_argument("-f","--save_folder",help="Save folder for group of experiments",type=str)
	parser.add_argument("-r","--run_name",help="Run name on wandb and save folder",type=str)
	parser.add_argument("--delay_factor",help="Delay factor to multiply depth with",type=int)
	parser.add_argument("--sideways",help="Stop approximating delays",action=argparse.BooleanOptionalAction,default=False)
	parser.add_argument("--harmonics",help="Use cosine harmonics for inputs",type=int,default=0)
	parser.add_argument("--binary",help="binarize the activations",type=float)
	parser.add_argument("--threshold",help="threshold activations",type=float)
	parser.add_argument("--weight_decay",help="weight decay",type=float,default=1E-3)
	parser.add_argument("--p_drop",help="dropout post pool",type=float,default=0.0)
	parser.add_argument("--lr",help="learning rate",type=float,default=1E-3)
	parser.add_argument("--dataset",help="pick dataset",type=str,default="mnist")
	parser.add_argument("--use_val",help="Use 10% of training as validation",action=argparse.BooleanOptionalAction,default=False)
	parser.add_argument("--wandb_project",help="Wandb group for logging",type=str,default="slow_fast_rules")
	parser.add_argument("--adam",help="use AdamW instead of SGD for rapid testing",action=argparse.BooleanOptionalAction,default=False)
	parser.add_argument("--n_groups",help="present images by classes in groups",type=int,default=1)
	parser.add_argument("--interp_factor",help="interpolate between images",type=int,default=1)
	parser.add_argument("--top_quantile",help="top quantile loss sparsification",type=float,default=0.0)
	parser.add_argument("--sparse_inputs",help="sparsify inputs as well as gradients",action=argparse.BooleanOptionalAction,default=False)
	parser.add_argument("--cnn",help="Use CNN instead of MLP",action=argparse.BooleanOptionalAction,default=False)
	parser.add_argument("--effective_batch",help="Full kernel length",type=int)
	parser.add_argument("--wandb_mode",help="wandb logging",type=str,default="online")
	parser.add_argument("--norm_op",help="batch/layer/none",type=str,default="batch")
	parser.add_argument("--val_batch",type=int,default=128)
	parser.add_argument("--warmup",type=float,default=0.05)
	parser.add_argument("--track_sim",help="track cosine similarity",action=argparse.BooleanOptionalAction,default=False)

	parsed_args = parser.parse_args()

	save_folder = "results"
	save_folder = os.path.join("results",parsed_args.save_folder)

	if not os.path.isdir(save_folder):
		os.makedirs(save_folder)

	if (parsed_args.binary is not None) & (parsed_args.threshold is not None):
		raise ValueError("Binarize or threshold?")
	
	elif (parsed_args.threshold is not None) & (parsed_args.binary is None):
		input_func = lambda x: thresh(x,parsed_args.threshold)

	elif (parsed_args.threshold is None) & (parsed_args.binary is not None):
		input_func = lambda x: relu_grad(x,parsed_args.binary)

	else:
		input_func = identity

	print(f"using adam: {parsed_args.adam}")

	optim_alg = torch.optim.AdamW if parsed_args.adam else torch.optim.SGD

	if parsed_args.dataset == "mnist":

		transform = transforms.Compose(
		[transforms.ToTensor(),
		transforms.Normalize((0.5,), (0.5,))])

		training_set = torchvision.datasets.MNIST('./data', train=True, transform=transform, download=True)
		validation_set = torchvision.datasets.MNIST('./data', train=False, transform=transform, download=True)

		training_set.data = training_set.data - training_set.data.float().mean()
		training_set.data = training_set.data/training_set.data.float().std()*0.5

		validation_set.data = validation_set.data - validation_set.data.float().mean()
		validation_set.data = validation_set.data/validation_set.data.float().std()*0.5

		training_set.data = training_set.data + 0.5
		validation_set.data = validation_set.data + 0.5
				
		cifar_10 = False

		x_grouped,y_grouped = create_grouped_data(training_set,parsed_args.n_groups)
		
	elif parsed_args.dataset == "cifar10":

		print("Training on CIFAR10")

		transform = transforms.Compose(
			[transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

		std = torch.Tensor([0.2023, 0.1994, 0.2010])
		mean = torch.Tensor([0.4914, 0.4822, 0.4465])

		training_set = torchvision.datasets.CIFAR10(root='./data', train=True,
												download=True, transform=transform)
		validation_set = torchvision.datasets.CIFAR10(root='./data',train=False,download=False,transform=transform)

		training_set.data = torch.Tensor(training_set.data)
		validation_set.data = torch.Tensor(validation_set.data)

		training_set.data = training_set.data - training_set.data.float().mean()
		training_set.data = training_set.data/training_set.data.float().std()*std

		validation_set.data = validation_set.data - validation_set.data.float().mean()
		validation_set.data = validation_set.data/validation_set.data.float().std()*std

		training_set.data = training_set.data + mean
		validation_set.data = validation_set.data + mean

		x_grouped,y_grouped = create_grouped_data(training_set,parsed_args.n_groups)
		
		cifar_10 = True

	if parsed_args.use_val:
		print("Use 10% of training for val")
		perm = torch.randperm(x_grouped.shape[0])
		x_shuffled = x_grouped[perm]
		y_shuffled = y_grouped[perm]

		n_val = int(x_grouped.shape[0]*0.1)

		x_val = x_shuffled[:n_val]
		y_val = y_shuffled[:n_val]

		validation_set.data = einx.rearrange("b g ... -> (b g) ...",x_val)
		validation_set.targets = einx.rearrange("b g ... -> (b g) ...",y_val)

		x_grouped = x_shuffled[n_val:]
		y_grouped = y_shuffled[n_val:]

		print(f"x train shape: {x_grouped.shape}")
		print(f"x test shape: {validation_set.data.shape}")

	e_h = "impulse" if parsed_args.e_trace_impulse else "exponential"
	d_h = "impulse" if parsed_args.grad_impulse else "exponential"

	e_f = perfect_impulse if parsed_args.e_trace_impulse else l1_norm_f_n
	d_f = perfect_impulse if parsed_args.grad_impulse else l1_norm_f_n
	
	device = ("cpu" if parsed_args.cpu else "cuda")
	sorted = "sorted" if parsed_args.sorted else "unsorted"

	run_params = {
		"sorted":parsed_args.sorted,
		"batch_size":parsed_args.batch_size,
		"e_trace_impulse":e_h,
		"delta_trace_impulse":d_h,
		"ssm_degree":parsed_args.ssm_degree,
		"training_steps":parsed_args.training_steps,
		"delay_factor":parsed_args.delay_factor,
		"sideways":parsed_args.sideways,
		"binary":parsed_args.binary,
		"threshold":parsed_args.threshold,
		"lr":parsed_args.lr,
		"use_val":parsed_args.use_val,
		"weight_decay":parsed_args.weight_decay,
		"p_drop":parsed_args.p_drop,
		"dataset":parsed_args.dataset,
		"top_quantile":parsed_args.top_quantile,
		"sparse_inputs":parsed_args.sparse_inputs,
		"norm":parsed_args.norm_op,
		"warmup":parsed_args.warmup,
		"track_sim":parsed_args.track_sim,
	}

	wandb.init(
		project=parsed_args.wandb_project,
		name=parsed_args.run_name,
		config=run_params,
		mode=parsed_args.wandb_mode
	)

	print(run_params)

	print(f"Training model of degree {parsed_args.ssm_degree} with {parsed_args.harmonics} harmonics on {sorted} {parsed_args.dataset} with {parsed_args.delay_factor}X diff using {e_h} for elibility trace and {d_h} for delta on {device}")
	print(f"Training with quantile {parsed_args.top_quantile} | sparse_inputs={parsed_args.sparse_inputs}")
	if parsed_args.sideways:
		print("Not trying to approximate delays")

	if parsed_args.cnn:
		print(f"Using a CNN with norm: {parsed_args.norm_op}")
		model = DelayCNN(
			n_hidden=parsed_args.hidden_dim,
			batch_size=parsed_args.batch_size,
			delta_delay_f=d_f,
			e_trace_delay_f=e_f,
			delay=parsed_args.delay_factor,
			n=parsed_args.ssm_degree,
			store_hebbian=not parsed_args.sideways,
			harmonics=parsed_args.harmonics,
			input_func=input_func,
			cifar_10=cifar_10,
			sparse_inputs=parsed_args.sparse_inputs,
			norm=parsed_args.norm_op,
			pdrop=parsed_args.p_drop,
		)

		prep = lambda x,y: random_prep_cnn(x,y,parsed_args.interp_factor)

		val_prep = vanilla_prep_cnn

		hook_class = CNNHook

	else:
		print("Using a MLP")
		model = DelayMLP(n_hidden=parsed_args.hidden_dim,
					batch_size=parsed_args.batch_size,
					delta_delay_f=d_f,
					e_trace_delay_f=e_f,
					delay=parsed_args.delay_factor,
					n=parsed_args.ssm_degree,
					store_hebbian=not parsed_args.sideways,
					harmonics=parsed_args.harmonics,
					input_func=input_func,
					cifar_10=cifar_10,
					sparse_inputs=parsed_args.sparse_inputs)

		prep = lambda x,y: grouped_inerpolated_prep(x,y,parsed_args.interp_factor)

		val_prep = vanilla_prep

		hook_class = MLPHook

	states = train(model=model,
				hook_class=hook_class,
				x_train=x_grouped,
				y_train=y_grouped,
				x_val=torch.Tensor(validation_set.data),
				y_val=torch.Tensor(validation_set.targets),
				batch_size=parsed_args.batch_size,
				val_batch_size=parsed_args.val_batch,
				n_groups=parsed_args.n_groups,
				interp_factor=parsed_args.interp_factor,
				n_train_steps=parsed_args.training_steps,
				val_steps=390,
				device=device,
				val_prep_batch=val_prep,
				train_prep_batch=prep,
				lr=parsed_args.lr,
				weight_decay=parsed_args.weight_decay,
				optim_alg=optim_alg,
				top_quantile=parsed_args.top_quantile,
				warmup_p=parsed_args.warmup,
				delay=parsed_args.delay_factor,
				track_sim=parsed_args.track_sim)
	
	states["run_params"] = run_params
	
	
	current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

	filename = f"{parsed_args.run_name}.pt"

	torch.save(states, os.path.join(save_folder, filename))