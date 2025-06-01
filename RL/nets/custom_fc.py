import torch.nn as nn
import torch
import numpy as np
import scipy
from scipy.linalg import expm
from collections import deque
import copy

from nets.utils import sparse_init


class CustomLinearET(nn.Module):
    def __init__(self, input_dim, out_dim, args):
        super().__init__()
        self.args = copy.deepcopy(args)
        self.fc = self.layer_init(nn.Linear(input_dim, out_dim, bias=False))
        self.last_input = None
        self.last_output = None
        self.grad_weights = 0
        self.gamma = args.gamma
        if args.gae_lambda_random:
            self.register_buffer(
                "gae_lambda",
                torch.tensor(np.random.uniform(0.1, 0.99, input_dim*out_dim).reshape(1, out_dim, input_dim), dtype=torch.float32)
            )
        elif args.gae_lambda == -1:
            self.register_buffer(
                "gae_lambda",
                torch.tensor(np.random.uniform(0.1, 0.99, input_dim).reshape(1, input_dim), dtype=torch.float32)
            )
        else:
            self.gae_lambda = args.gae_lambda
        self.et = 0

    def layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        if self.args.init == 'ortogonal':
            torch.nn.init.orthogonal_(layer.weight, std)
        elif self.args.init == 'kainming':
            torch.nn.init.kaiming_normal_(layer.weight)
        elif self.args.init == 'sparse':
            print('sparse init!')
            sparse_init(layer.weight, sparsity=0.9)

        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def forward(self, x):
        self.last_input = x
        out = self.fc(x)
        if self.training:
            torch.Tensor.retain_grad(out.requires_grad_(True))
            self.last_output = out
        return out

    def backward(self, ):
        weight_grad = self.last_output.grad.unsqueeze(-1) * self.last_input.unsqueeze(1)
        self.et = weight_grad + self.gae_lambda * self.gamma * self.et
        self.last_output = self.last_output.detach()

    def reset_weights(self):
        self.grad_weights = 0

    def reset(self):
        self.et = 0
        self.grad_weights = 0
        self.last_input = None
        self.last_output = None

    def backward_weights(self, reward, nonterminal):
        self.grad_weights = (-reward.unsqueeze(-1).unsqueeze(-1) * self.et).sum(0) + self.grad_weights
        self.et = nonterminal.unsqueeze(-1).unsqueeze(-1) * self.et

    def set_weights(self):
        self.fc.weight.grad = self.grad_weights

    def reset_et(self):
        self.et = 0


class CustomLinearETNoisy(CustomLinearET):
    def __init__(self, input_dim, out_dim, args, act=True, layer_number=None, name='ff'):
        super().__init__(input_dim, out_dim, args)
        self.queue = deque(maxlen=args.buffer_size)
        self.queue_out = deque(maxlen=args.buffer_size)
        self.queue_out_grad = deque(maxlen=args.buffer_size)
        self.queue_out_approx_grad = deque(maxlen=args.buffer_size)
        self.queue_activation_for_weight_upd = deque(maxlen=args.buffer_size)
        self.impulses = []
        self.impulse_responses = []
        self.delay = args.delay
        self.local_et = 0
        self.local_th_et = 0
        self.et_activation = 0
        self.act = act
        self.correction_count = 0
        self.correction_et_activation = 0
        self.correction_local_et = 0
        self.layer_number = layer_number
        self.running_mean = 0
        self.x_tilda = 0
        self.wm_samples = 0
        self.weight_grad_cos_similarity = 0
        self.out_grad_cos_similarity = 0
        self.activation_similarity = 0
        self.grad_activation_similarity = 0
        self.local_et_similarity = 0
        self.weight_grad_norm = 0
        self.out_grad_norm = 0
        self.name = name

        self.activation_ratio = 0
        self.et_activation_ratio = 0
        self.local_et_ratio = 0
        self.intersection_ratio = 0
        self.current_global_adv = 0
        self.adv_threshold = self.args.adv_threshold
        self.running_avr_thresholding = 0.95

        if self.args.constant_delay:
            idx = -self.delay
            self.layer_idx = idx
            self.reward_idx = idx
            self.T_delay = T_delay = self.delay
        else:
            if self.args.last_layer_no_delay:
                idx = -self.delay * (self.layer_number - 1) - 1
                self.layer_idx = idx
                self.reward_idx = idx
                self.T_delay = T_delay= self.delay * (self.layer_number - 1)
            else:
                idx = -self.delay * self.layer_number - 1
                self.layer_idx = idx
                self.reward_idx = idx
                self.T_delay = T_delay = self.delay * self.layer_number

            # if no delay in the last layer is toggled, we need to use vanilla gradient
            if self.args.last_layer_no_delay and self.layer_number == 1:
                self.args.top_grad = 'vanilla'
                self.args.backward_weights = 'vanilla'

        print('Layer number', self.layer_number, 'Layer idx', self.layer_idx, 'Reward idx', self.reward_idx, 'T_delay', T_delay)
        print("Using beta = ", args.beta)
        self.beta = args.beta

        if args.running_average_beta:
            self.c = 1 - self.beta
        else:
            self.c = 1

        if args.threshold >= 0:
            self.threshold = args.threshold
        elif args.threshold == -1:
            print('Using threshold = beta ** delay')
            self.register_buffer(
                "threshold",
                torch.tensor(self.beta ** self.delay - 1e-8)
            )

        self.alpha_ssm = args.alpha_ssm

        if self.args.backward_weights == 'SSM' or self.args.backward_weights == 'SSM_threshold':
            if args.ssm_cascade_size == 1:
                print('Using et_weights instead of SSM cascade as ssm_cascade_size = 1')
                self.args.backward_weights = 'et_weights'
                self.norm_scaler = 1
                if args.normalized_by_max_value:
                    self.norm_scaler = 1 / (self.beta ** T_delay)

            else:
                def build_cascade_matrix(alpha, n):
                    A = -alpha * np.eye(n)
                    A[np.arange(1, n), np.arange(0, n - 1)] = 1
                    return A

                def closed_form_integral_col0(A):
                    dif = expm(A) - np.eye(A.shape[0])
                    res = np.linalg.solve(A.T, dif.T).T  # solves Ax=B => A^(-1) B, but we need B A^(-1)
                    return res[0]

                alpha = (args.ssm_cascade_size - 1) / T_delay
                A = build_cascade_matrix(alpha, args.ssm_cascade_size)
                expA = expm(A)
                col0 = closed_form_integral_col0(A)

                self.register_buffer("expA", torch.tensor(expA, dtype=torch.float32).unsqueeze(0).unsqueeze(0).unsqueeze(0))
                self.register_buffer("col0", torch.tensor(col0, dtype=torch.float32).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(-1))
                self.register_buffer(
                        "ssm_cascade", torch.tensor(np.zeros((args.batch_size, out_dim, input_dim, args.ssm_cascade_size)), dtype=torch.float32)
                    )

                mass_inverse, max_value = self.get_normalization_scaler(expA, col0, T_delay)

                self.threshold = max_value - 1e-4
                if not self.args.no_mass_inverse:
                    self.norm_scaler = mass_inverse * args.ssm_scaler
                else:
                    self.norm_scaler = args.ssm_scaler

                if args.normalized_by_max_value:
                    self.norm_scaler = self.norm_scaler * 1 / max_value
                    self.threshold = 1 - 1e-4

            print('Using norm_scaler = ', self.norm_scaler)

        print('Using threshold = ', self.threshold)

    def ssm_update(self, x_t, b):
        term1 = self.expA @ x_t.unsqueeze(-1)
        term2 = self.col0 * b.unsqueeze(-1).unsqueeze(-1)
        out = term1 + term2
        return out.squeeze(-1)

    def get_normalization_scaler(self, expA, col0, T):

        def f(t):
            res = t < 1
            return res

        l = max(int(5 * T), 100)
        t_grid = np.linspace(0, l,  l + 1)
        ssm_output = list()

        for t in t_grid:
            self.ssm_cascade = self.ssm_update(self.ssm_cascade, torch.ones(self.ssm_cascade.shape[:3]) * self.alpha_ssm * f(t))
            ssm_output.append(self.ssm_cascade[0, 0, 0, -1].item())

        print('argmax', np.argmax(ssm_output), t_grid[np.argmax(ssm_output)])

        max_idx = np.argmax(ssm_output)
        max_value = ssm_output[max_idx]
        inverse_sum = 1 / sum(ssm_output)
        print('max value at', max_idx)
        print('max_value', max_value)
        print('sum', sum(ssm_output))

        self.ssm_cascade = torch.zeros_like(self.ssm_cascade)

        return inverse_sum, max_value * inverse_sum

    def forward(self, x):
        out = self.fc(x)

        if self.training:

            self.last_input = x
            torch.Tensor.retain_grad(out.requires_grad_(True))
            self.last_output = out
            if len(self.queue) == 0:
                for _ in range(self.args.buffer_size):
                    self.queue.append(x)
            else:
                self.queue.append(x)

            if np.random.rand() < 0.001:
                self.compute_activation_stats()

        return out

    def backward(self, grad):

        if len(self.queue_out_approx_grad) == 0:
            for _ in range(self.args.buffer_size):
                self.queue_out_approx_grad.append(grad)
        else:
            self.queue_out_approx_grad.append(grad)

        # store gradient into the buffer to compute stats later or to compute vanilla gradient
        last_grad = self.last_output.grad.detach()
        if len(self.queue_out_grad) == 0:
            for _ in range(self.args.buffer_size):
                self.queue_out_grad.append(last_grad)
        else:
            self.queue_out_grad.append(last_grad)

        # replace grad with vanilla one if needed
        if self.args.top_grad == 'vanilla':
            grad = self.last_output.grad
        elif self.args.top_grad == 'delayed_vanilla':
            grad = self.queue_out_grad[self.layer_idx]
        else:
            grad = self.queue_out_approx_grad[-self.delay]

        last_out = (self.last_output > 0)

        if len(self.queue_out) == 0:
            for _ in range(self.args.buffer_size):
                self.queue_out.append(last_out)

        self.queue_out.append(last_out)

        exact_coarsen_activations = (self.queue[self.layer_idx].unsqueeze(1) > 0).float()

        if self.act:
            local_et = last_out.unsqueeze(-1) * self.last_input.unsqueeze(1)
        else:
            local_et = torch.ones_like(self.last_output).unsqueeze(-1) * self.last_input.unsqueeze(1)

        if self.args.adv_threshold > 0:
            if self.args.adv_threadhold_reverse is not True:
                threshold = torch.abs(self.current_global_adv) > self.adv_threshold
                self.running_avr_thresholding = 0.99 * self.running_avr_thresholding + 0.01 * threshold.float().mean()
                if self.running_avr_thresholding > self.args.running_avr_thresholding:
                    self.adv_threshold = 1.001 * self.adv_threshold
                elif self.running_avr_thresholding < self.args.running_avr_thresholding-0.05:
                    self.adv_threshold = 0.999 * self.adv_threshold
                local_et = local_et * threshold.unsqueeze(-1).unsqueeze(-1)
            else:
                threshold = torch.abs(self.current_global_adv) < self.adv_threshold
                self.running_avr_thresholding = 0.99 * self.running_avr_thresholding + 0.01 * threshold.float().mean()
                if self.running_avr_thresholding > self.args.running_avr_thresholding:
                    self.adv_threshold = 0.999 * self.adv_threshold
                elif self.running_avr_thresholding < self.args.running_avr_thresholding-0.05:
                    self.adv_threshold = 1.001 * self.adv_threshold
                local_et = local_et * threshold.unsqueeze(-1).unsqueeze(-1)

        if self.args.skip_n_activations > 0:
            local_et = local_et * self.current_global_adv.unsqueeze(-1).unsqueeze(-1)

        if self.args.activation_threshold > 0:
            local_et = local_et * (local_et > self.args.activation_threshold).float()

        if self.args.clamp_activation_for_backward:
            local_et = torch.clamp(local_et, -1, 1)

        # backward weight computation
        if self.args.backward_weights == 'et_weights' or self.args.backward_weights == 'et_weights_threshold':
            self.local_et = self.beta * self.local_et + self.c * local_et
            self.local_et = self.norm_scaler * self.local_et
            if self.args.backward_weights == 'et_weights_threshold':
                self.local_et = self.local_et * (self.local_et > self.threshold).float()

            activation_for_weight_upd = self.local_et

        elif self.args.backward_weights == 'SSM' or self.args.backward_weights == 'SSM_threshold':
            self.ssm_cascade = self.ssm_update(self.ssm_cascade, self.alpha_ssm * local_et).detach()
            activation_for_weight_upd = self.norm_scaler * self.ssm_cascade[:, :, :, -1].clone()
            if self.args.backward_weights == 'SSM_threshold':
                activation_for_weight_upd = activation_for_weight_upd * (activation_for_weight_upd > self.threshold)

        elif self.args.backward_weights == 'vanilla':
            activation_for_weight_upd = self.last_input.unsqueeze(1)

        elif self.args.backward_weights == 'delayed_vanilla':
            activation_for_weight_upd = self.queue[self.layer_idx].unsqueeze(1)

        if self.args.collect_impulse_responses:
            self.queue_activation_for_weight_upd.append(activation_for_weight_upd.clone().detach())

        weight_grad = grad.unsqueeze(-1) * activation_for_weight_upd

        # RL eligibility traces update
        if self.args.et_strategy == "accumulate":
            self.et = weight_grad + self.gae_lambda * self.gamma * self.et
        elif self.args.et_strategy == "vanilla":
            self.et = weight_grad
        else:
            assert False, "Unknown et strategy"

        self.et.detach_()

        self.last_output = self.last_output.detach()

        # dummy backward pass for the previous layer
        out_grad = torch.ones_like(self.last_input)

        # compute statistic for debugging
        self.compute_stats(activation_for_weight_upd, weight_grad, grad, out_grad)

        return out_grad

    def compute_stats(self, activation_for_weight_upd, weight_grad, grad, out_grad):

        if self.args.collect_impulse_responses:
            if np.random.rand() < 0.005:
                if len(self.queue_activation_for_weight_upd) == self.args.buffer_size:
                    self.impulses = []
                    self.impulse_responses = []
                    j, k = np.random.randint(0, self.fc.weight.size(0)), np.random.randint(0, self.fc.weight.size(1))
                    for i in range(len(self.queue_activation_for_weight_upd) - self.delay * self.layer_number):
                        if self.act:
                            et_layer = (self.queue_out[i][0][j] > 0).float() * self.queue[i][0][k]
                        else:
                            et_layer = self.queue[i][0][k]
                        delayed_i = i + self.delay * self.layer_number
                        self.impulses.append(et_layer)
                        self.impulse_responses.append((self.queue_activation_for_weight_upd[delayed_i][0][j][k]).item())

        if np.random.rand() < 0.005:
            # weight_grad stats
            if self.args.delay_grad:
                in_true_grad = self.queue_out_grad[self.layer_idx]
                true_act = self.queue[self.layer_idx]
            else:
                in_true_grad = self.queue_out_grad[-1]
                true_act = self.last_input

            if self.step > 10:
                if self.act:
                    et_layer = (self.queue_out[self.layer_idx].unsqueeze(-1) > 0).float() * self.queue[self.layer_idx].unsqueeze(1)
                else:
                    et_layer = torch.ones_like(self.queue_out[self.layer_idx]).unsqueeze(-1) * self.queue[self.layer_idx].unsqueeze(1)
                self.local_et_similarity = et_sim = (activation_for_weight_upd * et_layer).sum() / (activation_for_weight_upd.norm() * et_layer.norm())
                print("Step", self.step, "Layer ",  self.layer_number, self.name, "ET similarity: ", et_sim)

            weight_true_grad = in_true_grad.unsqueeze(-1) * true_act.unsqueeze(1)
            self.weight_grad_cos_similarity = weight_grad_similarity = (weight_grad * weight_true_grad).sum() / (weight_grad.norm() * weight_true_grad.norm())
            self.weight_grad_norm = weight_grad.norm()
            print("Step", self.step, "Layer ",  self.layer_number, self.name, "Weight grad cosine similarity: ", weight_grad_similarity)

            self.grad_activation_similarity = grad_sim = ((in_true_grad != 0) * (grad != 0)).sum() / (in_true_grad != 0).sum()
            print("Step", self.step, "Layer ",  self.layer_number, self.name, 'grad_sim', grad_sim)

            self.activation_similarity = act_sim = ((true_act > 0) * (activation_for_weight_upd.sum(1) > 0)).sum() / (true_act > 0).sum()
            print("Step", self.step, "Layer ",  self.layer_number, self.name, 'act_sim', act_sim)

            # out_grad stats
            true_out_grad = in_true_grad.mm(self.fc.weight) * (true_act > 0).float()
            self.out_grad_cos_similarity = out_grad_similarity = (out_grad * true_out_grad).sum() / (out_grad.norm() * true_out_grad.norm())
            self.out_grad_norm = out_grad.norm()
            print("Step", self.step, "Layer ",  self.layer_number, self.name, "Out grad cosine similarity: ", out_grad_similarity)

    def set_step(self, step):
        self.step = step

    def reset(self):
        self.et = 0
        self.grad_weights = 0
        self.et_activation = 0
        self.local_et = 0
        self.local_th_et = 0
        self.last_input = None
        self.last_output = None
        self.queue.clear()
        self.queue_out_grad.clear()
        self.queue_out.clear()

    def reset_et(self):
        self.et = 0

    def compute_activation_stats(self):
        self.activation_ratio = self.compute_activation_ratio()
        self.et_activation_ratio = self.compute_et_activation_ratio()
        self.local_et_ratio = self.compute_local_et_ratio()
        self.intersection_ratio = self.compute_intersection_ratio()

    def compute_activation_ratio(self):
        try:
            ratio = (self.last_input > 0).sum() / self.last_input.numel()
            return ratio
        except:
            return 0

    def compute_et_activation_ratio(self):
        try:
            ratio = (self.et_activation > 0).sum() / self.et_activation.numel()
            return ratio
        except:
            return 0

    def compute_local_et_ratio(self):
        try:
            ratio = (self.local_et > 0).sum() / self.local_et.numel()
            return ratio
        except:
            return 0

    def compute_intersection_ratio(self):
        try:
            ratio = ((self.queue[-1] > 0) * (self.queue[-2] > 0)).sum() /(self.queue[-1] > 0).sum()
            return ratio
        except:
            return 0
