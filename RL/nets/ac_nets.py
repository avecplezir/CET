import torch.nn as nn
import torch
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
import numpy as np
from collections import deque

from nets.utils import layer_init, transform_obs
from nets.custom_fc import CustomLinearET, CustomLinearETNoisy


def get_critics(envs, args):
    if args.env_id in ['CartPole-v1', 'LunarLander-v2', 'Acrobot-v1', 'MountainCar-v0']:
        return nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), args.actor_hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(args.actor_hidden_dim, args.actor_hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(args.actor_hidden_dim, 1), std=1.0),
        )
    else:
        return nn.Sequential(
            layer_init(nn.Conv2d(envs.single_observation_space.shape[-1], 16, 3, padding='same')),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, 3, padding='same')),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 32, 3, padding='same')),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(3200, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 1), std=1)
        )

class AgentActorMLP(nn.Module):
    def __init__(self, envs, args):
        super().__init__()

        self.args = args
        obs_shape = envs.single_observation_space.shape
        self.input_dim = np.prod(obs_shape)
        print('input_dim', self.input_dim)
        self.critic = get_critics(envs, args)

        actor_layers = []
        actor_layers.extend([layer_init(nn.Linear(self.input_dim, args.actor_hidden_dim, bias=False)), nn.ReLU()])
        for _ in range(args.N_hidden_layers):
            actor_layers.extend([layer_init(nn.Linear(args.actor_hidden_dim, args.actor_hidden_dim, bias=False)), nn.ReLU()])
        actor_layers.append(layer_init(nn.Linear(args.actor_hidden_dim, envs.single_action_space.n, bias=False)))
        self.actor = nn.Sequential(*actor_layers)

    def get_value(self, x):
        x = transform_obs(x, self.args.env_id)
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        x = transform_obs(x, self.args.env_id)
        logits = self.actor(x.reshape(x.size(0), -1))
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


class AgentActorMLPEligibilityTraces(nn.Module):
    def __init__(self, envs, args):
        super().__init__()

        self.args = args
        obs_shape = envs.single_observation_space.shape
        self.input_dim = np.prod(obs_shape)
        self.critic = get_critics(envs, args)

        actor_layers = []
        actor_layers.extend([CustomLinearET(self.input_dim, args.actor_hidden_dim, args)])
        for _ in range(args.N_hidden_layers):
            actor_layers.extend([CustomLinearET(args.actor_hidden_dim, args.actor_hidden_dim, args)])
        actor_layers.append(CustomLinearET(args.actor_hidden_dim, envs.single_action_space.n, args))
        self.actor = nn.ModuleList(actor_layers)


    def get_value(self, x):
        x = transform_obs(x, self.args.env_id)
        return self.critic(x)

    def set_step(self, *args):
        pass

    def get_action_and_value(self, x, action=None):

        obs = transform_obs(x, self.args.env_id)
        x = obs.reshape(obs.size(0), -1)

        for block in self.actor[:-2]:
            x = block(x)
            x = F.relu(x)

        x = self.actor[-2](x)
        x = F.relu(x)
        logits = self.actor[-1](x)

        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(obs)

    def reset(self):
        for block in self.actor:
            block.reset()

    def backward_weights(self, reward, dones):
        nonterminal = 1 - dones
        for block in self.actor:
            block.backward_weights(reward, nonterminal)

    def backward(self, grads=None):
        for block in self.actor:
            block.backward()

    def set_weights(self):
        for block in self.actor:
            block.set_weights()


class AgentActorETNoisy(AgentActorMLPEligibilityTraces):
    def __init__(self, envs, args):
        nn.Module.__init__(self)

        self.args = args
        obs_shape = envs.single_observation_space.shape
        self.input_dim = np.prod(obs_shape)
        self.critic = get_critics(envs, args)

        actor_layers = []
        N_layers = args.N_hidden_layers + 2
        actor_layers.extend([CustomLinearETNoisy(self.input_dim, args.actor_hidden_dim, args, layer_number=N_layers)])
        N_layers -= 1
        for _ in range(args.N_hidden_layers):
            actor_layers.extend([CustomLinearETNoisy(args.actor_hidden_dim, args.actor_hidden_dim, args, layer_number=N_layers)])
            N_layers -= 1
        print('last layer number:', N_layers)
        actor_layers.append(CustomLinearETNoisy(args.actor_hidden_dim, envs.single_action_space.n, args, act=False, layer_number=N_layers))

        self.actor = nn.ModuleList(actor_layers)
        self.queue_reward = deque(maxlen=args.buffer_size)
        self.queue_nonterminal = deque(maxlen=args.buffer_size)
        self.queue_last_grad = deque(maxlen=args.buffer_size)
        self.iteration = 0

    def backward(self, grads=None, adv=None):
        self.iteration += 1

        if self.args.adv_threshold > 0:
            for block in self.actor:
                block.current_global_adv = adv

        if self.args.skip_n_activations > 0:
            for block in self.actor:
                skipping = (self.iteration % self.args.skip_n_activations == 0) * torch.ones_like(adv)
                block.current_global_adv = skipping

        if self.args.delay_grad:
            if grads is None:
                new_grads = []
                grad = self.actor[-1].last_output.grad.clone()
                for l_idx, block in enumerate(self.actor[::-1]):
                    new_grads.append(grad)
                    grad = block.backward(grad)
                    block.reset_et()
                return new_grads
            else:
                new_grads = []
                new_grad = self.actor[-1].last_output.grad
                for l_idx, (block, grad) in enumerate(zip(self.actor[::-1], grads)):
                    new_grads.append(new_grad)
                    new_grad = block.backward(grad)

                return new_grads

        else:
            grad = self.actor[-1].last_output.grad
            for l_idx, block in enumerate(self.actor[::-1]):
                grad = block.backward(grad)

    def backward_weights(self, reward, dones):
        if len(self.queue_reward) == 0:
            for _ in range(self.args.buffer_size):
                self.queue_reward.append(reward)
                self.queue_nonterminal.append(1 - dones)
        else:
            self.queue_reward.append(reward)
            self.queue_nonterminal.append(1 - dones)
        if self.args.delay_grad:
            for block in self.actor:
                block.backward_weights(self.queue_reward[block.reward_idx], self.queue_nonterminal[block.reward_idx])
        else:
            nonterminal = 1 - dones
            for block in self.actor:
                block.backward_weights(reward, nonterminal)

    def reset(self):
        for block in self.actor:
            block.reset()
        self.queue_reward.clear()
        self.queue_nonterminal.clear()

    def reset_weights(self):
        for block in self.actor:
            block.reset_weights()

    def set_step(self, step):
        for block in self.actor:
            block.set_step(step)

    def reset_et(self):
        for block in self.actor:
            block.reset_et()


