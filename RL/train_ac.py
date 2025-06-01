import os
import random
import time
from dataclasses import dataclass
import json

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from utils import make_env_discrete
import nets.ac_nets as nets
from trainers.trainers_ac import train_et_continual


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "CETs"
    """the wandb's project name"""
    wandb_entity: str = ''
    """the entity (team) of wandb's project"""
    wandb_log_dir: str = "."
    """the directory to save the wandb logs"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "MinAtar/SpaceInvaders-v0"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the actor"""
    num_envs: int = 32
    """the number of parallel game environments"""
    num_steps: int = 32
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    gae_lambda_random: bool = False
    """randomize the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: int = 3
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    trainer:str = 'et_continual'
    N_hidden_layers: int = 1
    """the number of hidden layers in the policy and value networks"""
    agent: str = 'AgentActorETNoisy'
    """the agent class to use"""
    save_model: bool = False
    """save the model"""
    eval_model: bool = True
    """evaluate the model"""
    history_states: int = 1
    """number of history states to consider"""
    num_last_actions: int = 0
    """number of last actions to consider"""
    actor_hidden_dim: int = 256
    """the hidden dimension of the actor network in case of MLP actor"""
    beta: float = 0.9
    """beta for the eligibility traces"""
    running_average_beta: bool = False
    wd: float = 0. #0.00001
    """weight decay"""
    log_interval: int = 100
    """log interval"""
    episodic_return_threshold: int = 10000
    """episodic return threshold"""
    backward_weights: str = 'vanilla'
    """how to compute weights gradient"""
    backward: str = 'vanilla'
    """how to compute backward pass for the next layer """
    top_grad: str = 'grad'
    """top gradient to use for the backward pass"""
    delay_grad: bool = False
    """in-parallel gradient update"""
    delay: int = 1
    """how many steps it takes a gradient to go through one layer"""
    init: str = 'ortogonal'
    """initialization of the weights in actor"""
    et_strategy: str = None
    """accumulate eligibility traces in backward pass with approximate gradient"""
    threshold: float = 0.
    """threshold for the activation to compute backward pass"""
    buffer_size: int = 100
    """buffer size for to store the activations and gradients"""
    ssm_scaler: float = 1.
    """scale for the ssm impulse response"""
    ssm_cascade_size: int = 1
    """cascade size for the ssm impulse response"""
    normalized_by_max_value: bool = False
    """normalize the SSM coefficient by having max response equal to 1"""
    collect_impulse_responses: bool = False
    """whether to collect impulse responses"""
    collect_grad_stats: bool = True
    """collect gradient statistics"""
    train_only_last_layer: bool = False
    """train only the last layer"""
    train_only_last_two: bool = False
    """train only the last two layers"""
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    adv_threshold: float = -1
    """initial advantage threshold to zero out activations"""
    running_avr_thresholding: float = 0.95
    """desired running average of zeroouted activations"""
    adv_threadhold_reverse: bool = False
    """reverse the > threshold to < threshold"""
    warmup_iters: int = 0
    """number of warmup iterations for learning rate"""
    clamp_adv: float = 0
    """clamp the advantage"""
    skip_n_activations: int = 0
    """skip n activations"""
    activation_threshold: float = 0
    """threshold for the activations"""
    clamp_activation_for_backward: bool = False
    """clamp the activation for the backward pass"""
    positive_obs: bool = False
    """make observation positive by dobling observation dimention"""
    alpha_ssm: float = 1.
    """alpha for the ssm impulse response"""
    last_layer_no_delay: bool = False
    """no delay for the last layer"""
    time_per_state: float = 0.2
    """time per env state transition, used to set env in CartPole and LunarLander"""
    constant_delay: bool = False
    """use the same delay for all layer layers"""
    no_mass_inverse: bool = False
    """do not use mass inverse for ssm cascade"""


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.mininum_steps_size = int(args.num_steps // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    print('\n args.batch_size', args.batch_size, 'args.minibatch_size', args.minibatch_size,
          'args.mininum_steps_size', args.mininum_steps_size, 'args.num_iterations', args.num_iterations, '\n')
    group_name = f"{args.agent}_{args.exp_name}"
    run_name = f"{group_name}_s{args.seed}_{int(time.time())}"
    args.buffer_size = args.delay * (args.N_hidden_layers + 2 - int(args.last_layer_no_delay)) + 1
    print('args.buffer_size', args.buffer_size)
    args.group_name = group_name
    args.run_name = run_name
    args.randint = random.randint(0, 1000)
    args.out_dir = f"runs/{args.env_id}/{args.group_name}/{run_name}_randint{args.randint}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            group=group_name,
            name=run_name,
            monitor_gym=True,
            save_code=True,
            dir=args.wandb_log_dir,
        )
    writer = SummaryWriter(args.out_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    with open(f'{args.out_dir}/args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    args.device = device

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env_discrete(args.env_id, i, args.capture_video, run_name, args) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # filling in missing arguments from environment
    if 'MinAtar' in args.env_id or 'MiniGrid' in args.env_id:
        args.input_channels = envs.single_observation_space.shape[-1]
    else:
        args.input_channels = envs.single_observation_space.shape[0]

    args.obs_shape = envs.single_observation_space.shape
    args.batch_size = args.num_envs

    Agent = getattr(nets, args.agent)
    agent = Agent(envs, args).to(device)
    print(agent)
    if args.train_only_last_layer:
        actor_params = agent.actor[-1].parameters()
    elif args.train_only_last_two:
        actor_params = list(agent.actor[-1].parameters()) + list(agent.actor[-2].parameters())
    else:
        actor_params = agent.actor.parameters()
    optimizer = optim.Adam(
        [
        {'params': agent.critic.parameters(), 'lr': args.learning_rate, 'eps': 1e-5, 'weight_decay': args.wd},
        {'params': actor_params, 'lr': args.learning_rate, 'eps': 1e-5, 'weight_decay': args.wd}
        ],
    )

    train_et_continual(args, optimizer, device, envs, agent, writer)

    envs.close()
    writer.close()