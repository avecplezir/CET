import gymnasium as gym
from collections import deque
import numpy as np
from gymnasium.spaces import Box
import torch


class BucketObsWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_buckets=10, step=1):
        super().__init__(env)

        self.buckets = np.arange(n_buckets, step=step)
        print('self.buckets', self.buckets)
        low, high = env.observation_space.low, env.observation_space.high
        obs_low = list(np.array(low).flatten()) * len(self.buckets)
        obs_high = list(np.array(high).flatten()) * len(self.buckets)
        self.observation_space = Box(low=np.array(obs_low), high=np.array(obs_high))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def observation(self, obs):
        new_obs = []
        for i in range(len(self.buckets)-1):
            mask = (self.buckets[i] <= obs) * (obs < self.buckets[i+1])
            new_obs.append(mask*(obs-self.buckets[i]))
        mask = (obs >= self.buckets[-1])
        new_obs.append(mask * (obs-self.buckets[-1]))
        obs = np.concatenate(new_obs, axis=0)
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(obs), reward, terminated, truncated, info


class PositiveObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        low, high = env.observation_space.low, env.observation_space.high
        obs_low = list(np.array(low).flatten())
        obs_high = list(np.array(high).flatten())
        low = np.array(obs_low + obs_low)
        high = np.array(obs_high + obs_high)
        self.observation_space = Box(low=low, high=high)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def observation(self, obs):
        mask = (obs > 0)
        obs = np.concatenate([mask*obs, (mask-1)*obs])
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(obs), reward, terminated, truncated, info


class LastActionWrapper(gym.Wrapper):
    def __init__(self, env, num_last_actions=1):
        super().__init__(env)

        self.num_last_actions = num_last_actions
        self.last_actions = deque(maxlen=num_last_actions)

        low, high = env.observation_space.low, env.observation_space.high
        obs_low = list(np.array(low).flatten())
        obs_high = list(np.array(high).flatten())

        action_low = list(np.array([env.action_space.low for _ in range(num_last_actions)]).squeeze().flatten())
        action_high = list(np.array([env.action_space.high for _ in range(num_last_actions)]).squeeze().flatten())

        low = np.array(obs_low + action_low)
        high = np.array(obs_high + action_high)
        self.observation_space = Box(low=low, high=high)

    def observation(self, obs):
        last_actions = np.concatenate(list(self.last_actions), axis=0)
        new_obs = np.concatenate([obs, last_actions])
        return new_obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if 'initial_observation' not in info:
            info['initial_observation'] = obs
        [self.last_actions.append(self.action_space.sample() * 0) for _ in range(self.num_last_actions)]
        return self.observation(obs), info

    def step(self, action):
        self.last_actions.append(action)
        obs, reward, terminated, truncated, info = self.env.step(action)
        if 'initial_observation' not in info:
            info['initial_observation'] = obs
        return self.observation(obs), reward, terminated, truncated, info


class FrameConcat(gym.ObservationWrapper):
    def __init__(
        self,
        env: gym.Env,
        num_concat: int,
    ):
        gym.ObservationWrapper.__init__(self, env)

        self.num_concat = num_concat
        self.frames = deque(maxlen=num_concat)

        low, high = env.observation_space.low, env.observation_space.high
        low = np.array([[low] * num_concat]).squeeze().flatten()
        high = np.array([[high] * num_concat]).squeeze().flatten()

        self.observation_space = Box(low=low, high=high)

    def observation(self, ):
        return np.concatenate(list(self.frames), axis=0)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if 'initial_observation' not in info:
            info['initial_observation'] = obs
        self.frames.append(obs)
        return self.observation(), reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if 'initial_observation' not in info:
            info['initial_observation'] = obs
        [self.frames.append(obs) for _ in range(self.num_concat)]
        return self.observation(), info


class LastActionWrapperMinAtar(gym.Wrapper):
    def __init__(self, env, num_last_actions=1):
        super().__init__(env)

        self.num_last_actions = num_last_actions
        self.last_actions = deque(maxlen=num_last_actions)

        obs_low, obs_high = env.observation_space.low, env.observation_space.high
        self.n_actions = n_actions = env.action_space.n
        obs_shape = env.observation_space.shape
        action_low = np.zeros((obs_shape[0], obs_shape[1], n_actions*num_last_actions))
        action_high = np.ones((obs_shape[0], obs_shape[1], n_actions*num_last_actions))

        low = np.concatenate([obs_low, action_low], axis=-1)
        high = np.concatenate([obs_high, action_high], axis=-1)
        self.observation_space = Box(low=low, high=high)

    def observation(self, obs):
        last_actions = np.concatenate(list(self.last_actions), axis=0)
        last_actions = np.eye(self.n_actions)[last_actions]
        last_actions = np.tile(last_actions.flatten(), (obs.shape[0], obs.shape[1], 1))
        new_obs = np.concatenate([obs, last_actions], axis=-1)
        return new_obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if 'initial_observation' not in info:
            info['initial_observation'] = obs
        [self.last_actions.append(self.action_space.sample()[None] * 0) for _ in range(self.num_last_actions)]
        return self.observation(obs), info

    def step(self, action):
        self.last_actions.append(action[None])
        obs, reward, terminated, truncated, info = self.env.step(action)
        if 'initial_observation' not in info:
            info['initial_observation'] = obs
        return self.observation(obs), reward, terminated, truncated, info


class FrameConcatMinAtar(gym.ObservationWrapper):
    def __init__(
        self,
        env: gym.Env,
        num_concat: int,
    ):
        gym.ObservationWrapper.__init__(self, env)

        self.num_concat = num_concat
        self.frames = deque(maxlen=num_concat)

        low, high = env.observation_space.low, env.observation_space.high
        low = np.concatenate([low] * num_concat, axis=-1)
        high = np.concatenate([high] * num_concat, axis=-1)
        self.observation_space = Box(low=low, high=high)

    def observation(self, ):
        return np.concatenate(list(self.frames), axis=-1)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if 'initial_observation' not in info:
            info['initial_observation'] = obs
        self.frames.append(obs)
        return self.observation(), reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if 'initial_observation' not in info:
            info['initial_observation'] = obs
        [self.frames.append(obs) for _ in range(self.num_concat)]
        return self.observation(), info


def atari_make_env(env_id, idx, capture_video, run_name, args):
    from stable_baselines3.common.atari_wrappers import (
        ClipRewardEnv,
        EpisodicLifeEnv,
        FireResetEnv,
        MaxAndSkipEnv,
        NoopResetEnv,
    )
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        env = NoopResetEnv(env, noop_max=30)
        if not args.remove_skip:
            env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        env.action_space.seed(args.seed)
        return env

    return thunk


def mini_atari_make_env(env_id, idx, capture_video, run_name, args):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
        else:
            env = gym.make(env_id)

        env = gym.wrappers.RecordEpisodeStatistics(env)

        if args.history_states > 1:
            env = FrameConcatMinAtar(env, args.history_states)

        if args.num_last_actions > 0:
            env = LastActionWrapperMinAtar(env, num_last_actions=args.num_last_actions)

        env.metadata['render_fps'] = 30
        return env

    return thunk


def minigrid_make_env(env_id, idx, capture_video, run_name, args):
    from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper

    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)

        env = gym.wrappers.RecordEpisodeStatistics(env)

        env = RGBImgPartialObsWrapper(env)
        env = ImgObsWrapper(env)

        if args.history_states > 1:
            env = FrameConcatMinAtar(env, args.history_states)

        if args.num_last_actions > 0:
            env = LastActionWrapperMinAtar(env, num_last_actions=args.num_last_actions)

        return env

    return thunk


def cartpool_make_env(env_id, idx, capture_video, run_name, args):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env.unwrapped.tau = args.time_per_state
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if args.positive_obs:
            env = PositiveObsWrapper(env)
        return env

    return thunk


def make_env_discrete(env_id, idx, capture_video, run_name, args):

    if 'MinAtar' in env_id:
        args.inverse_channels = True
        return mini_atari_make_env(env_id, idx, capture_video, run_name, args)
    elif 'MiniGrid' in env_id:
        args.inverse_channels = False
        return minigrid_make_env(env_id, idx, capture_video, run_name, args)
    elif "Atari" in env_id:
        args.inverse_channels = False
        return atari_make_env(env_id, idx, capture_video, run_name, args)
    else:
        return cartpool_make_env(env_id, idx, capture_video, run_name, args)


def evaluate(
    actor,
    args,
    device,
    eval_episodes: int = 100,
    greedy: bool = False,
):
    envs = gym.vector.SyncVectorEnv([make_env_discrete(args.env_id, 0, True, args.run_name, args)])
    obs, _ = envs.reset(seed=args.seed)

    actor.eval()
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        with torch.no_grad():
            if not greedy:
                actions, _, _, _ = actor.get_action_and_value(torch.Tensor(obs).to(device))
            else:
                _, log_prob, _, _ = actor.get_action_and_value(torch.Tensor(obs).to(device))
                actions = torch.argmax(log_prob, dim=-1, keepdim=True)
        obs, _, _, _, infos = envs.step(actions.cpu().numpy())

        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                episodic_returns += [info["episode"]["r"]]
    actor.train()

    return episodic_returns


