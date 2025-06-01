# RL Experiments for "Learning From the Past with Cascading Eligibility Traces"

This repository contains reinforcement learning experiments for the paper **"Learning From the Past with Cascading Eligibility Traces."**

We compare three Actor-Critic training setups:
- **Standard backpropagation**
- **Classical eligibility traces (ET)**
- **Cascading eligibility traces (CET)**

Experiments are run on the following environments:
- `CartPole`
- `LunarLander`
- `MinAtar/SpaceInvaders`

## Hyperparameter Search

The following values were used in the hyperparameter search:

- **Learning rate**: `2.5e-4`, `5e-4`, `9e-4`, `1e-4`
- **ET discounting factor, Beta**: `0.5`, `0.7`, `0.9`, `0.99`
- **Delay values** (behavioral timescales): `10`, `20`, `40`, `160`, `320`, `640`  
  (Correspond to `2`, `4`, `8`, `32`, `64`, `128` seconds in the paper)
- **CET order (`ssm_cascade_size`)**: `2`, `5`, `8`, `10`
- **Normalization scheme for CET**: `normalized_by_max_value`

Refer to the Appendix of the paper for the optimal parameters used in each setting.

## Installation

To run the experiments, install the required packages:

```bash
pip install -r requirements.txt
```

## Default backprop

To run the Actor-Critic agent with standard backpropagation:

```
python train_ac.py --agent AgentActorETNoisy \
                    --env_id 'CartPole-v1'  \
                    --positive_obs \
                    --anneal_lr \
                    --num_steps 128 \
                    --num_envs 4 \
                    --top_grad "vanilla" \
                    --backward_weights "vanilla" \
                    --backward "None" \
                    --et_strategy "accumulate" \
                    --total_timesteps 5000000 \
                    --learning_rate $lr \
                    --exp_name "vanilla_lr{$lr}" \
                    --track
```

```
python train_ac.py --agent AgentActorETNoisy \
                    --env_id 'LunarLander-v2'  \
                    --num_steps 128 \
                    --num_envs 4 \
                    --positive_obs \
                    --top_grad "vanilla" \
                    --backward_weights "vanilla" \
                    --backward "None" \
                    --et_strategy "accumulate" \
                    --total_timesteps 5000000 \
                    --learning_rate $lr \
                    --exp_name "vanilla_lr{$lr}" \
                    --track
```

```
python train_ac.py --agent AgentActorETNoisy \
                    --env_id MinAtar/SpaceInvaders-v0 \
                    --gae_lambda_random \
                    --num_steps 32 \
                    --top_grad vanilla \
                    --backward_weights vanilla \
                    --backward None \
                    --et_strategy accumulate \
                    --total_timesteps 10000000 \
                    --learning_rate $lr \
                    --exp_name "vanilla_lr{$lr}" \
                    --track
```

## Behavioral timescale

To run CET experiments across different delays (behavioral timescale), use the following command pattern:

```
python train_ac.py --agent AgentActorETNoisy \
                    --env_id 'CartPole-v1'  \
                    --positive_obs \
                    --delay_grad \
                    --anneal_lr \
                    --num_steps 128 \
                    --num_envs 4 \
                    --constant_delay \
                    --delay $delay \
                    --normalized_by_max_value \
                    --ssm_cascade_size $order \
                    --top_grad "delayed_vanilla" \
                    --backward_weights "SSM" \
                    --backward "None" \
                    --et_strategy "accumulate" \
                    --total_timesteps 5000000 \
                    --learning_rate $lr \
                    --exp_name "grid_exactssm{$order}d{$delay}_lr{$lr}" \
                    --track
```

```
python train_ac.py --agent AgentActorETNoisy \
                    --env_id 'LunarLander-v2'  \
                    --delay_grad \
                    --num_steps 128 \
                    --num_envs 4 \
                    --positive_obs \
                    --constant_delay \
                    --delay $delay \
                    --ssm_cascade_size $order \
                    --top_grad "delayed_vanilla" \
                    --backward_weights "SSM" \
                    --backward "None" \
                    --et_strategy "accumulate" \
                    --total_timesteps 5000000 \
                    --learning_rate $lr \
                    --exp_name "cet{$order}d{$delay}_lr{$lr}" \
                    --track
```

```
python train_ac.py --agent AgentActorETNoisy \
                    --env_id MinAtar/SpaceInvaders-v0 \
                    --delay_grad \
                    --gae_lambda_random \
                    --num_steps 32 \
                    --delay $delay \
                    --ssm_cascade_size $order \
                    --constant_delay \
                    --normalized_by_max_value \
                    --clip_vloss \
                    --top_grad delayed_vanilla \
                    --backward_weights SSM \
                    --backward None \
                    --et_strategy accumulate \
                    --total_timesteps 10000000 \
                    --learning_rate $lr \
                    --exp_name "cet{$order}d{$delay}_lr{$lr}" \
                    --track
```

To run classic ET across different delays on behavioral timescale (behavioral timescale):

```
python train_ac.py --agent AgentActorETNoisy \
                    --env_id 'CartPole-v1'  \
                    --positive_obs \
                    --delay_grad \
                    --anneal_lr \
                    --num_steps 128 \
                    --num_envs 4 \
                    --norm_adv 3 \
                    --constant_delay \
                    --delay $delay \
                    --ssm_cascade_size 1 \
                    --top_grad "delayed_vanilla" \
                    --backward_weights "SSM_threshold" \
                    --backward "None" \
                    --et_strategy "accumulate" \
                    --total_timesteps 5000000 \
                    --learning_rate $lr \
                    --beta $beta \
                    --exp_name "et_d{$delay}_lr{$lr}_beta{$beta}" \
                    --track
```

```
python train_ac.py --agent AgentActorETNoisy \
                    --env_id 'LunarLander-v2'  \
                    --delay_grad \
                    --num_steps 128 \
                    --num_envs 4 \
                    --norm_adv 3 \
                    --positive_obs \
                    --constant_delay \
                    --delay $delay \
                    --ssm_cascade_size 1 \
                    --top_grad "delayed_vanilla" \
                    --backward_weights "SSM" \
                    --backward "None" \
                    --et_strategy "accumulate" \
                    --total_timesteps 5000000 \
                    --learning_rate $lr \
                    --beta $beta \
                    --exp_name "et_d{$delay}_lr{$lr}_beta{$beta}" \
                    --track
```

```
python train_ac.py --agent AgentActorETNoisy \
                    --env_id MinAtar/SpaceInvaders-v0 \
                    --delay_grad \
                    --gae_lambda_random \
                    --norm_adv 3 \
                    --num_steps 32 \
                    --delay $delay \
                    --beta $beta \
                    --ssm_cascade_size 1 \
                    --constant_delay \
                    --clip_vloss \
                    --top_grad delayed_vanilla \
                    --backward_weights SSM \
                    --backward None \
                    --et_strategy accumulate \
                    --total_timesteps 10000000 \
                    --learning_rate $lr \
                    --exp_name "et_d{$delay}_lr{$lr}_beta{$beta}" \
                    --track
```

## Retrograde signaling

To run CET for retrograde signaling:

```
python train_ac.py --agent AgentActorETNoisy \
                    --env_id 'CartPole-v1'  \
                    --positive_obs \
                    --delay_grad \
                    --anneal_lr \
                    --num_steps 128 \
                    --num_envs 4 \
                    --norm_adv 3 \
                    --delay 400 \
                    --last_layer_no_delay \
                    --normalized_by_max_value \
                    --ssm_cascade_size $order \
                    --top_grad "delayed_vanilla" \
                    --backward_weights "SSM" \
                    --backward "None" \
                    --et_strategy "accumulate" \
                    --total_timesteps 5000000 \
                    --learning_rate $lr \
                    --exp_name "retro_cet{$order}d400_lr{$lr}" \
                    --time_per_state 0.3 \
                    --track
```

```
python train_ac.py --agent AgentActorETNoisy \
                    --env_id 'LunarLander-v2'  \
                    --delay_grad \
                    --num_steps 128 \
                    --num_envs 4 \
                    --norm_adv 3 \
                    --positive_obs \
                    --delay 400 \
                    --last_layer_no_delay \
                    --normalized_by_max_value \
                    --ssm_cascade_size $order \
                    --top_grad "delayed_vanilla" \
                    --backward_weights "SSM" \
                    --backward "None" \
                    --et_strategy "accumulate" \
                    --total_timesteps 5000000 \
                    --learning_rate $lr \
                    --exp_name "retro_cet{$order}d400_lr{$lr}" \
                    --time_per_state 0.3 \
                    --track
```

To run classic ET for retrograde signaling:

```
python train_ac.py --agent AgentActorETNoisy \
                    --env_id 'CartPole-v1'  \
                    --positive_obs \
                    --delay_grad \
                    --anneal_lr \
                    --num_steps 128 \
                    --num_envs 4 \
                    --norm_adv 3 \
                    --delay 400 \
                    --last_layer_no_delay \
                    --ssm_cascade_size 1 \
                    --top_grad "delayed_vanilla" \
                    --backward_weights "SSM" \
                    --backward "None" \
                    --et_strategy "accumulate" \
                    --total_timesteps 5000000 \
                    --learning_rate $lr \
                    --beta $beta \
                    --exp_name "retro_et_d400_lr{$lr}_beta{$beta}" \
                    --time_per_state 0.3 \
                    --track
```

```
python train_ac.py --agent AgentActorETNoisy \
                    --env_id 'LunarLander-v2'  \
                    --delay_grad \
                    --num_steps 128 \
                    --num_envs 4 \
                    --norm_adv 3 \
                    --positive_obs \
                    --delay 400 \
                    --last_layer_no_delay \
                    --ssm_cascade_size 1 \
                    --top_grad "delayed_vanilla" \
                    --backward_weights "SSM" \
                    --backward "None" \
                    --et_strategy "accumulate" \
                    --total_timesteps 5000000 \
                    --learning_rate $lr \
                    --beta $beta \
                    --exp_name "retro_et_d400_lr{$lr}_beta{$beta}" \
                    --time_per_state 0.3 \
                    --track
```