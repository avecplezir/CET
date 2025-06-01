import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


def get_lr(it, args):
    assert it <= args.num_iterations
    if it < args.warmup_iters:
        return it / args.warmup_iters
    else:
        decay_ratio = 1 - (it - args.warmup_iters) / (args.num_iterations + 1 - args.warmup_iters)
        return decay_ratio


def train_et_continual(args,
                     optimizer,
                     device,
                     envs,
                     agent,
                     writer:SummaryWriter,
                       ):
    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + args.obs_shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    global_iter = 0
    start_time = time.time()
    next_obs, _ = envs.reset()
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    global_step_threshold = 0
    grads = None

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            # frac = 1.0 - (iteration - 1.0) / args.num_iterations
            frac = get_lr(iteration, args)
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        agent.eval()
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            global_iter += 1
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)

            values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        if global_step > global_step_threshold:
                            print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                            writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                            writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                            global_step_threshold += args.episodic_return_threshold

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            advantages_simple = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                advantages_simple[t] = delta
            returns = advantages + values

        agent.train()
        # Optimizing the policy and value network
        clipfracs = []

        critic_loss = 0
        entropy_loss = 0
        for idx in range(0, args.num_steps):
            b_advantages_simple = advantages_simple[idx]
            b_returns = returns[idx]
            b_values = values[idx]

            agent.set_step(idx)
            _, newlogprob, entropy, newvalue = agent.get_action_and_value(obs[idx], action=actions.long()[idx])
            mb_b_advantages_simple = b_advantages_simple
            if args.norm_adv == 1:
                mb_b_advantages_simple = (mb_b_advantages_simple - mb_b_advantages_simple.mean()) / (mb_b_advantages_simple.std() + 1e-8)
            elif args.norm_adv == 2:
                mb_b_advantages_simple = (mb_b_advantages_simple - advantages.mean()) / (advantages.std() + 1e-8)

            pg_loss = newlogprob.mean()

            # Value loss
            newvalue = newvalue.view(-1)
            if args.clip_vloss:
                v_loss_unclipped = (newvalue - b_returns) ** 2
                v_clipped = b_values + torch.clamp(
                    newvalue - b_values,
                    -args.clip_coef,
                    args.clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((newvalue - b_returns) ** 2).mean()

            entropy_loss += -args.ent_coef * entropy.mean()
            loss = pg_loss / args.num_steps
            critic_loss += v_loss * args.vf_coef

            loss.backward(retain_graph=True)
            grads = agent.backward(grads=grads, adv=mb_b_advantages_simple)

            # setting et to zero for the inital iteration to not repeat the same et in case of delayed gradient
            # if global_iter > args.buffer_size:
            if global_iter > 30:
                agent.backward_weights(mb_b_advantages_simple, dones[idx])
            else:
                agent.reset_et()
                agent.reset_weights()

        critic_loss /= args.num_steps
        optimizer.zero_grad()
        agent.set_weights()
        critic_loss.backward()
        entropy_loss /= args.num_steps
        entropy_loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
        optimizer.step()
        agent.reset_weights()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if iteration % args.log_interval == 0:
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            if args.collect_grad_stats:
                for i, block in enumerate(agent.actor):
                    writer.add_scalar(f"activation_ratio/{i}", block.activation_ratio, global_step)
                    writer.add_scalar(f"et_activation_ratio/{i}", block.et_activation_ratio, global_step)
                    writer.add_scalar(f"local_et_ratio/{i}", block.local_et_ratio, global_step)
                    writer.add_scalar(f"intersection_ratio/{i}", block.intersection_ratio, global_step)
                    writer.add_scalar(f"weight_grad_cos_similarity/{i}", block.weight_grad_cos_similarity, global_step)
                    writer.add_scalar(f"out_grad_cos_similarity/{i}", block.out_grad_cos_similarity, global_step)
                    writer.add_scalar(f"grad_activation_similarity/{i}", block.grad_activation_similarity, global_step)
                    writer.add_scalar(f"activation_similarity/{i}", block.activation_similarity, global_step)
                    writer.add_scalar(f"local_et_similarity/{i}", block.local_et_similarity, global_step)
                    writer.add_scalar(f"weight_grad_norm/{i}", block.weight_grad_norm, global_step)
                    writer.add_scalar(f"out_grad_norm/{i}", block.out_grad_norm, global_step)
                    if args.collect_impulse_responses:
                        for i, block in enumerate(agent.actor):
                            for j, (impulse, impulse_response) in  enumerate(zip(block.impulses, block.impulse_responses)):
                                writer.add_scalar(f"impulse/{i}", impulse, global_step+j)
                                writer.add_scalar(f"impulse_response/{i}", impulse_response, global_step + j)

                    if args.adv_threshold > 0:
                        for i, block in enumerate(agent.actor):
                            writer.add_scalar(f"running_avr_thresholding/{i}", block.running_avr_thresholding, global_step)
                            writer.add_scalar(f"adv_threshold/{i}", block.adv_threshold, global_step)

    if args.save_model:
        model_path = f"runs/{args.run_name}/{args.exp_name}.actor.slowagent_model"
        print('Saving model to', model_path)
        torch.save(agent.state_dict(), model_path)

    if args.eval_model:
        print('evaluating model after training')
        from utils import evaluate
        episodic_returns = evaluate(agent, args, args.device)

        episodic_returns = np.array(episodic_returns)
        mean, std = np.mean(episodic_returns), np.std(episodic_returns)
        writer.add_scalar("eval/mean_episodic_return", mean)
        writer.add_scalar("eval/std_episodic_return", std)

        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

    envs.close()
    writer.close()
