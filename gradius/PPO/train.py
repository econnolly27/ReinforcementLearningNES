"""
@author: Viet Nguyen <nhviet1009@gmail.com>
From: https://github.com/uvipen/Super-mario-bros-PPO-pytorch

Re-implemented to use gym-retro by Gerardo Aragon-Camarasa

Modified for Benchmarking Reinforcement Learning Algorithms in NES Games by Erin-Louise Connolly
"""
import argparse
import csv
import os
import shutil
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.multiprocessing as _mp
import torch.nn.functional as F
from torch.distributions import Categorical

from src.env import MultipleEnvironments
from src.model import PPO
from src.process import evaluate

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['DISPLAY'] = ':1'

TEST_ON_THE_GO = True


def get_args():
    timestr = time.strftime("%Y%m%d-%H%M%S")
    parser = argparse.ArgumentParser(
        """Implementation of model described in the paper: Proximal Policy Optimization Algorithms for Super Mario Bros""")
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--action_type", type=str, default="complex")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.9,
                        help='discount factor for rewards')
    parser.add_argument('--tau', type=float, default=1.0,
                        help='parameter for GAE')
    parser.add_argument('--beta', type=float, default=0.01,
                        help='entropy coefficient')
    parser.add_argument('--epsilon', type=float, default=0.2,
                        help='parameter for Clipped Surrogate Objective')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument("--num_local_steps", type=int, default=512)
    parser.add_argument("--num_global_steps", type=int, default=2e6)
    parser.add_argument("--num_processes", type=int, default=4,
                        help="Number of concurrent processes, has to be larger than 1")
    parser.add_argument("--save_interval", type=int, default=50,
                        help="Number of steps between savings")
    parser.add_argument("--max_actions", type=int, default=200,
                        help="Maximum repetition steps in test phase")
    parser.add_argument("--timestr", type=str, default=timestr)
    parser.add_argument("--log_path", type=str,
                        default="tensorboard/ppo_gradius" + timestr)
    parser.add_argument("--saved_path", type=str, default="trained_models/" + timestr)
    args = parser.parse_args()
    return args


def train(opt):
    seed = 123
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        print("using cuda")
    else:
        torch.manual_seed(seed)
        print("not using cuda")

    opt.saved_path = os.getcwd() + '/gradius/PPO/' + opt.saved_path

    if not os.path.isdir(opt.saved_path):
        os.makedirs(opt.saved_path)

    start_time = time.time()

    now = datetime.now()

    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)
    start_time = time.time()

    savefile = opt.saved_path + '/gradius_PPO_train' + opt.timestr + '.csv'
    print(savefile)
    title = ['Loops', 'Steps', 'Time','MeanReward', 'Reward1',"Reward2","Reward3","Reward4","Score1","Score2","Score3","Score4"]
    with open(savefile, 'w', newline='') as sfile:
        writer = csv.writer(sfile)
        writer.writerow(title)

    # Create environments
    envs = MultipleEnvironments(
        opt.world, opt.stage, opt.action_type, opt.num_processes)

    # Create model and optimizer
    model = PPO(envs.num_states, envs.num_actions)
    if torch.cuda.is_available():
        model.cuda()
    model.share_memory()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    # Start test/evaluation model
    if TEST_ON_THE_GO:
        mp = _mp.get_context("spawn")
        process = mp.Process(target=evaluate, args=(
            opt, model, envs.num_states, envs.num_actions))
        process.start()

    # Reset envs
    curr_states = []
    [curr_states.append(env.reset()) for env in envs.envs]
    curr_states = torch.from_numpy(np.concatenate(curr_states, 0))
    if torch.cuda.is_available():
        curr_states = curr_states.cuda()

    tot_loops = 0
    tot_steps = 0

    # Start main loop
    while True:
        # Save model each loop
        if tot_loops % opt.save_interval == 0 and tot_loops > 0:
            torch.save(model.state_dict(
            ), "{}/PPO_gradius_{}_{}".format(opt.saved_path, opt.world, opt.stage))
            torch.save(model.state_dict(), "{}/PPO_gradius_{}_{}_{}".format(
                opt.saved_path, opt.world, opt.stage, tot_loops))



        if tot_steps > opt.num_global_steps:
            torch.save(model.state_dict(
            ), "{}/PPO_gradius_{}_{}".format(opt.saved_path, opt.world, opt.stage))

            end_time = time.time() - start_time
            print('The code runs for {}'.format(time.strftime("%H:%M:%S", time.gmtime(end_time))))
            print("Training process terminated")
            process.terminate()
            torch.cuda.empty_cache()
            os._exit(0)

        # Accumulate evidence
        tot_loops += 1
        old_log_policies = []
        actions = []
        values = []
        states = []
        rewards = []
        dones = []
        scores = []
        for _ in range(opt.num_local_steps):
            # From given states, predict an action
            states.append(curr_states)
            logits, value = model(curr_states)

            values.append(value.squeeze())
            policy = F.softmax(logits, dim=1)
            old_m = Categorical(policy)
            action = old_m.sample()
            actions.append(action)
            old_log_policy = old_m.log_prob(action)
            old_log_policies.append(old_log_policy)

            # Evaluate predicted action
            result = []
            if torch.cuda.is_available():
                [result.append(env.step(act.item()))
                 for env, act in zip(envs.envs, action.cpu())]
            else:
                [result.append(env.step(act.item()))
                 for env, act in zip(envs.envs, action)]

            state, reward, done, info = zip(*result)
            state = torch.from_numpy(np.concatenate(state, 0))
            if torch.cuda.is_available():
                state = state.cuda()
                reward = torch.cuda.FloatTensor(reward)
                done = torch.cuda.FloatTensor(done)
            else:
                reward = torch.FloatTensor(reward)
                done = torch.FloatTensor(done)


            rewards.append(reward)
            dones.append(done)
            for i in range(4):
                scores.append((info[i]['score']))

            curr_states = state

        # Training stage
        _, next_value, = model(curr_states)
        next_value = next_value.squeeze()
        old_log_policies = torch.cat(old_log_policies).detach()
        actions = torch.cat(actions)
        values = torch.cat(values).detach()
        states = torch.cat(states)
        gae = 0
        R = []
        for value, reward, done in list(zip(values, rewards, dones))[::-1]:
            gae = gae * opt.gamma * opt.tau
            gae = gae + reward + opt.gamma * next_value.detach() * (1 - done) - \
                value.detach()
            next_value = value
            R.append(gae + value)

        R = R[::-1]
        R = torch.cat(R).detach()
        advantages = R - values
        avg_loss = []
        for _ in range(opt.num_epochs):
            indice = torch.randperm(opt.num_local_steps * opt.num_processes)
            for j in range(opt.batch_size):
                batch_indices = indice[int(j * (opt.num_local_steps * opt.num_processes / opt.batch_size)): int((j + 1) * (
                    opt.num_local_steps * opt.num_processes / opt.batch_size))]
                logits, value = model(states[batch_indices])
                new_policy = F.softmax(logits, dim=1)
                new_m = Categorical(new_policy)
                new_log_policy = new_m.log_prob(actions[batch_indices])
                ratio = torch.exp(
                    new_log_policy - old_log_policies[batch_indices])
                actor_loss = -torch.mean(torch.min(ratio * advantages[batch_indices], torch.clamp(
                    ratio, 1.0 - opt.epsilon, 1.0 + opt.epsilon) * advantages[batch_indices]))
                critic_loss = F.smooth_l1_loss(
                    R[batch_indices], value.squeeze())
                entropy_loss = torch.mean(new_m.entropy())
                total_loss = actor_loss + critic_loss - opt.beta * entropy_loss
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                avg_loss.append(total_loss.cpu().detach().numpy().tolist())

        avg_loss = np.mean(avg_loss)
        all_rewards = torch.cat(rewards).cpu().numpy()
        tot_steps += opt.num_local_steps * opt.num_processes
        mu_reward = np.mean(all_rewards)
        ep_time = time.time() - start_time

        data = [tot_loops, tot_steps, "{:.6f}".format(ep_time), "{:.4f}".format(
            mu_reward),(all_rewards[0]),(all_rewards[1]),(all_rewards[2]),(all_rewards[3]),(scores[0]),(scores[1]),(scores[2]),(scores[3])]

        with open(savefile, 'a', newline='') as sfile:
            writer = csv.writer(sfile)
            writer.writerows([data])
        elapsed_time = time.time() - start_time
        print("Steps: {}. Time elapsed: {}".format(
            tot_steps, time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))

if __name__ == "__main__":
    opt = get_args()
    print(opt.timestr)
    train(opt)
