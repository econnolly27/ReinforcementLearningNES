"""
@author: Viet Nguyen <nhviet1009@gmail.com>
From: https://github.com/uvipen/Super-mario-bros-PPO-pytorch

Re-implemented to use gym-retro
"""

import os
import argparse
import torch
from src.env import MultipleEnvironments
from src.model import PPO
from src.process import evaluate
import torch.multiprocessing as _mp
from torch.distributions import Categorical
import torch.nn.functional as F
import numpy as np
import shutil, csv, time
from src.helpers import flag_get
from src.helpers import _is_stage_over
from datetime import datetime

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['DISPLAY'] = ':1'

TEST_ON_THE_GO = True

def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of model described in the paper: Proximal Policy Optimization Algorithms for Super Mario Bros""")
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--action_type", type=str, default="simple")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.9, help='discount factor for rewards')
    parser.add_argument('--tau', type=float, default=1.0, help='parameter for GAE')
    parser.add_argument('--beta', type=float, default=0.01, help='entropy coefficient')
    parser.add_argument('--epsilon', type=float, default=0.2, help='parameter for Clipped Surrogate Objective')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument("--num_local_steps", type=int, default=512)
    parser.add_argument("--num_global_steps", type=int, default=5e6)
    parser.add_argument("--num_processes", type=int, default=8, help="Number of concurrent processes, has to be larger than 1")
    parser.add_argument("--save_interval", type=int, default=50, help="Number of steps between savings")
    parser.add_argument("--max_actions", type=int, default=200, help="Maximum repetition steps in test phase")
    parser.add_argument("--log_path", type=str, default="tensorboard/ppo_super_mario_bros")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    args = parser.parse_args()
    return args


def check_flag(info):
    out = 0
    for i in info:
        #print(info)
        if flag_get(i):
            out += 1
    return out


def train(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
        print("using cuda")
    else:
        torch.manual_seed(123)
        print("not using cuda")
    
    opt.saved_path = os.getcwd() + '/mario/PPO/' + opt.saved_path
    # if os.path.isdir(opt.log_path):
    #     shutil.rmtree(opt.log_path)Miles Jupp
    
    # os.makedirs(opt.log_path)
    
    if not os.path.isdir(opt.saved_path):
        os.makedirs(opt.saved_path)

    start_time = time.time()       

    now = datetime.now()

    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)
    start_time = time.time()


    savefile = opt.saved_path + '/PPO_train.csv'
    print(savefile)
    title = ['Loops', 'Steps', 'Time', 'AvgLoss', 'MeanReward', "StdReward", "TotalReward", "Flags"]
    with open(savefile, 'w', newline='') as sfile:
        writer = csv.writer(sfile)
        writer.writerow(title)

    # Create environments
    envs = MultipleEnvironments(opt.world, opt.stage, opt.action_type, opt.num_processes)

    # Create model and optimizer
    model = PPO(envs.num_states, envs.num_actions)
    if torch.cuda.is_available():
        model.cuda()
    model.share_memory()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    # Start test/evaluation model
    if TEST_ON_THE_GO:
        # evaluate(opt, model, envs.num_states, envs.num_actions)
        mp = _mp.get_context("spawn")
        process = mp.Process(target=evaluate, args=(opt, model, envs.num_states, envs.num_actions))
        process.start()
    
    # Reset envs
    #[agent_conn.send(("reset", None)) for agent_conn in envs.agent_conns]
    curr_states = []
    [curr_states.append(env.reset()) for env in envs.envs]
    # curr_states = [agent_conn.recv() for agent_conn in envs.agent_conns]
    curr_states = torch.from_numpy(np.concatenate(curr_states, 0))
    if torch.cuda.is_available():
        curr_states = curr_states.cuda()

    tot_loops = 0
    tot_steps = 0

    # Start main loop 
    while True:
        # Save model each loop
        if tot_loops % opt.save_interval == 0 and tot_loops > 0:
            torch.save(model.state_dict(), "{}/PPO_super_mario_bros_{}_{}".format("trained_models", opt.world, opt.stage))
            torch.save(model.state_dict(), "{}/PPO_super_mario_bros_{}_{}_{}".format("trained_models", opt.world, opt.stage, tot_loops))


        # Accumulate evidence
        tot_loops += 1
        old_log_policies = []
        actions = []
        values = []
        states = []
        rewards = []
        dones = []
        flags = []
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
            # ac = action.cpu().item()
            if torch.cuda.is_available():
                # [agent_conn.send(("step", act)) for agent_conn, act in zip(envs.agent_conns, action.cpu())]
                [result.append(env.step(act.item())) for env, act in zip(envs.envs, action.cpu())]
            else:
                #[agent_conn.send(("step", act)) for agent_conn, act in zip(envs.agent_conns, action)]
                [result.append(env.step(act.item())) for env, act in zip(envs.envs, action)]

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
            flags.append(check_flag(info) / opt.num_processes)
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
            gae = gae + reward + opt.gamma * next_value.detach() * (1 - done) - value.detach()
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
                ratio = torch.exp(new_log_policy - old_log_policies[batch_indices])
                actor_loss = -torch.mean(torch.min(ratio * advantages[batch_indices], torch.clamp(ratio, 1.0 - opt.epsilon, 1.0 + opt.epsilon) * advantages[batch_indices]))
                # critic_loss = torch.mean((R[batch_indices] - value) ** 2) / 2
                critic_loss = F.smooth_l1_loss(R[batch_indices], value.squeeze())
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
        sum_reward = np.sum(all_rewards)
        mu_reward = np.mean(all_rewards)
        std_reward = np.std(all_rewards)
        any_flags = np.sum(flags)
        ep_time = time.time() - start_time
        # data = [tot_loops, tot_steps, ep_time, avg_loss, mu_reward, std_reward, sum_reward, any_flags]
        data = [tot_loops, tot_steps, "{:.6f}".format(ep_time), "{:.4f}".format(avg_loss), "{:.4f}".format(mu_reward), "{:.4f}".format(std_reward), "{:.2f}".format(sum_reward), any_flags]

        with open(savefile, 'a', newline='') as sfile:
            writer = csv.writer(sfile)
            writer.writerows([data])
        elapsed_time = time.time() - start_time
        print("Steps: {}. Total loss: {}. Time elapsed: {}".format(tot_steps, total_loss,time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))
        if check_flag(info):
            print("Stage finished")


if __name__ == "__main__":
    opt = get_args()
    train(opt)
