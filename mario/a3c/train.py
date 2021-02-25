"""
@author: Viet Nguyen <nhviet1009@gmail.com>
From: https://github.com/uvipen/Super-mario-bros-A3C-pytorch

Re-implemented to use gym-retro
"""
import os
import argparse
import torch
from src.env import MultipleEnvironments
from src.env import create_train_env
from src.model import ActorCritic
from src.optimizer import GlobalAdam
from src.process import local_test
import torch.multiprocessing as _mp
import shutil
import csv
import time
from src.helpers import flag_get
from datetime import datetime
import numpy as np
from torch.distributions import Categorical
import torch.nn.functional as F
from src.helpers import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY, flag_get


os.environ['OMP_NUM_THREADS'] = '1'
os.environ['DISPLAY'] = ':1'

TEST_ON_THE_GO = True


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of model described in the paper: Asynchronous Methods for Deep Reinforcement Learning for Super Mario Bros""")
    parser.add_argument("--world", type=int, default=2)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--action_type", type=str, default="simple")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.9,
                        help='discount factor for rewards')
    parser.add_argument('--tau', type=float, default=1.0,
                        help='parameter for GAE')
    parser.add_argument('--beta', type=float, default=0.01,
                        help='entropy coefficient')
    parser.add_argument("--num_local_steps", type=int, default=50)
    parser.add_argument("--num_global_steps", type=int, default=5e6)
    parser.add_argument("--num_processes", type=int, default=4)
    parser.add_argument("--save_interval", type=int,
                        default=500, help="Number of steps between savings")
    parser.add_argument("--max_actions", type=int, default=200,
                        help="Maximum repetition steps in test phase")
    parser.add_argument("--log_path", type=str,
                        default="tensorboard/a3c_super_mario_bros")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--load_from_previous_stage", type=bool, default=False,
                        help="Load weight from previous trained stage")
    parser.add_argument("--use_gpu", type=bool, default=True)
    args = parser.parse_args()
    return args


def check_flag(info):
    out = 0
    for i in info:
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

    opt.saved_path = os.getcwd() + '/mario/a3c/' + opt.saved_path

    start_time = time.time()

    now = datetime.now()

    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)
    start_time = time.time()

    if not os.path.isdir(opt.saved_path):
        os.makedirs(opt.saved_path)

    savefile = opt.saved_path + '/a3c_train.csv'
    print(savefile)
    title = ['Loops', 'Steps', 'Time', 'AvgLoss',
             'MeanReward', "StdReward", "TotalReward", "Flags"]
    with open(savefile, 'w', newline='') as sfile:
        writer = csv.writer(sfile)
        writer.writerow(title)

    # Create environments
    if opt.action_type == "right":
        actions = RIGHT_ONLY
    elif opt.action_type == "simple":
        actions = SIMPLE_MOVEMENT
    else:
        actions = COMPLEX_MOVEMENT

    envs = create_train_env(actions, mp_wrapper=False)
    num_actions = len(actions)
    num_states = envs.observation_space.shape[0]
    print(num_states)
    print(envs)
    # Create model and optimizer
    model = ActorCritic(num_states, num_actions)
    if torch.cuda.is_available():
        model.cuda()
    model.share_memory()
    optimizer = GlobalAdam(model.parameters(), lr=opt.lr)
    # Start processes

   # Start test/evaluation model
    if TEST_ON_THE_GO:
        mp = _mp.get_context("spawn")
        process = mp.Process(target=local_test, args=(1,
            opt, model, num_states, num_actions))
        process.start()

    # Reset envs
    #[agent_conn.send(("reset", None)) for agent_conn in envs.agent_conns]
   # curr_states = []
   # [curr_states.append(env.reset()) for env in envs.envs]
    # curr_states = [agent_conn.recv() for agent_conn in envs.agent_conns]
    #curr_states = torch.from_numpy(np.concatenate(curr_states, 0))
    #if torch.cuda.is_available():
     #   curr_states = curr_states.cuda()

    done = True
    curr_step = 0
    curr_episode = 0
    tot_loops = 0
    tot_steps = 0
    #state = torch.from_numpy(env.reset())

    while True:
        # Save model each loop
        if tot_loops % opt.save_interval == 0 and tot_loops > 0:
            torch.save(model.state_dict(
            ), "{}/a3c_super_mario_bros_{}_{}".format("trained_models", opt.world, opt.stage))
            torch.save(model.state_dict(), "{}/a3c_super_mario_bros_{}_{}_{}".format(
                "trained_models", opt.world, opt.stage, tot_loops))
        curr_episode += 1

        if done:
            model.load_state_dict(model.state_dict())
            h_0 = torch.zeros((1, 512), dtype=torch.float)
            c_0 = torch.zeros((1, 512), dtype=torch.float)
        else:
            h_0 = h_0.detach()
            c_0 = c_0.detach()
        if opt.use_gpu:
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()

        log_policies = []
        values = []
        rewards = []
        entropies = []
        states = []
        state = torch.from_numpy(envs.reset())
        state=state.cuda()
        for _ in range(opt.num_local_steps):
            curr_step += 1
            one = 0
            two = 0
            logits, value, h_0, c_0, one, two = model(
                state, h_0, c_0, one, two)
            policy = F.softmax(logits, dim=1)
            log_policy = F.log_softmax(logits, dim=1)
            entropy = -(policy * log_policy).sum(1, keepdim=True)

            m = Categorical(policy)
            action = m.sample().item()

            state, reward, done, _ = envs.step(action)
            state = torch.from_numpy(state)
            if opt.use_gpu:
                state = state.cuda()
            if curr_step > opt.num_global_steps:
                done = True

            if done:
                curr_step = 0
                state = torch.from_numpy(envs.reset())
                if opt.use_gpu:
                    state = state.cuda()

            values.append(value)
            log_policies.append(log_policy[0, action])
            rewards.append(reward)
            entropies.append(entropy)

            if done:
                break

        R = torch.zeros((1, 1), dtype=torch.float)
        if opt.use_gpu:
            R = R.cuda()
        if not done:
            _, R, _, _, states, actions = model(
                state, h_0, c_0, num_states,num_actions)

        gae = torch.zeros((1, 1), dtype=torch.float)
        if opt.use_gpu:
            gae = gae.cuda()
        actor_loss = 0
        critic_loss = 0
        entropy_loss = 0
        next_value = R
        avg_loss = []

        for value, log_policy, reward, entropy in list(zip(values, log_policies, rewards, entropies))[::-1]:
            gae = gae * opt.gamma * opt.tau
            gae = gae + reward + opt.gamma * next_value.detach() - value.detach()
            next_value = value
            actor_loss = actor_loss + log_policy * gae
            R = R * opt.gamma + reward
            critic_loss = critic_loss + (R - value) ** 2 / 2
            entropy_loss = entropy_loss + entropy

        total_loss = -actor_loss + critic_loss - opt.beta * entropy_loss
        # writer.add_scalar("Train_{}/Loss".format(index), total_loss, curr_episode)
        optimizer.zero_grad()
        #print(total_loss)
        total_loss.backward()
        # print(total_loss)
        for local_param, global_param in zip(model.parameters(), model.parameters()):
            if global_param.grad is not None:
                break
            global_param._grad = local_param.grad
        # env.render()

        optimizer.step()
        avg_loss.append(total_loss.cpu().detach().numpy().tolist())
        avg_loss = np.mean(avg_loss)
        all_rewards = rewards
        #tot_steps += opt.num_local_steps * opt.num_processes
        sum_reward = np.sum(all_rewards)
        mu_reward = np.mean(all_rewards)
        std_reward = np.std(all_rewards)
        #any_flags = np.sum(flags)
        ep_time = time.time() - start_time
        #data = [tot_loops, tot_steps, ep_time, avg_loss, mu_reward, std_reward, sum_reward, any_flags]
        data = [curr_step, curr_episode, "{:.6f}".format(ep_time), "{:.4f}".format(avg_loss), "{:.4f}".format(mu_reward), "{:.4f}".format(std_reward), "{:.2f}".format(sum_reward)]

        with open(savefile, 'a', newline='') as sfile:
           writer = csv.writer(sfile)
           writer.writerows([data])
        elapsed_time = time.time() - start_time
        # if check_flag(info):
        #   print("Stage finished")

        # if curr_episode == int(opt.num_global_steps / opt.num_local_steps):
        #print("Training process {} terminated".format(index))
        # if save:
        #   end_time = timeit.default_timer()
        #  print('The code runs for %.2f s ' % (end_time - start_time))

        print("Steps: {}. Total loss: {}. Time elapsed: {}".format(
            curr_step, total_loss, time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))


if __name__ == "__main__":
    opt = get_args()
    train(opt)
