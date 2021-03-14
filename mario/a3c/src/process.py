"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""

import torch
from src.env import create_train_env
from src.model import ActorCritic
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque
#from tensorboardX import SummaryWriter
import timeit
from src.helpers import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY, flag_get
import csv
import time
import numpy as np
import sys

def local_train(index, opt, global_model, optimizer, save=False):
    torch.manual_seed(123 + index)
    start_time = time.time()

    savefile = opt.saved_path + '/a3c_train' + opt.timestr + '.csv'
    title = ['Loops', 'Steps', 'Time', 'AvgLoss',
             'MeanReward', "StdReward", "TotalReward", "Flags"]
    with open(savefile, 'w', newline='') as sfile:
        writer = csv.writer(sfile)
        writer.writerow(title)

    env, num_states, num_actions = create_train_env(opt.world, opt.stage,opt.action_type)
    local_model = ActorCritic(num_states, num_actions)
    if opt.use_gpu:
        local_model.cuda()
    local_model.train()
    state = torch.from_numpy(env.reset())
    if opt.use_gpu:
        state = state.cuda()
    done = True
    curr_step = 0
    curr_episode = 0
    tot_reward=0
    tot_step=0
    got_flag=0

    while True:
        if save:
            if curr_episode % opt.save_interval == 0 and curr_episode > 0:
                torch.save(global_model.state_dict(),
                           "{}/a3c_super_mario_bros_{}_{}".format(opt.saved_path, opt.world, opt.stage))
                torch.save(global_model.state_dict(),"{}/a3c_super_mario_bros_{}_{}_{}".format(opt.saved_path, opt.world, opt.stage,curr_episode))
            print("Process {}. Episode {}".format(index, curr_episode))
        curr_episode += 1
        local_model.load_state_dict(global_model.state_dict())
        if done:
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

        for _ in range(opt.num_local_steps):
            curr_step += 1
            logits, value, h_0, c_0 = local_model(state, h_0, c_0)
            policy = F.softmax(logits, dim=1)
            log_policy = F.log_softmax(logits, dim=1)
            entropy = -(policy * log_policy).sum(1, keepdim=True)

            m = Categorical(policy)
            action = m.sample().item()

            state, reward, done, _ = env.step(action)
            #print(type(reward))
            state = torch.from_numpy(state)
            if opt.use_gpu:
                state = state.cuda()
            if curr_step > opt.num_global_steps:
                done = True

            if done:
                curr_step = 0
                state = torch.from_numpy(env.reset())
                if opt.use_gpu:
                    state = state.cuda()

            values.append(value)
            log_policies.append(log_policy[0, action])
            rewards.append(reward)
            entropies.append(entropy)
            tot_reward += reward

            if done:
                break

        R = torch.zeros((1, 1), dtype=torch.float)
        if opt.use_gpu:
            R = R.cuda()
        if not done:
            _, R, _, _ = local_model(state, h_0, c_0)

        gae = torch.zeros((1, 1), dtype=torch.float)
        if opt.use_gpu:
            gae = gae.cuda()
        actor_loss = 0
        critic_loss = 0
        entropy_loss = 0
        next_value = R

        for value, log_policy, reward, entropy in list(zip(values, log_policies, rewards, entropies))[::-1]:
            gae = gae * opt.gamma * opt.tau
            gae = gae + reward + opt.gamma * next_value.detach() - value.detach()
            next_value = value
            actor_loss = actor_loss + log_policy * gae
            R = R * opt.gamma + reward
            critic_loss = critic_loss + (R - value) ** 2 / 2
            entropy_loss = entropy_loss + entropy

        total_loss = -actor_loss + critic_loss - opt.beta * entropy_loss
        optimizer.zero_grad()
        total_loss.backward()

        any_flags = 0
        ep_time = time.time() - start_time

        avg_loss = 0
        mean_reward=0
       #print(total_loss)


        for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
            if global_param.grad is not None:
                break
            global_param._grad = local_param.grad
       # if done:
        #    ep_time = time.time() - start_time
         ##   print(curr_episode)
#
 #           if curr_episode % 100 == 0:
  #              data = [curr_step, curr_episode, "{:.6f}".format(ep_time), "{:.4f}".format(avg_loss), "{:.4f}".format(
   #             mean_reward), "{:.4f}".format(reward), "{:.2f}".format(tot_reward), any_flags]
#
                #data = [curr_episode, "{:.4f}".format(ep_time), "{:.2f}".format(reward), got_flag]
 #               with open(savefile, 'a', newline='') as sfile:
  #                  writer = csv.writer(sfile)
   #                 writer.writerows([data])

        if curr_episode % 100 == 0:

            data = [curr_step, curr_episode, "{:.6f}".format(ep_time), "{:.4f}".format(avg_loss), "{:.4f}".format(
            mean_reward), "{:.4f}".format(reward), "{:.2f}".format(reward), any_flags]
            
            with open(savefile, 'a', newline='') as sfile:
                writer = csv.writer(sfile)
                writer.writerows([data])

            #tot_reward = 0
       
       # ep_time = time.time() - start_time
       # print(curr_episode)
       # data = [curr_episode, "{:.4f}".format(ep_time), "{:.2f}".format(reward), got_flag]
        #with open(savefile, 'a', newline='') as sfile:
         #   writer = csv.writer(sfile)
          #  writer.writerows([data])
        #tot_reward=0
        optimizer.step()

        if curr_episode == int(opt.num_global_steps / opt.num_local_steps):
            print("Training process {} terminated".format(index))
            if save:
                end_time = timeit.default_timer()
                sys.exit('The code runs for %.2f s ' % (end_time - start_time))
                
            return


def local_test(index, opt, global_model):
    torch.manual_seed(123 + index)
    start_time = time.time()

    #writer = SummaryWriter(opt.log_path)
  #  savefile = opt.saved_path + '/A3C_test' + opt.timestr +  '.csv'
   # #print(savefile)
    #title = ['Steps', 'Time', 'TotalReward', "Flag"]
   # with open(savefile, 'w', newline='') as sfile:
    #    writer = csv.writer(sfile)
     #   writer.writerow(title)
    savefile = opt.saved_path + '/a3c_test' + opt.timestr + '.csv'
    print(savefile)
    title = ['Steps', 'Time', 'TotalReward', "Flag"]
    with open(savefile, 'w', newline='') as sfile:
        writer = csv.writer(sfile)
        writer.writerow(title)

    env, num_states, num_actions = create_train_env(opt.world, opt.stage,opt.action_type)
    local_model = ActorCritic(num_states, num_actions)
    local_model.eval()
    state = torch.from_numpy(env.reset())
    done = True
    curr_step = 0
    tot_reward=0
    tot_step=0
    got_flag=0
    actions = deque(maxlen=opt.max_actions)
    while True:
        curr_step += 1
        tot_step+=1
        if done:
            local_model.load_state_dict(global_model.state_dict())
        with torch.no_grad():
            if done:
                h_0 = torch.zeros((1, 512), dtype=torch.float)
                c_0 = torch.zeros((1, 512), dtype=torch.float)
            else:
                h_0 = h_0.detach()
                c_0 = c_0.detach()

        logits, value, h_0, c_0 = local_model(state, h_0, c_0)
        policy = F.softmax(logits, dim=1)
        action = torch.argmax(policy).item()
        state, reward, done, info = env.step(action)
        tot_reward += reward

        if flag_get(info):
            got_flag = 1
            print("Got flag")
            done = True
       #     torch.save(local_model.state_dict(),
         #              "{}/a3c_super_mario_bros_{}".format(opt.saved_path, curr_step))

        env.render()
        actions.append(action)
        if curr_step > opt.num_global_steps:
            done = True
            torch.save(local_model.state_dict(),
                       "{}/a3c_super_mario_bros_{}".format(opt.saved_path, curr_step))

            sys.exit("Training terminated")

        if done:
            ep_time = time.time() - start_time
            data = [tot_step, "{:.4f}".format(ep_time), "{:.2f}".format(tot_reward), got_flag]
            with open(savefile, 'a', newline='') as sfile:
                writer = csv.writer(sfile)
                writer.writerows([data])
            
            curr_step = 0
            got_flag = 0
            tot_reward = 0

            actions.clear()
            state = env.reset()

        state = torch.from_numpy(state)
