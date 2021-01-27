"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""

import torch
from src.env import create_train_env
from src.model import ActorCritic
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque
from tensorboardX import SummaryWriter
import timeit
from src.helpers import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY, flag_get
import numpy as np
import time



def local_train(index, opt, global_model, optimizer,num_states,num_actions, save=False):
    torch.manual_seed(123 + index)
    if save:
        start_time = timeit.default_timer()
    writer = SummaryWriter(opt.log_path)

    if opt.action_type == "right":
        actions = RIGHT_ONLY
    elif opt.action_type == "simple":
        actions = SIMPLE_MOVEMENT
    else:
        actions = COMPLEX_MOVEMENT

    env = create_train_env(actions, mp_wrapper=False)
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
    start_time = time.time()
    while True:
        if save:
            if curr_episode % opt.save_interval == 0 and curr_episode > 0:
                torch.save(global_model.state_dict(),
                           "{}/a3c_super_mario_bros_{}_{}".format(opt.saved_path, opt.world, opt.stage))
            #print("Process {}. Episode {}".format(index, curr_episode))
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
        
        old_log_policies = []
        actions = []
        states = []
        dones = []
        flags = []

        for _ in range(opt.num_local_steps):
            curr_step += 1
            one=0
            two=0
            logits, value, h_0, c_0,one,two = local_model(state, h_0, c_0,one,two)
            policy = F.softmax(logits, dim=1)
            log_policy = F.log_softmax(logits, dim=1)
            entropy = -(policy * log_policy).sum(1, keepdim=True)

            m = Categorical(policy)
            action = m.sample().item()

            state, reward, done, _ = env.step(action)
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

            if done:
                break

        R = torch.zeros((1, 1), dtype=torch.float)
        if opt.use_gpu:
            R = R.cuda()
        if not done:
            _, R, _, _,states,actions = local_model(state, h_0, c_0,num_states,num_actions)

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
        print(total_loss)
        total_loss.backward()
        #print(total_loss)
        for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
            if global_param.grad is not None:
                break
            global_param._grad = local_param.grad
        #env.render()

        optimizer.step()
        avg_loss.append(total_loss.cpu().detach().numpy().tolist())
        avg_loss = np.mean(avg_loss)
        all_rewards = rewards
        #tot_steps += opt.num_local_steps * opt.num_processes
        sum_reward = np.sum(all_rewards)
        mu_reward = np.mean(all_rewards)
        std_reward = np.std(all_rewards)
        any_flags = np.sum(flags)
        ep_time = time.time() - start_time
        # data = [tot_loops, tot_steps, ep_time, avg_loss, mu_reward, std_reward, sum_reward, any_flags]
       # data = [curr_step, curr_episode, "{:.6f}".format(ep_time), "{:.4f}".format(avg_loss), "{:.4f}".format(mu_reward), "{:.4f}".format(std_reward), "{:.2f}".format(sum_reward), any_flags]

        #with open(savefile, 'a', newline='') as sfile:
         #   writer = csv.writer(sfile)
          #  writer.writerows([data])
        elapsed_time = time.time() - start_time
        #if check_flag(info):
         #   print("Stage finished")

        if curr_episode == int(opt.num_global_steps / opt.num_local_steps):
            print("Training process {} terminated".format(index))
            if save:
                end_time = timeit.default_timer()
                print('The code runs for %.2f s ' % (end_time - start_time))
            
        print("Steps: {}. Total loss: {}. Time elapsed: {}".format(curr_step, total_loss,time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))



def local_test(index, opt, global_model,num_states,num_actions):

    if opt.action_type == "right":
        actions = RIGHT_ONLY
    elif opt.action_type == "simple":
        actions = SIMPLE_MOVEMENT
    else:
        actions = COMPLEX_MOVEMENT

    torch.manual_seed(123 + index)
    env = create_train_env(actions, mp_wrapper=False)
    local_model = ActorCritic(num_states, num_actions)
    local_model.eval()
    state = torch.from_numpy(env.reset())
    done = True
    curr_step = 0
    actions = deque(maxlen=opt.max_actions)
    while True:
        curr_step += 1
        if done:
            local_model.load_state_dict(global_model.state_dict())
        with torch.no_grad():
            if done:
                h_0 = torch.zeros((1, 512), dtype=torch.float)
                c_0 = torch.zeros((1, 512), dtype=torch.float)
            else:
                h_0 = h_0.detach()
                c_0 = c_0.detach()
        one=0
        two=0
        logits, value, h_0, c_0,one,two = local_model(state, h_0, c_0,one,two)
        policy = F.softmax(logits, dim=1)
        action = torch.argmax(policy).item()
        state, reward, done, _ = env.step(action)
        env.render()
        actions.append(action)
        if curr_step > opt.num_global_steps or actions.count(actions[0]) == actions.maxlen:
            done = True
        if done:
            curr_step = 0
            actions.clear()
            state = env.reset()
        state = torch.from_numpy(state)