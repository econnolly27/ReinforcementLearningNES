"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""

import csv
import os
import sys
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from src.env import create_train_env
from src.helpers import COMPLEX_MOVEMENT, RIGHT_ONLY, SIMPLE_MOVEMENT, flag_get
from src.model import PPO


def evaluate(opt, global_model, num_states, num_actions):
    torch.manual_seed(123)
    if opt.action_type == "right":
        actions = RIGHT_ONLY
    elif opt.action_type == "simple":
        actions = SIMPLE_MOVEMENT
    else:
        actions = COMPLEX_MOVEMENT

    savefile = opt.saved_path + '/mario_PPO_test' + opt.timestr + '.csv'
    print(savefile)
    title = ['Steps', 'Time', 'TotalReward', "Flag"]
    with open(savefile, 'w', newline='') as sfile:
        writer = csv.writer(sfile)
        writer.writerow(title)

    env = create_train_env(opt.world,opt.stage,actions, mp_wrapper=False)
    local_model = PPO(num_states, num_actions)
    if torch.cuda.is_available():
        local_model.cuda()
    local_model.eval()

    state = torch.from_numpy(env.reset())
    if torch.cuda.is_available():
        state = state.cuda()
    
    done = True
    curr_step = 0
    tot_step = 0
    actions = deque(maxlen=opt.max_actions)
    tot_reward = 0
    got_flag = 0
    while True:
        start_time = time.time()
        curr_step += 1
        tot_step += 1
        if done:
            local_model.load_state_dict(global_model.state_dict())

        logits, value = local_model(state)
        policy = F.softmax(logits, dim=1)
        action = torch.argmax(policy).item() # This selects the best action to take
        state, reward, done, info = env.step(action)
        tot_reward += reward
        if flag_get(info):
            got_flag = 1
            done = True
           # print("Got flag in testing at time {} step {}".format(tot_step,time.strftime("%H:%M:%S", time.gmtime(time.time()))))

            #Uncomment previous lines if you want to save when flag is reached
            #torch.save(local_model.state_dict(),
             #          "{}/PPO_super_mario_bros_{}".format(opt.saved_path, curr_step))

        env.render()
        actions.append(action)

        if actions.count(actions[0]) == actions.maxlen:
            done = True

        #print(tot_step)
        #if tot_step > opt.num_global_steps * 2:
           # torch.save(local_model.state_dict(),
              #         "{}/PPO_super_mario_bros_{}".format(opt.saved_path, curr_step))

            #sys.exit("Test process terminated")
        if done:
            # print("Evaluate: Done!")
            ep_time = time.time() - start_time
            data = [tot_step, "{:.4f}".format(ep_time), "{:.2f}".format(tot_reward), got_flag]
            with open(savefile, 'a', newline='') as sfile:
                writer = csv.writer(sfile)
                writer.writerows([data])
            
            curr_step = 0
            got_flag = 0
            tot_reward = 0
            actions.clear()
            # time.sleep(10) # Sleep for 10 secs
            state = env.reset()

        state = torch.from_numpy(state)
        if torch.cuda.is_available():
            state = state.cuda()
