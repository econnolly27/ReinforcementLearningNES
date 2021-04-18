"""
@author: Viet Nguyen <nhviet1009@gmail.com>
From: https://github.com/uvipen/Super-mario-bros-A3C-pytorch

Modified for Benchmarking Reinforcement Learning Algorithms in NES Games by Erin-Louise Connolly
"""

import torch.nn.functional as F
from src.model import ActorCritic
from src.env import create_train_env
import torch
import argparse
import os
import time
import csv
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['DISPLAY'] = ':1'


def get_args():
    timestr = time.strftime("%Y%m%d-%H%M%S")
    parser = argparse.ArgumentParser(
        """Implementation of model described in the paper: Proximal Policy Optimization Algorithms for Contra Nes""")
    parser.add_argument("--action_type", type=str, default="complex")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--output_path", type=str, default="output")
    parser.add_argument("--timestr", type=str, default=timestr)

    args = parser.parse_args()
    return args


def test(opt):
    torch.manual_seed(123)

    opt.saved_path = os.getcwd() + '/spaceinvaders/a3c/' + opt.saved_path
    savefile = opt.saved_path + '/spaceinvaders_a3c_test' + opt.timestr + '.csv'
    print(savefile)
    title = ['Score']
    with open(savefile, 'w', newline='') as sfile:
        writer = csv.writer(sfile)
        writer.writerow(title)
    env, num_states, num_actions = create_train_env(1, 1, opt.action_type)
    model = ActorCritic(num_states, num_actions)
    if torch.cuda.is_available():

        model.load_state_dict(torch.load(
            "{}/a3c_spaceinvaders".format(opt.saved_path)))
        model.cuda()
    else:
        model.load_state_dict(torch.load("{}/a3c_spaceinvaders".format(opt.saved_path),
                                         map_location=lambda storage, loc: storage))
    model.eval()
    state = torch.from_numpy(env.reset())
    done = True
    scores = []

    while True:
        if done:
            h_0 = torch.zeros((1, 512), dtype=torch.float)
            c_0 = torch.zeros((1, 512), dtype=torch.float)
            env.reset()
        else:
            h_0 = h_0.detach()
            c_0 = c_0.detach()
        if torch.cuda.is_available():
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()
            state = state.cuda()

        logits, value, h_0, c_0 = model(state, h_0, c_0)
        policy = F.softmax(logits, dim=1)
        action = torch.argmax(policy).item()
        action = int(action)
        state, reward, done, info = env.step(action)
        state = torch.from_numpy(state)
        scores.append(info['score']*10)
        data = [info['score']*10]
        with open(savefile, 'a', newline='') as sfile:
            writer = csv.writer(sfile)
            writer.writerows([data])
        env.render()
        if done:
            # scores.append(info['score'])
            print(max(scores))
            break


if __name__ == "__main__":
    opt = get_args()
    test(opt)
