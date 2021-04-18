"""
@author: Viet Nguyen <nhviet1009@gmail.com>
From: https://github.com/uvipen/Super-mario-bros-PPO-pytorch

Modified for Benchmarking Reinforcement Learning Algorithms in NES Games by Erin-Louise Connolly
"""
import os
import argparse
import torch
from src.env import create_train_env
from src.model import PPO
from src.env import MultipleEnvironments
from src.helpers import JoypadSpace, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
import torch.nn.functional as F
import time
import csv
os.environ['DISPLAY'] = ':1'
os.environ['OMP_NUM_THREADS'] = '1'

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

    opt.saved_path = os.getcwd() + '/spaceinvaders/PPO/' + opt.saved_path
    savefile = opt.saved_path + '/spaceinvaders_ppo_test' + opt.timestr + '.csv'
    print(savefile)
    title = ['Score']
    with open(savefile, 'w', newline='') as sfile:
        writer = csv.writer(sfile)
        writer.writerow(title)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    if opt.action_type == "right":
        actions = RIGHT_ONLY
    elif opt.action_type == "simple":
        actions = SIMPLE_MOVEMENT
    else:
        actions = COMPLEX_MOVEMENT
    env = create_train_env(1, 1, actions)
   # env = MultipleEnvironments(opt.world, opt.stage, opt.action_type, 1)

    model = PPO(env.observation_space.shape[0], len(actions))
    print(os.getcwd())
    if torch.cuda.is_available():
        #model.load_state_dict(torch.load("trained_models/abc"))
        model.load_state_dict(torch.load("{}/ppo_spaceinvaders".format(opt.saved_path)))
        model.cuda()
    else:
        model.load_state_dict(torch.load("{}/ppo_spaceinvaders".format(opt.saved_path),
                                         map_location=lambda storage, loc: storage))
    model.eval()
    state = torch.from_numpy(env.reset())
    scores= []
    while True:
        if torch.cuda.is_available():
            state = state.cuda()
        logits, value = model(state)
        policy = F.softmax(logits, dim=1)
        action = torch.argmax(policy).item()
        state, reward, done, info = env.step(action)
        state = torch.from_numpy(state)
        scores.append(info['score']*10)
        data = [info['score']*10]
        with open(savefile, 'a', newline='') as sfile:
            writer = csv.writer(sfile)
            writer.writerows([data])

        env.render()
        if done: 
            #scores.append(info['score'])
            print(max(scores))
            break


if __name__ == "__main__":
    opt = get_args()
    test(opt)
