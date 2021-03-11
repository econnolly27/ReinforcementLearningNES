"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""

import os

os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import torch
from src.env import create_train_env
from src.model import ActorCritic
from src.helpers import flag_get
import torch.nn.functional as F
os.environ['DISPLAY'] = ':1'

def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of model described in the paper: Proximal Policy Optimization Algorithms for Contra Nes""")
    parser.add_argument("--model_world", type=int, default=1)
    parser.add_argument("--model_stage", type=int, default=1)
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--action_type", type=str, default="complex")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--output_path", type=str, default="output")
    args = parser.parse_args()
    return args


def test(opt):
    torch.manual_seed(123)
    
    opt.saved_path = os.getcwd() + '/mario/a3c/' + opt.saved_path
    env, num_states, num_actions = create_train_env(opt.world, opt.stage, opt.action_type)
    model = ActorCritic(num_states, num_actions)
    if torch.cuda.is_available():
        #model.load_state_dict(torch.load("{}/a3c_super_mario_bros_{}_{}".format(opt.saved_path, opt.world, opt.stage)))

        model.load_state_dict(torch.load("{}/a3c_super_mario_bros_{}_{}".format(opt.saved_path, opt.model_world, opt.model_stage)))
        model.cuda()
    else:
        model.load_state_dict(torch.load("{}/a3c_super_mario_bros_{}_{}".format(opt.saved_path, opt.model_world, opt.model_stage),
                                         map_location=lambda storage, loc: storage))
    model.eval()
    state = torch.from_numpy(env.reset())
    done = True
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
        env.render()
        if flag_get(info):
            print("World {} stage {} completed".format(opt.world, opt.stage))
            break


if __name__ == "__main__":
    opt = get_args()
    test(opt)
