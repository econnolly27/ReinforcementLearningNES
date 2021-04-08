"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import os
import argparse
import torch
from src.env import create_train_env
from src.model import PPO
from src.env import MultipleEnvironments
from src.helpers import JoypadSpace, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY, flag_get
import torch.nn.functional as F

os.environ['DISPLAY'] = ':1'
os.environ['OMP_NUM_THREADS'] = '1'

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

    opt.saved_path = os.getcwd() + '/gradius/PPO/' + opt.saved_path

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
    env = create_train_env(opt.world, opt.stage, actions)
   # env = MultipleEnvironments(opt.world, opt.stage, opt.action_type, 1)

    model = PPO(env.observation_space.shape[0], len(actions))
    print(os.getcwd())
    if torch.cuda.is_available():
        #model.load_state_dict(torch.load("trained_models/abc"))
        model.load_state_dict(torch.load("{}/PPO_gradius_{}_{}".format(opt.saved_path, opt.model_world, opt.model_stage)))
        model.cuda()
    else:
        model.load_state_dict(torch.load("{}/PPO_gradius_{}_{}".format(opt.saved_path, opt.model_world, opt.model_stage),
                                         map_location=lambda storage, loc: storage))
    model.eval()
    state = torch.from_numpy(env.reset())
    while True:
        if torch.cuda.is_available():
            state = state.cuda()
        logits, value = model(state)
        policy = F.softmax(logits, dim=1)
        action = torch.argmax(policy).item()
        state, reward, done, info = env.step(action)
        state = torch.from_numpy(state)
        env.render()
        if flag_get(info):
            print("World {} stage {} completed".format(opt.world, opt.stage))
            break


if __name__ == "__main__":
    opt = get_args()
    test(opt)
