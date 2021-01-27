"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""

import os
import argparse
import torch
from src.env import MultipleEnvironments
from src.env import create_train_env
from src.model import ActorCritic
from src.optimizer import GlobalAdam
from src.process import local_train, local_test
import torch.multiprocessing as _mp
import shutil,csv,time
from src.helpers import flag_get
from datetime import datetime

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
    parser.add_argument('--gamma', type=float, default=0.9, help='discount factor for rewards')
    parser.add_argument('--tau', type=float, default=1.0, help='parameter for GAE')
    parser.add_argument('--beta', type=float, default=0.01, help='entropy coefficient')
    parser.add_argument("--num_local_steps", type=int, default=50)
    parser.add_argument("--num_global_steps", type=int, default=5e6)
    parser.add_argument("--num_processes", type=int, default=4)
    parser.add_argument("--save_interval", type=int, default=500, help="Number of steps between savings")
    parser.add_argument("--max_actions", type=int, default=200, help="Maximum repetition steps in test phase")
    parser.add_argument("--log_path", type=str, default="tensorboard/a3c_super_mario_bros")
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

    opt.saved_path = os.getcwd() + '/donkeykong/a3c/' + opt.saved_path


    start_time = time.time()       

    now = datetime.now()

    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)
    start_time = time.time()

    #if os.path.isdir(opt.log_path):
     #   shutil.rmtree(opt.log_path)
    #os.makedirs(opt.log_path)

    if not os.path.isdir(opt.saved_path):
        os.makedirs(opt.saved_path)

    savefile = opt.saved_path + '/a3c_train.csv'
    print(savefile)
    title = ['Loops', 'Steps', 'Time', 'AvgLoss', 'MeanReward', "StdReward", "TotalReward", "Flags"]
    with open(savefile, 'w', newline='') as sfile:
        writer = csv.writer(sfile)
        writer.writerow(title)

    envs = MultipleEnvironments(opt.world, opt.stage, opt.action_type, opt.num_processes)


    mp = _mp.get_context("spawn")

    #env, num_states, num_actions = create_train_env(opt.world, opt.stage, opt.action_type)
   
   
    #global_model = ActorCritic(num_states, num_actions)

    global_model = ActorCritic(envs.num_states, envs.num_actions)
    if torch.cuda.is_available():
        global_model.cuda()
    global_model.share_memory()
    optimizer = torch.optim.Adam(global_model.parameters(), lr=opt.lr)

    # Start test/evaluation model
    #if TEST_ON_THE_GO:
        # evaluate(opt, model, envs.num_states, envs.num_actions)
        #mp = _mp.get_context("spawn")
        #process = mp.Process(target=evaluate, args=(opt, model, envs.num_states, envs.num_actions))
        #process.start()

    if opt.load_from_previous_stage:
        if opt.stage == 1:
            previous_world = opt.world - 1
            previous_stage = 4
        else:
            previous_world = opt.world
            previous_stage = opt.stage - 1
        file_ = "{}/a3c_super_mario_bros_{}_{}".format(opt.saved_path, previous_world, previous_stage)
        if os.path.isfile(file_):
            global_model.load_state_dict(torch.load(file_))

    #optimizer = GlobalAdam(global_model.parameters(), lr=opt.lr)
    processes = []
    for index in range(opt.num_processes):
        if index == 0:
            process = mp.Process(target=local_train, args=(index, opt, global_model, optimizer,envs.num_states,envs.num_actions, True))
        else:
            process = mp.Process(target=local_train, args=(index, opt, global_model, optimizer,envs.num_states,envs.num_actions))
        process.start()
        processes.append(process)
    process = mp.Process(target=local_test, args=(opt.num_processes, opt, global_model,envs.num_states,envs.num_actions))
    process.start()
    processes.append(process)
    for process in processes:
        process.join()

    with open(savefile, 'a', newline='') as sfile:
        writer = csv.writer(sfile)
        writer.writerows([data])
    elapsed_time = time.time() - start_time
    print(elapsed_time)
    #print("Steps: {}. Total loss: {}. Time elapsed: {}".format(tot_steps, total_loss,time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))
    #if check_flag(info):
    #    print("Stage finished")


if __name__ == "__main__":
    opt = get_args()
    train(opt)