from setup_env import setup_env, MarioDiscretizerComplex, MarioDiscretizerSimple
from model import ActorCritic
import os
import time
from collections import deque
import csv
import numpy as np
from itertools import count

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

# def preprocesing(img):
#     img_out = Image.fromarray(img).convert('L').resize((84,84))
#     # img_out.show()
#     img_out = np.array(img_out).astype(np.float32).reshape(1,84,84)/255
#     return img_out

def train(rank, args, shared_model, counter, lock, optimizer=None, select_sample=True, debug=False):

    FloatTensor = torch.cuda.FloatTensor if args.use_cuda else torch.FloatTensor

    env = setup_env(args.env_name, args.lvl_id)
    if args.action_space == 0:
        env = MarioDiscretizerSimple(env)
    elif args.action_space == 1:
        env = MarioDiscretizerComplex(env)

    model = ActorCritic(env.observation_space.shape[2], env.action_space.n)

    if args.use_cuda:
        model.cuda()

    if optimizer is None:
        optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)

    model.train()

    state = env.reset()

    savefile = os.getcwd() + '/baselines/A3C/save/mario_curves_train.csv'
    title = ['Process', 'Episodes', 'Time', 'Steps', 'TotalReward', 'EpisodeLength', "Loss", "gamemode", "status", "lives"]
    with open(savefile, 'a', newline='') as sfile:
        writer = csv.writer(sfile)
        writer.writerow(title)    

    # start_time = time.time()
    done = True
    episode_length = 0
    reward_sum = 0
    no_episodes = 0
    final_loss = 0
    for num_iter in count():
        ep_start_time = time.time()
        if rank == 0:
            if num_iter % args.save_interval == 0 and num_iter > 0:
                print ("Saving model at :" + args.save_path)            
                torch.save(shared_model.state_dict(), args.save_path)

        if num_iter % (args.save_interval * 2.5) == 0 and num_iter > 0 and rank == 1:    # Second saver in-case first processes crashes 
            print ("Saving model for process 1 at :" + args.save_path)            
            torch.save(shared_model.state_dict(), args.save_path)
        
        model.load_state_dict(shared_model.state_dict())
        if done:
            cx = Variable(torch.zeros(1, 512)).type(FloatTensor)
            hx = Variable(torch.zeros(1, 512)).type(FloatTensor)
        else:
            cx = Variable(cx.data).type(FloatTensor)
            hx = Variable(hx.data).type(FloatTensor)

        values = []
        log_probs = []
        rewards = []
        entropies = []
        for step in range(args.num_steps):
            episode_length += 1
            state_m = np.array(state).astype(np.float32)
            state_m = torch.from_numpy(state_m)
            state_inp = Variable(state_m.unsqueeze(0)).type(FloatTensor)
            value, logit, (hx, cx) = model((state_inp, (hx, cx)))
            prob = F.softmax(logit, dim=-1)
            log_prob = F.log_softmax(logit, dim=-1)
            entropy = -(log_prob * prob).sum(-1, keepdim=True)
            entropies.append(entropy)
            
            if select_sample:
                action = prob.multinomial(num_samples=1).data
            else:
                action = prob.max(-1, keepdim=True)[1].data

            log_prob = log_prob.gather(-1, Variable(action))
            
            action_out = action.to(torch.device("cpu"))

            ac = action_out.numpy()[0][0].tolist()
            state, reward, done, info = env.step(ac)
            done = done or episode_length >= args.max_episode_length
            # if rank == 0:
            #     env.render()

            reward = max(min(reward, 1), -1)

            if not debug:
                with lock:
                    counter.value += 1
                
            # env.locked_levels = [False] + [True] * 31
            values.append(value)
            log_probs.append(log_prob)
            reward_sum += reward
            rewards.append(0.001* reward)

            if done:
                #env.change_level(0)
                print ("Process {} has completed.".format(rank))
                print("Episode length: {0} (length rewards: {1})".format(episode_length, len(rewards)))
                final_episode_length = episode_length
                episode_length = 0
                break

        R = torch.zeros(1, 1)
        if not done:
            state_nd = np.array(state).astype(np.float32)
            state_nd = torch.from_numpy(state_nd)
            state_inp = Variable(state_nd.unsqueeze(0)).type(FloatTensor)
            value, _, _ = model((state_inp, (hx, cx)))
            R = value.data

        values.append(Variable(R).type(FloatTensor))
        policy_loss = 0
        value_loss = 0
        R = Variable(R).type(FloatTensor)
        gae = torch.zeros(1, 1).type(FloatTensor)

        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            delta_t = rewards[i] + args.gamma * \
                values[i + 1].data - values[i].data
            gae = gae * args.gamma * args.tau + delta_t

            policy_loss = policy_loss - \
                log_probs[i] * Variable(gae).type(FloatTensor) - args.entropy_coef * entropies[i]

        total_loss = policy_loss + args.value_loss_coef * value_loss
        optimizer.zero_grad()

        (total_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        final_loss += total_loss.data.cpu().numpy()[0][0]

        ensure_shared_grads(model, shared_model)
        optimizer.step()

        if done:
            no_episodes += 1

            if info["gameMode"] == 2:
                flagQ = "Yes"
            else:
                flagQ = "No"

            if not debug:
                data = [rank, no_episodes, time.time() - ep_start_time, counter.value, reward_sum, final_episode_length, final_loss, info["gameMode"], info['status'], info['lives']]
            else:
                data = [no_episodes, time.time() - ep_start_time, 100000, reward_sum, final_episode_length,  final_loss, info["gameMode"], info['status'], info['lives']]
            reward_sum = 0
            final_loss = 0
            state = env.reset()
                    
            with open(savefile, 'a', newline='') as sfile:
                writer = csv.writer(sfile)
                writer.writerows([data])
    

def test(rank, args, shared_model, counter):

    FloatTensor = torch.cuda.FloatTensor if args.use_cuda else torch.FloatTensor

    env = setup_env(args.env_name, args.lvl_id)
    if args.action_space == 0:
        env = MarioDiscretizerSimple(env)
    elif args.action_space == 1:
        env = MarioDiscretizerComplex(env)

    model = ActorCritic(env.observation_space.shape[2], env.action_space.n)
    if args.use_cuda:
        model.cuda()
    model.eval()

    state = env.reset()

    reward_sum = 0
    done = True
    savefile = os.getcwd() + '/baselines/A3C/save/mario_curves_test.csv'

    title = ['Episode', 'Time', 'Steps', 'TotalReward', 'EpisodeLength', "Flag?"]
    with open(savefile, 'a', newline='') as sfile:
        writer = csv.writer(sfile)
        writer.writerow(title)    

    start_time = time.time()

    # actions = deque(maxlen=4000)
    episode_length = 0
    no_episodes = 0
    with torch.no_grad():
        while True:
            episode_length += 1
            ep_start_time = time.time()
            if done:
                print("loading shared model in test...")
                model.load_state_dict(shared_model.state_dict())
                cx = Variable(torch.zeros(1, 512)).type(FloatTensor)
                hx = Variable(torch.zeros(1, 512)).type(FloatTensor)

            else:
                cx = Variable(cx.data).type(FloatTensor)
                hx = Variable(hx.data).type(FloatTensor)

            state_m = np.array(state).astype(np.float32)
            state_m = torch.from_numpy(state_m)
            state_inp = Variable(state_m.unsqueeze(0)).type(FloatTensor)
            _, logit, (hx, cx) = model((state_inp, (hx, cx)))
            prob = F.softmax(logit, dim=-1)
            action = prob.max(-1, keepdim=True)[1].data
            action_out = action.to(torch.device("cpu"))

            ac = action_out.numpy()[0][0].tolist()
            state, reward, done, info = env.step(ac)
            env.render()

            done = done or episode_length >= args.max_episode_length
            reward_sum += reward

            # actions.append(action[0][0])
            # if actions.count(actions[0]) == actions.maxlen:
            #     done = True

            if done:
                no_episodes += 1
                print("Episode {}, Time {}, num steps {}, FPS {:.0f}, episode reward {}, episode length {}".format(no_episodes,
                    time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)), 
                    counter.value, counter.value / (time.time() - start_time),
                    reward_sum, episode_length))

                if info["gameMode"] == 2:
                    flagQ = "Yes"
                else:
                    flagQ = "No"
                
                data = [no_episodes, time.time() - ep_start_time, counter.value, reward_sum, episode_length, flagQ]
                
                with open(savefile, 'a', newline='') as sfile:
                    writer = csv.writer(sfile)
                    writer.writerows([data])
                
                reward_sum = 0
                episode_length = 0
                time.sleep(1)
                # env.locked_levels = [False] + [True] * 31
                #env.change_level(0)
                state = env.reset()