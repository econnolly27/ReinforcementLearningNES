"""
Gym-Retro Random-agent
"""
import argparse
#import gym
#from gym import spaces
import retro
import os
import time
import csv

os.environ['DISPLAY'] = ':1'

timestr = time.strftime("%Y%m%d-%H%M%S")
parser = argparse.ArgumentParser()
#parser.add_argument('--game', default='SMB-JU', help='the name or path for the game to run')
parser.add_argument('--state', default ='Level1-1', help='the initial state file to load, minus the extension')
parser.add_argument('--scenario', '-s', default='scenario', help='the scenario file to load, minus the extension')
parser.add_argument('--record', '-r', action='store_true', help='record bk2 movies')
parser.add_argument('--verbose', '-v', action='count', default=2, help='increase verbosity (can be specified multiple times)')
parser.add_argument('--quiet', '-q', action='count', default=0, help='decrease verbosity (can be specified multiple times)')
parser.add_argument('--players', '-p', type=int, default=1, help='number of players/agents (default: 1)')
args = parser.parse_args()

saved_path = os.getcwd() + '/testing/randomagent/'

ENV_NAME = 'Gradius-Nes'
savefile = saved_path + ENV_NAME + timestr + 'random.csv'
print(savefile)
title = ['Score']
with open(savefile, 'w', newline='') as sfile:
        writer = csv.writer(sfile)
        writer.writerow(title)

SCRIPT_DIR = os.getcwd()
LVL_ID = 'Level1'
retro.data.Integrations.add_custom_path(os.path.join(SCRIPT_DIR, "retro_integration"))
print(retro.data.list_games(inttype=retro.data.Integrations.CUSTOM_ONLY))
print(ENV_NAME in retro.data.list_games(inttype=retro.data.Integrations.CUSTOM_ONLY))

obs_type = retro.Observations.IMAGE
env = retro.make(ENV_NAME, LVL_ID, record=False, inttype=retro.data.Integrations.CUSTOM_ONLY, obs_type=obs_type)
verbosity = args.verbose - args.quiet
scores = []
try:
    while True:
        ob = env.reset()
        t = 0
        totrew = [0] * args.players
        while True:
            ac = env.action_space.sample()
            ob, rew, done, info = env.step(ac)
            if ENV_NAME == 'Arkanoid-Nes':
                scores.append(info['score'])
                data = [info['score']]
            else:
                scores.append(info['score']*10)
                data = [info['score']*10]

            with open(savefile, 'a', newline='') as sfile:
                writer = csv.writer(sfile)
                writer.writerows([data])

            t += 1
            if t % 10 == 0:
                env.render()
            if args.players == 1:
                rew = [rew]
            for i, r in enumerate(rew):
                totrew[i] += r
            if done:
                env.reset()
                env.render()
                try:
                    if verbosity >= 0:
                        print(max(scores))
                        input("press enter to continue")
                        print()
                    else:
                        input("")
                except EOFError:
                    exit(0)
                #break
except KeyboardInterrupt:
    exit(0)