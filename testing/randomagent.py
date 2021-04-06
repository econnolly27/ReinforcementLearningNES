"""
Gym-Retro Random-agent
"""
import argparse
#import gym
#from gym import spaces
import retro
import os
os.environ['DISPLAY'] = ':1'

parser = argparse.ArgumentParser()
#parser.add_argument('--game', default='SMB-JU', help='the name or path for the game to run')
parser.add_argument('--state', default ='Level1-1', help='the initial state file to load, minus the extension')
parser.add_argument('--scenario', '-s', default='scenario', help='the scenario file to load, minus the extension')
parser.add_argument('--record', '-r', action='store_true', help='record bk2 movies')
parser.add_argument('--verbose', '-v', action='count', default=2, help='increase verbosity (can be specified multiple times)')
parser.add_argument('--quiet', '-q', action='count', default=0, help='decrease verbosity (can be specified multiple times)')
parser.add_argument('--players', '-p', type=int, default=1, help='number of players/agents (default: 1)')
args = parser.parse_args()

ENV_NAME = 'PacManNamco-Nes'
SCRIPT_DIR = os.getcwd()
LVL_ID = 'Level1'
retro.data.Integrations.add_custom_path(os.path.join(SCRIPT_DIR, "retro_integration"))
print(retro.data.list_games(inttype=retro.data.Integrations.CUSTOM_ONLY))
print(ENV_NAME in retro.data.list_games(inttype=retro.data.Integrations.CUSTOM_ONLY))

obs_type = retro.Observations.IMAGE
env = retro.make(ENV_NAME, LVL_ID, record=False, inttype=retro.data.Integrations.CUSTOM_ONLY, obs_type=obs_type)
verbosity = args.verbose - args.quiet
try:
    while True:
        ob = env.reset()
        t = 0
        totrew = [0] * args.players
        while True:
            ac = env.action_space.sample()
            ob, rew, done, info = env.step(ac)
            t += 1
            if t % 10 == 0:
                if verbosity > 1:
                    infostr = ''
                    if info:
                        infostr = ', info: ' + ', '.join(['%s=%i' % (k, v) for k, v in info.items()])
                    print(('t=%i' % t) + infostr)
                env.render()
            if args.players == 1:
                rew = [rew]
            for i, r in enumerate(rew):
                totrew[i] += r
                if verbosity > 0:
                    if r > 0:
                        print('t=%i p=%i got reward: %g, current reward: %g' % (t, i, r, totrew[i]))
                    if r < 0:
                        print('t=%i p=%i got penalty: %g, current reward: %g' % (t, i, r, totrew[i]))
            if done:
                env.reset()
                env.render()
                try:
                    if verbosity >= 0:
                        if args.players > 1:
                            print("done! total reward: time=%i, reward=%r" % (t, totrew))
                        else:
                            print("done! total reward: time=%i, reward=%d" % (t, totrew[0]))
                        input("press enter to continue")
                        print()
                    else:
                        input("")
                except EOFError:
                    exit(0)
                #break
except KeyboardInterrupt:
    exit(0)