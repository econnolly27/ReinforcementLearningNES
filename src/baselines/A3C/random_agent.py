import retro
import os
from setup_env import setup_env, MarioDiscretizerSimple, MarioDiscretizerComplex

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Buttons: ['B', None, 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A']

def main():
    # retro.data.Integrations.add_custom_path(os.path.join(SCRIPT_DIR, "retro_integration"))
    # print(retro.data.list_games(inttype=retro.data.Integrations.CUSTOM_ONLY))
    # print("SMB-JU" in retro.data.list_games(inttype=retro.data.Integrations.CUSTOM_ONLY))
    # obs_type = retro.Observations.IMAGE # or retro.Observations.RAM
    # env = retro.make("SMB-JU", "Level1-1", record=False, inttype=retro.data.Integrations.CUSTOM_ONLY, obs_type=obs_type)

    env = setup_env("SMB-JU", "Level1-1")
    env = MarioDiscretizerSimple(env)
    verbosity = 2
    try:
        while True:
            ob = env.reset()
            t = 0
            totrew = [0] * 1
            while True:
                ac_list = []
                ac_list.append(env.action_space.sample())
                ac_list.append(env.action_space.sample())
                print(ac_list)
                ob, rew, done, info = env.step(ac_list)
                t += 1
                if t % 10 == 0:
                    if verbosity > 1:
                        infostr = ''
                        if info:
                            infostr = ', info: ' + ', '.join(['%s=%i' % (k, v) for k, v in info.items()])
                        print(('t=%i' % t) + infostr)
                    env.render()
                
                rew = [rew]
                for i, r in enumerate(rew):
                    totrew[i] += r
                    if verbosity > 0:
                        if r > 0:
                            print('t=%i p=%i got reward: %g, current reward: %g' % (t, i, r, totrew[i]))
                        if r < 0:
                            print('t=%i p=%i got penalty: %g, current reward: %g' % (t, i, r, totrew[i]))
                if done:
                    try:
                        if verbosity >= 0:
                            print("done! total reward: time=%i, reward=%d" % (t, totrew[0]))
                            input("press enter to continue")
                            print()
                        else:
                            input("")
                    except EOFError:
                        exit(0)
                    break
    except KeyboardInterrupt:
        exit(0)


if __name__ == "__main__":
    main()
