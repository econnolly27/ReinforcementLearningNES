import os
import retrowrapper as rw
import retro

SCRIPT_DIR = os.getcwd() #os.path.dirname(os.path.abspath(__file__))
ENV_NAME = 'SMB-JU'
LVL_ID = 'Level1-1'

if __name__ == "__main__":
    
    retro.data.Integrations.add_custom_path(os.path.join(SCRIPT_DIR, "retro_integration"))
    print(retro.data.list_games(inttype=retro.data.Integrations.CUSTOM_ONLY))
    print(ENV_NAME in retro.data.list_games(inttype=retro.data.Integrations.CUSTOM_ONLY))
    obs_type = retro.Observations.IMAGE # or retro.Observations.RAM
    env = retro.make(ENV_NAME, state=LVL_ID, record=False, inttype=retro.data.Integrations.CUSTOM_ONLY, obs_type=obs_type)
    env.close()

    env1 = rw.RetroWrapper(ENV_NAME, state=LVL_ID, record=False, inttype=retro.data.Integrations.CUSTOM_ONLY, obs_type=obs_type)
    env2 = rw.RetroWrapper(ENV_NAME, state=LVL_ID, record=False, inttype=retro.data.Integrations.CUSTOM_ONLY, obs_type=obs_type)
    _obs = env1.reset()
    _obs = env2.reset()

    done = False
    while not done:
        action = env1.action_space.sample()
        print(action)
        _obs, _rew, done, _info = env1.step(action)
        env1.render()

        action = env2.action_space.sample()
        _obs, _rew, done, _info = env2.step(action)
        env2.render()