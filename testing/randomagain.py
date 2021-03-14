import retro
import os

os.environ['DISPLAY'] = ':1'


ENV_NAME = 'SMB-JU'
SCRIPT_DIR = os.getcwd()
LVL_ID = 'Level1-1'
retro.data.Integrations.add_custom_path(os.path.join(SCRIPT_DIR, "retro_integration"))
print(retro.data.list_games(inttype=retro.data.Integrations.CUSTOM_ONLY))
print(ENV_NAME in retro.data.list_games(inttype=retro.data.Integrations.CUSTOM_ONLY))

obs_type = retro.Observations.IMAGE
env = retro.make(ENV_NAME, LVL_ID, record=False, inttype=retro.data.Integrations.CUSTOM_ONLY, obs_type=obs_type)
env=env.unwrapped
while True:
    ob=env.reset()
    while True:
        action = env.action_space.sample()
        ob, reward, done, info = env.step(action)
        env.render()
        if done:
            env.reset()
        env.close()