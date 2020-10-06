import gym
import retro
import os
import numpy as np
from PIL import Image
from gym import spaces
from collections import deque
import cv2

SCRIPT_DIR = os.getcwd() #os.path.dirname(os.path.abspath(__file__))


# Taken from: https://gitlab.cs.duke.edu/mark-nemecek/vel/-/blob/cfa17ddd8c328331076b3992449665ccd2471bd3/vel/openai/baselines/common/atari_wrappers.py
class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(self.height, self.width, 1), dtype=np.uint8)

    def observation(self, frame):
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        # return frame[:, :, None]
        frame = Image.fromarray(frame).convert('L').resize((self.width, self.height))
        # self._display_last_frame(frame)
        # frame = np.array(frame).astype(np.float32).reshape(1, self.width, self.height) / 255
        frame = np.array(frame).astype(np.float32).reshape(1, self.width, self.height) / 255
        return frame

    def _display_last_frame(self, img):
        img.show()


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.

        Returns lazy array, which is much more memory efficient.

        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        # self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1], shp[2] * k), dtype=np.uint8)
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1], shp[2] * k), dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.

        This object should only be converted to numpy array before being passed to the model.

        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=0)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]


# Taken from: https://github.com/openai/retro/blob/master/retro/examples/discretizer.py
class Discretizer(gym.ActionWrapper):
    """
    Wrap a gym environment and make it use discrete actions.

    Args:
        combos: ordered list of lists of valid button combinations
    """

    def __init__(self, env, combos):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiBinary)
        buttons = env.unwrapped.buttons
        self._decode_discrete_action = []
        for combo in combos:
            arr = [False] * env.action_space.n # np.array([False] * env.action_space.n)
            for button in combo:
                arr[buttons.index(button)] = True
            self._decode_discrete_action.append(arr)

        self.action_space = gym.spaces.Discrete(len(self._decode_discrete_action))

    def action(self, act):
        if type(act) is list:
            out = np.zeros((self.unwrapped.action_space.n,), dtype=bool) # [0] * self.unwrapped.action_space.n
            for a in act:
                dec_act = self._decode_discrete_action[a].copy()
                out += dec_act
        else:
            out = self._decode_discrete_action[act].copy()
        return out


# Define classes per game per buttons combo
class MarioDiscretizerSimple(Discretizer):
    """
    Use Mario Bros specific discrete actions
    based on https://github.com/openai/retro-baselines/blob/master/agents/sonic_util.py
    Buttons: ['B', None, 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A']
    """
    def __init__(self, env):
        combo_list = [[None], ['B'], ['A'], ['LEFT'], ['RIGHT']]
        super().__init__(env=env, combos=combo_list)

class MarioDiscretizerComplex(Discretizer):
    """
    Use Mario Bros specific discrete actions
    based on https://github.com/openai/retro-baselines/blob/master/agents/sonic_util.py
    Buttons: ['B', None, 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A']
    """
    def __init__(self, env):
        # combo_list = [[None],['RIGHT'],['RIGHT', 'A'],['RIGHT', 'B'],['RIGHT', 'A', 'B'],['A'], ['LEFT'],['LEFT', 'A'],['LEFT', 'B'],['LEFT', 'A', 'B'],['DOWN'],['UP']]
        combo_list = [[None],['RIGHT'],['RIGHT', 'A'],['RIGHT', 'B'],['RIGHT', 'A', 'B'],['A']]
        super().__init__(env=env, combos=combo_list)


def setup_env(env_id, level_id):
    retro.data.Integrations.add_custom_path(os.path.join(SCRIPT_DIR, "retro_integration"))

    print(retro.data.list_games(inttype=retro.data.Integrations.CUSTOM_ONLY))
    print(env_id in retro.data.list_games(inttype=retro.data.Integrations.CUSTOM_ONLY))
    
    obs_type = retro.Observations.IMAGE # or retro.Observations.RAM
    env = retro.make(env_id, level_id, record=False, inttype=retro.data.Integrations.CUSTOM_ONLY, obs_type=obs_type)

    env = WarpFrame(env)
    # env = FrameStack(env, 4)

    return env

# x=setup_env("SMB-JU", "Level1-1")