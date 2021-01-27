"""An environment wrapper to convert binary to discrete action space."""
"""An environment wrapper to convert binary to discrete action space."""
import gym
from gym import Env
from gym import Wrapper
# import numpy as np

# Following 3 functions come from: gym-super-mario-bros (https://github.com/Kautenja/gym-super-mario-bros)
def _is_world_over(info):
    """Return a boolean determining if the world is over."""
    # 0x0770 contains GamePlay mode:
    # 0 => Demo
    # 1 => Standard
    # 2 => End of world
    return info["gameMode"] == 2

def _is_stage_over(info):
    """Return a boolean determining if the level is over."""
    # RAM addresses for enemy types on the screen
    _ENEMY_TYPE_ADDRESSES = [info["enemyType1"], info["enemyType2"], info["enemyType3"], info["enemyType4"], info["enemyType5"]]

    # enemies whose context indicate that a stage change will occur (opposed to an
    # enemy that implies a stage change wont occur -- i.e., a vine)
    # Bowser = 45 - ram: 0x2D
    # Flagpole = 49 - ram: 0x31
    _STAGE_OVER_ENEMIES = [45, 49]

    # iterate over the memory addresses that hold enemy types
    for address in _ENEMY_TYPE_ADDRESSES:
        # check if the byte is either Bowser (0x2D) or a flag (0x31)
        # this is to prevent returning true when Mario is using a vine
        # which will set the byte at 0x001D to 3
        if address in _STAGE_OVER_ENEMIES:
            # player float state set to 3 when sliding down flag pole
            return info["floatState"] == 3

    return False

def flag_get(info):
    """Return a boolean determining if the agent reached a flag."""
    return _is_world_over(info) or _is_stage_over(info)


class JoypadSpace(Wrapper):
    """An environment wrapper to convert binary to discrete action space."""

    # a mapping of buttons to binary values
    # _button_map = {
    #     'right':  0b10000000,
    #     'left':   0b01000000,
    #     'down':   0b00100000,
    #     'up':     0b00010000,
    #     'start':  0b00001000,
    #     'select': 0b00000100,
    #     'B':      0b00000010,
    #     'A':      0b00000001,
    #     'noop':   0b00000000,
    # }

    _button_list = ['B', 'noop', 'select', 'start', 'up', 'down', 'left', 'right', 'A']

    @classmethod
    def buttons(cls) -> list:
        """Return the buttons that can be used as actions."""
        return list(cls._button_map.keys())

    def __init__(self, env: Env, actions: list):
        """
        Initialize a new binary to discrete action space wrapper.

        Args:
            env: the environment to wrap
            actions: an ordered list of actions (as lists of buttons).
                The index of each button list is its discrete coded value

        Returns:
            None

        """ 
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiBinary)

        self.action_space = gym.spaces.Discrete(len(actions))
        # create the action map from the list of discrete actions
        self._action_map = {}
        self._action_meanings = {}
        # iterate over all the actions (as button lists)
        buttons = self._button_list #list(self._button_map.keys())
        for action, button_list in enumerate(actions):
            # the value of this action's bitmap
            arr = [0] * env.action_space.n #np.array([False] * env.action_space.n)
            # iterate over the buttons in this button list
            for button in button_list:
                arr[buttons.index(button)] = 1
                # byte_action |= self._button_map[button]
            # set this action maps value to the byte action value
            self._action_map[action] = arr
            self._action_meanings[action] = ' '.join(button_list)

        # # create the action map from the list of discrete actions
        # self._action_map = []
        # buttons = env.unwrapped.buttons # < problem is herer!!!!!
        # for combo in actions:
        #     arr = [False] * env.action_space.n #np.array([False] * env.action_space.n)
        #     for button in combo:
        #         arr[buttons.index(button)] = True
        #     self._action_map.append(arr)

        # create the new action space
        # self.action_space = gym.spaces.Discrete(len(self._action_map))

    def step(self, action):
        """
        Take a step using the given action.

        Args:
            action (int): the discrete action to perform

        Returns:
            a tuple of:
            - (numpy.ndarray) the state as a result of the action
            - (float) the reward achieved by taking the action
            - (bool) a flag denoting whether the episode has ended
            - (dict) a dictionary of extra information

        """
        # take the step and record the output
        return self.env.step(self._action_map[action])

    def reset(self):
        """Reset the environment and return the initial observation."""
        return self.env.reset()

    # def get_keys_to_action(self):
    #     """Return the dictionary of keyboard keys to actions."""
    #     # get the old mapping of keys to actions
    #     old_keys_to_action = self.env.unwrapped.get_keys_to_action()
    #     # invert the keys to action mapping to lookup key combos by action
    #     action_to_keys = {v: k for k, v in old_keys_to_action.items()}
    #     # create a new mapping of keys to actions
    #     keys_to_action = {}
    #     # iterate over the actions and their byte values in this mapper
    #     for action, byte in self._action_map.items():
    #         # get the keys to press for the action
    #         keys = action_to_keys[byte]
    #         # set the keys value in the dictionary to the current discrete act
    #         keys_to_action[keys] = action

    #     return keys_to_action

    def get_action_meanings(self):
        """Return a list of actions meanings."""
        actions = sorted(self._action_meanings.keys())
        return [self._action_meanings[action] for action in actions]


"""Static action sets for binary to discrete action space wrappers."""
# actions for the simple run right environment
RIGHT_ONLY = [
    ['noop'],
    ['right'],
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B'],
]


# actions for very simple movement
SIMPLE_MOVEMENT = [
    ['noop'],
    ['right'],
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B'],
    ['A'],
    ['left'],
]


# actions for more complex movement
COMPLEX_MOVEMENT = [
    ['noop'],
    ['right'],
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B'],
    ['A'],
    ['left'],
    ['left', 'A'],
    ['left', 'B'],
    ['left', 'A', 'B'],
    ['down'],
    ['up'],
]



# # Taken from: https://github.com/openai/retro/blob/master/retro/examples/discretizer.py
# class Discretizer(gym.ActionWrapper):
#     """
#     Wrap a gym environment and make it use discrete actions.

#     Args:
#         combos: ordered list of lists of valid button combinations
#     """

#     def __init__(self, env, combos):
#         super().__init__(env)
#         assert isinstance(env.action_space, gym.spaces.MultiBinary)
#         buttons = env.unwrapped.buttons
#         self._decode_discrete_action = []
#         for combo in combos:
#             arr = np.array([False] * env.action_space.n)
#             for button in combo:
#                 arr[buttons.index(button)] = True
#             self._decode_discrete_action.append(arr)

#         self.action_space = gym.spaces.Discrete(len(self._decode_discrete_action))

#     def action(self, act):
#         if type(act) is list:
#             out = np.zeros((self.unwrapped.action_space.n,), dtype=bool) # [0] * self.unwrapped.action_space.n
#             for a in act:
#                 dec_act = self._decode_discrete_action[a].copy()
#                 out += dec_act
#         else:
#             out = self._decode_discrete_action[act].copy()
#         return out

# # Define classes per game per buttons combo
# class MarioDiscretizerSimple(Discretizer):
#     """
#     Use Mario Bros specific discrete actions
#     based on https://github.com/openai/retro-baselines/blob/master/agents/sonic_util.py
#     Buttons: ['B', None, 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A']
#     """
#     def __init__(self, env):
#         combo_list = [[None], ['B'], ['A'], ['LEFT'], ['RIGHT']]
#         super().__init__(env=env, combos=combo_list)

# class MarioDiscretizerComplex(Discretizer):
#     """
#     Use Mario Bros specific discrete actions
#     based on https://github.com/openai/retro-baselines/blob/master/agents/sonic_util.py
#     Buttons: ['B', None, 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A']
#     """
#     def __init__(self, env):
#         # combo_list = [[None],['RIGHT'],['RIGHT', 'A'],['RIGHT', 'B'],['RIGHT', 'A', 'B'],['A'], ['LEFT'],['LEFT', 'A'],['LEFT', 'B'],['LEFT', 'A', 'B'],['DOWN'],['UP']]
#         combo_list = [[None],['RIGHT'],['RIGHT', 'A'],['RIGHT', 'B'],['RIGHT', 'A', 'B'],['A']]
#         super().__init__(env=env, combos=combo_list)