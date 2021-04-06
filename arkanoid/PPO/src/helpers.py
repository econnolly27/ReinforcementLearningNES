"""An environment wrapper to convert binary to discrete action space."""
"""An environment wrapper to convert binary to discrete action space."""
import gym
from gym import Env
from gym import Wrapper

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
