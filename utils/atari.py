import json
import os

import atari_py
import numpy as np
from gym import Wrapper, utils, spaces, error
from gym.envs.atari import AtariEnv


def get_hashable_state(state):
    state.flags.writeable = False
    return state.ravel().data


class Atari(AtariEnv):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, game='pong', obs_type='ram', frameskip=(2, 5), repeat_action_probability=0.,
                 full_action_space=False):
        """Frameskip should be either a tuple (indicating a random range to
        choose from, with the top value exclude), or an int."""

        utils.EzPickle.__init__(self, game, obs_type, frameskip, repeat_action_probability)
        assert obs_type in ('ram', 'image')

        self.game_path = atari_py.get_game_path(game)
        if not os.path.exists(self.game_path):
            raise IOError('You asked for game %s but path %s does not exist'%(game, self.game_path))
        self._obs_type = obs_type
        self.frameskip = frameskip
        self.ale = atari_py.ALEInterface()
        self.viewer = None

        # Tune (or disable) ALE's action repeat:
        # https://github.com/openai/gym/issues/349
        assert isinstance(repeat_action_probability, (float, int)), "Invalid repeat_action_probability: {!r}".format(repeat_action_probability)
        self.ale.setFloat('repeat_action_probability'.encode('utf-8'), repeat_action_probability)

        self.seed()

        (screen_width, screen_height) = self.ale.getScreenDims()
        self._buffer = np.empty((screen_height, screen_width, 3), dtype=np.uint8)

        self._action_set = (self.ale.getLegalActionSet() if full_action_space
                            else self.ale.getMinimalActionSet())
        self.action_space = spaces.Discrete(len(self._action_set))

        (screen_width,screen_height) = self.ale.getScreenDims()
        if self._obs_type == 'ram':
            self.observation_space = spaces.Box(low=0, high=255, dtype=np.uint8, shape=(128,))
        elif self._obs_type == 'image':
            self.observation_space = spaces.Box(low=0, high=255, shape=(screen_height, screen_width, 3), dtype=np.uint8)
        else:
            raise error.Error('Unrecognized observation type: {}'.format(self._obs_type))

    def get_image(self):
        return self.ale.getScreenRGB(self._buffer).copy()


class AtariUCT(Wrapper):
    def __init__(self, env, act_repeats):
        super(AtariUCT, self).__init__(env)
        assert act_repeats >= 1
        self.act_repeats = int(act_repeats)

    def reset(self):
        observation = self.env.reset()
        return get_hashable_state(observation)

    def step(self, action):
        total_reward = 0
        observation, reward, terminal, info = [None] * 4

        for _ in range(self.act_repeats):
            observation, reward, terminal, info = self.env.step(action)
            total_reward = total_reward + reward
            if terminal:
                break

        return get_hashable_state(observation), total_reward, terminal, info

    def clone_state(self):
        return self.env.ale.cloneState()

    def restore_state(self, state):
        return self.env.ale.restoreState(state)


def create_environment(game, version, act_repeats):
    # load config
    with open("resources/game_config.json", 'r') as f:
        game_config = json.loads(f.read())
    params = game_config['VERSIONS'][version]

    env = Atari(game=game, **params)
    return AtariUCT(env, act_repeats)
