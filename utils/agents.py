import cv2
from collections import deque

import numpy as np

from utils.runs import make_state


class BaseAgent:
    def __init__(self, actions_limit, seed=None, not_sample=False):
        self.actions_limit = actions_limit
        self.random_number_generator = np.random.RandomState(seed)
        self.not_sample = not_sample

    def get_probabilities(self, frame):
        raise NotImplementedError

    def choose_action(self, frame, sample=True):
        probabilities = self.get_probabilities(frame)

        if self.not_sample:
            sample = False

        if sample:
            return self.random_number_generator\
                .multinomial(1, probabilities - np.finfo(np.float32).epsneg).argmax()
        else:
            return probabilities.argmax()

    def seed(self, seed):
        self.random_number_generator.seed(seed)

    def reset(self):
        pass


class RandomAgent(BaseAgent):
    def __init__(self, actions_limit):
        super(RandomAgent, self).__init__(actions_limit)

    def get_probabilities(self, frame):
        return np.asarray([1./self.actions_limit for _ in range(self.actions_limit)])

    def choose_action(self, frame, sample=True):
        return self.random_number_generator.randint(self.actions_limit)


class CNNAgent(BaseAgent):
    def __init__(self, model_path, flip_map=None, gray_state=True, **kwargs):
        # load model (pytorch way)
        model = object()
        if flip_map is not None:
            assert model.output_shape[1] == len(flip_map)

        super(CNNAgent, self).__init__(actions_limit=model.output_shape[1], **kwargs)

        self.gray_state = gray_state
        if len(model.input_shape) == 5:
            self.frames_limit = model.input_shape[2]
            self.rnn = True
        else:
            self.frames_limit = model.input_shape[1]
            self.rnn = False

        if not gray_state:
            self.frames_limit /= 3
        self.height, self.width = model.input_shape[2:]
        self.model = model
        self.flip_map = flip_map
        self.reset()

    def reset(self):
        self.model.reset_states()
        self.buffer = deque(maxlen=self.frames_limit)

    def get_probabilities(self, frame):
        if self.flip_map:
            frame = cv2.flip(frame, 1)
        state = make_state(frame, self.buffer, self.height, self.width, make_gray=self.gray_state)

        if self.rnn:
            probabilities = self.model.predict(np.asarray([state]))[0][0]
        else:
            probabilities = self.model.predict(state)[0]

        if self.flip_map:
            return probabilities[self.flip_map]
        return probabilities


class RNNAgent(BaseAgent):
    def __init__(self, model_path, flip_map=None, gray_state=True, **kwargs):
        # load model (pytorch way)
        model = object()
        if flip_map is not None:
            assert model.output_shape[1] == len(flip_map)

        super(RNNAgent, self).__init__(actions_limit=model.output_shape[1], **kwargs)

        self.gray_state = gray_state
        if len(model.input_shape) == 5:
            self.frames_limit = model.input_shape[2]
            self.rnn = True
        else:
            self.frames_limit = model.input_shape[1]
            self.rnn = False

        if not gray_state:
            self.frames_limit /= 3
        self.height, self.width = model.input_shape[2:]
        self.model = model
        self.flip_map = flip_map
        self.reset()

    def reset(self):
        self.model.reset_states()
        self.buffer = deque(maxlen=self.frames_limit)

    def get_probabilities(self, frame):
        if self.flip_map:
            frame = cv2.flip(frame, 1)
        state = make_state(frame, self.buffer, self.height, self.width, make_gray=self.gray_state)

        if self.rnn:
            probabilities = self.model.predict(np.asarray([state]))[0][0]
        else:
            probabilities = self.model.predict(state)[0]

        if self.flip_map:
            return probabilities[self.flip_map]
        return probabilities
