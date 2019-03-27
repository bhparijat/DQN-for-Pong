import numpy as np
import random
from math import sqrt, log


def random_argmax(values):
    max_value = max(values)
    best_indices = [index for index, value in enumerate(values) if value == max_value]
    return random.choice(best_indices)


def ucb(visits, value, parent_visits, exploration):
    return value + exploration * sqrt(log(parent_visits)/visits)


def rollout(environment, agent, search_horizontal, gamma):
    total_reward = 0.
    g = 1.
    step = 0

    if hasattr(agent, 'reset_bufs'):
        agent.reset_bufs()

    while True:
        frame = environment.env.get_image()
        action = agent.choose_action(frame)
        # action = environment.action_space.sample()
        _, reward, terminal, _ = environment.step(action)
        total_reward = total_reward + reward * g
        g = g * gamma
        step = step + 1
        if terminal or 0 < search_horizontal <= step:
            break

    return total_reward


def moving_average(value, visits, reward):
    return (visits * value + reward) / (visits + 1)  # visits + 1 because it is visited one more time now


def update_values(reward, node, gamma):
    """
    Recursively update the parent values with reward
    :param reward:
    :param node:
    :param gamma:
    :return:
    """
    while node is not None:
        node.value = moving_average(node.value, node.visits, reward)
        node.visits = node.visits + 1

        reward = node.reward + gamma * reward

        if node.parent:
            node.parent.action_values[node.parent_action] = \
                moving_average(node.parent.action_values[node.parent_action],
                               node.parent.action_visits[node.parent_action],
                               reward)
            node.parent.action_visits[node.parent_action] = node.parent.action_visits[node.parent_action] + 1

        node = node.parent


def sample(environment, agent, node, search_horizontal, gamma, exploration):
    breadth = 1
    node, leaf = node.selection(environment, exploration)

    while not leaf and breadth < search_horizontal:
        breadth = breadth + 1
        node, leaf = node.selection(environment, exploration)

    reward = node.value
    if leaf and not node.terminal:
        reward = rollout(environment, agent, search_horizontal, gamma)

    update_values(reward, node, gamma)


def uct_action(environment, agent, node, simulation_steps, search_horizontal, gamma, exploration):
    for _ in range(simulation_steps):
        sample(environment, agent, node, search_horizontal, gamma, exploration)

    return node.best_action()


class Node:
    num_actions = 0  # branch factor

    def __init__(self, state, terminal=0, reward=0, parent=None, parent_action=None):
        self.state = state  # state of the node
        self.terminal = terminal  # if the node is terminal
        self.reward = reward  # immediate reward
        self.parent = parent  # parent
        self.parent_action = parent_action  # action taken by the parent
        self.children = [None for _ in range(self.num_actions)]  # child for each action
        self.action_visits = [0 for _ in range(self.num_actions)]  # visits of each child
        self.action_values = [0 for _ in range(self.num_actions)]  # values of each child
        self.value = 0.  # value of the node
        self.visits = 0  # number of visits
        self.actions_count = 0

    def best_action(self):
        return random_argmax(self.action_visits)

    def add_child(self, ind, child):
        if self.children[ind] is None:
            self.actions_count = self.actions_count + 1
        self.children[ind] = child

    def selection(self, environment, exploration):
        # restore state
        environment.restore_state(self.state)

        # explore leaf nodes if possible
        if self.actions_count < self.num_actions:
            leaf = True
            action = random.choice([action for action in range(self.num_actions) if self.children[action] is None])
            frame, reward, terminal, _ = environment.step(action)
            state = environment.clone_state()

            node = Node(state, terminal, reward, self, action)
            self.add_child(action, ({tuple(frame): node}))
        else:
            if exploration == -1.:
                action_values = [(value - np.mean(self.action_values)) / (np.std(self.action_values) + 1e-8)
                                 for value in self.action_values]
                exploration = sqrt(2)
            elif exploration == -2.:
                action_values = (np.asarray(self.action_values) - np.min(self.action_values)) \
                                / (np.max(self.action_values) - np.min(self.action_values) + 1e-8)
                exploration = sqrt(2)
            elif exploration >= 0:
                action_values = self.action_values
            else:
                raise ValueError("Illegal exploration value of %g" % exploration)

            ucb_values = [ucb(self.action_visits[action], action_values[action], self.visits, exploration)
                          for action in range(self.num_actions)]
            action = random_argmax(ucb_values)

            frame, reward, terminal, _ = environment.step(action)
            if tuple(frame) in self.children[action]:
                node = self.children[action][tuple(frame)]
                node.reward = reward
                node.terminal = terminal
                leaf = False
            else:
                state = environment.clone_state()
                node = Node(state, terminal, reward, self, action)
                self.add_child(action, ({tuple(frame): node}))
                leaf = True

        return node, leaf
