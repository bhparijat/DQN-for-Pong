import argparse
import os
import pickle
from datetime import datetime
from multiprocessing import Process
from time import time
from utils.runs import resize
import numpy as np

from utils.agents import RandomAgent, CNNAgent, RNNAgent
from utils.atari import create_environment
from utils.uct import Node, uct_action

parser = argparse.ArgumentParser(description="Run commands",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--env', type=str, default="pong", help="Environment")
parser.add_argument('--version', type=str, default="v0", help="Environment Version")
parser.add_argument('--act_repeats', type=int, default=1, help="Action repeat times")
parser.add_argument('--max_steps', type=int, default=20, help="Maximum number of steps")
parser.add_argument('--num_workers', type=int, default=8, help="Number of concurrent workers")
parser.add_argument('--runs_per_worker', type=int, default=50, help="Number of runs per worker")
parser.add_argument('--save_dir', type=str, default='data', help="Path where to dump collected data")
parser.add_argument('--num_simulations', type=int, default=100,
                    help="Number of simulations for selecting action with rollout policy")
parser.add_argument('--search_horizontal', type=int, default=20, help="Horizontal search for each simulation")
parser.add_argument('--gamma', type=float, default=1.0, help="Factor of reward")
parser.add_argument('--exploration', type=float, default=1., help="Coefficient for exploration part")
parser.add_argument('--prune_tree', default=False, action='store_true',
                    help="Prune tree after choosing action with uct")
parser.add_argument('--rollout_agent_name', type=str, default='random', help="Rollout agent name: [random, cnn, rnn]")
parser.add_argument('--behavior_agent_name', type=str, default='random',
                    help="Behavior agent name: [random, uct, cnn, rnn]")
parser.add_argument('--eps_greedy', type=float, default=0., help="Probability of selecting random action")
parser.add_argument('--save_frequency', type=int, default=10, help="Frequency of saving uct data")
parser.add_argument('--report_frequency', type=int, default=100, help="Frequency of reporting uct progress")


def run(env, version, act_repeats, max_steps, rollout_agent_name, behavior_agent_name, eps_greedy,
        num_simulations, search_horizontal, gamma, exploration, prune_tree, report_frequency, runs_per_worker,
        save_dir, save_frequency, process, height, width, downsample):

    def save_data():
        # print("save dir is ",(save_dir==None), "length of frames",len(frames))
        if save_dir is not None and len(frames) > 0:
            # print("type",type(frames))
            resized_frames = [resize(frame, i+1, height, width, downsample)
                              for i, frame in enumerate(frames)]
            # print(type(resized_frames))
            run_data = {
                'frames': resized_frames,
                'actions': actions,
                'reward': total_reward,
                'action_visits': action_visits,
                'action_values': action_values,
                'rewards': rewards,
                'action_meanings': environment.env.get_action_meanings()
            }
            file_name = os.path.join(save_dir, "run_process_%d_run_%d_steps_%d.pkl"
                                     % (process, iteration, step))
            with open(file_name, 'wb') as file:
                pickle.dump(run_data, file)

            del actions[:]
            del frames[:]
            del action_visits[:]
            del action_values[:]
            del rewards[:]

    environment = create_environment(env, version, act_repeats)
    Node.num_actions = environment.action_space.n

    # assign policy agent
    # rollout agent
    if rollout_agent_name is None or rollout_agent_name == 'random':
        rollout_agent = RandomAgent(environment.action_space.n)
    elif rollout_agent_name == 'cnn':
        rollout_agent = CNNAgent(rollout_agent_name)
    else:
        rollout_agent = RNNAgent(rollout_agent_name)
    # behavior agent
    if behavior_agent_name is None or behavior_agent_name == 'uct':
        behavior_agent = 'uct'
    elif behavior_agent_name == 'random':
        behavior_agent = RandomAgent(environment.action_space.n)
    elif behavior_agent_name == 'cnn':
        behavior_agent = CNNAgent(behavior_agent_name)
    else:
        behavior_agent = RNNAgent(behavior_agent_name)

    for iteration in range(runs_per_worker):
        terminal = False

        environment.reset()
        frame = environment.env.get_image()
        #print(frame.shape)
        #frame = frame.tolist()
        #print("type of frame is:",type(frame))
        node = Node(environment.clone_state())

        total_reward = 0
        step = 0
        start_time = time()
        actions = []
        frames = []
        action_visits = []
        action_values = []
        rewards = []

        while not terminal:
            uct_a = uct_action(environment, rollout_agent, node, num_simulations, search_horizontal, gamma, exploration)

            # choose action
            if np.random.rand() < eps_greedy:
                action = environment.action_space.sample()
            elif behavior_agent == 'uct':
                action = uct_a
            else:
                action = behavior_agent.choose_action(frame)

            if save_dir is not None:
                actions.append(uct_a)
                frames.append(frame)
                action_visits.append(node.action_visits)
                action_values.append(node.action_values)

            # take a step
            environment.restore_state(node.state)
            frame, reward, terminal, _ = environment.step(action)
            # print("size after taking step", frame.shape)
            frame = environment.env.get_image()
            # print("size after taking step", frame.shape)
            if save_dir is not None:
                rewards.append(reward)

            if prune_tree:
                if frame in node.children[action]:
                    node = node.children[action][frame]
                    node.parent = None  # set the parent to none and not the link to this
                else:
                    node = Node(environment.clone_state())
            else:
                node = Node(environment.clone_state())

            total_reward = total_reward + reward
            step = step + 1

            if step % report_frequency == 0:
                print("[process=%d] run=%d steps=%d time=%f total_reward=%f"
                      % (process, iteration, step, time() - start_time, total_reward))
                start_time = time()

            if step % save_frequency == 0:
                save_data()

            if 0 < max_steps < step:
                break

        # last chunk of data
        print("[process=%d] run=%d steps=%d time=%f total_reward=%f"
              % (process, iteration, step, time() - start_time, total_reward))
        save_data()

    environment.close()


def collect_data(env, version, act_repeats, max_steps, rollout_agent_name, behavior_agent_name, eps_greedy,
                 num_simulations, search_horizontal, gamma, exploration, prune_tree, report_frequency, runs_per_worker,
                 num_workers, save_dir, save_frequency, height, width, downsample):
    # create required directories
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # dump data generation info
    generation_info = os.path.join(save_dir, "collect-%s.txt" % datetime.now().strftime('%m%d%Y-%H%M%S'))
    with open(generation_info, 'w') as info_file:
        info_file.write('env: %s\n' % env)
        info_file.write('version: %s\n' % version)
        info_file.write('act_repeats: %d\n' % act_repeats)
        info_file.write('max_steps: %d\n' % max_steps)
        info_file.write('num_workers: %d\n' % num_workers)
        info_file.write('runs_per_worker: %d\n' % runs_per_worker)
        info_file.write('save_dir: %s\n' % save_dir)
        info_file.write('num_simulations: %d\n' % num_simulations)
        info_file.write('search_horizontal: %d\n' % search_horizontal)
        info_file.write('gamma: %f\n' % gamma)
        info_file.write('exploration: %f\n' % exploration)
        info_file.write('prune_tree: %d\n' % prune_tree)
        info_file.write('rollout_agent_name: %s\n' % rollout_agent_name)
        info_file.write('behavior_agent_name: %s\n' % behavior_agent_name)
        info_file.write('eps_greedy: %f\n' % eps_greedy)
        info_file.write('save_frequency: %d\n' % save_frequency)
        info_file.write('report_frequency: %d\n' % report_frequency)

    # collect data in multiple threads
    workers = []
    num_workers = 4
    for i in range(num_workers):
        worker = Process(target=run, args=(
            env, version, act_repeats, max_steps, rollout_agent_name, behavior_agent_name, eps_greedy, num_simulations,
            search_horizontal, gamma, exploration, prune_tree, report_frequency, runs_per_worker, save_dir,
            save_frequency, i, height, width, downsample
        ))
        worker.daemon = True
        worker.start()
        workers.append(worker)

    for worker in workers:
        worker.join()


if __name__ == '__main__':
    args = parser.parse_args()

    kwargs = {
        'height': 50,
        'width': 50,
        'downsample': 0,
        'min_score': np.inf
    }

    collect_data(
        args.env, args.version, args.act_repeats, args.max_steps, args.rollout_agent_name,
        args.behavior_agent_name, args.eps_greedy, args.num_simulations, args.search_horizontal, args.gamma,
        args.exploration, args.prune_tree, args.report_frequency, args.runs_per_worker, args.num_workers, args.save_dir,
        args.save_frequency, kwargs['height'], kwargs['width'], kwargs['downsample']
    )
