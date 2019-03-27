import cv2
import os
import pickle
import tarfile

import numpy as np

kwargs = {
        'height': 50,
        'width': 50,
        'downsample': 0,
        'min_score': np.inf
    }

def rgb2gray(frame, average='mean'):
    if average == 'cv2':
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = frame.astype(np.float32)
    elif average == 'mean':
        frame = frame.astype(np.float32)
        frame = frame.mean(axis=2)
    else:
        raise NotImplementedError("wrong average type: %s" % average)

    frame *= (1.0 / 255.0)
    frame -= 0.5
    return frame


def make_color_state(frame):
    frame = np.rollaxis(frame, 3, 1)
    frame = frame.reshape([-1] + list(frame.shape[-2:]))
    frame = frame.astype(np.float32)
    frame *= (1.0 / 255.0)
    return frame


def make_state(frame, buffer, height=84, width=84, downsample=None, make_gray=True, average='mean'):
    frame = resize(frame, height, width, downsample)
    if make_gray:
        frame = rgb2gray(frame, average)
    buffer.append(frame)
    while len(buffer) < buffer.maxlen:
        buffer.append(frame)

    frame = np.array(buffer)
    if not make_gray:
        frame = make_color_state(frame)
    return np.expand_dims(frame, 0)


def resize(frame, id, height, width, downsample):
    # print("shape is in resize frame", frame.shape, id)
    h, w = frame.shape[:2]

    if downsample > 0:
        width = int(w / downsample)
        height = int(h / downsample)
    else:
        downsample = 0.5 * w / width + 0.5 * h / height

    if downsample > 4:
        frame = cv2.resize(frame, (width * 2, height * 2))
    return cv2.resize(frame, (width, height))


# def preprocess_run(run, **kwargs):
#     if run['reward'] >= kwargs['min_score']:
#         run['frames'] = [resize(frame, **kwargs) for frame in run['frames']]


def is_not_pickle_file(file_name):
    # true if the file name end with .pkl
    return not file_name.endswith('.pkl')


# def load_runs(dirs, height=84, width=84, downsample=None, min_score=np.inf, **kwargs):
def load_runs(save_dir):
    files = os.listdir(save_dir)

    count = 0
    pos_rewards = 0
    dataset = []
    # lf = None
    print("length of files is", len(files))
    for i, file_name in enumerate(files):
        if is_not_pickle_file(file_name):
            continue

        frame_path = os.path.join(save_dir, file_name)
        with open(frame_path, 'rb') as f:
            f.seek(0)
            # print(i)
            try:
                data = pickle.load(f)
            except pickle.UnpicklingError:
                continue
            # print(data.keys())
            count = count + len(data['frames'])
            frames = data['frames']
            actions = np.array(data['actions'])
            # actions = np.newaxis(actions)
            for i, frame in enumerate(frames):
                frame = np.swapaxes(frame, 0, 2)
                dataset.append((frame, actions[i]))
                # lf = frame
            if data['reward'] > 0:
                pos_rewards = pos_rewards + 1
                # print(data['reward'])

    return dataset
