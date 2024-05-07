import os
import random
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dqnFiles.Dqn import Dqn
from ReplayMemory import ReplayMemory
from dqnFiles.game.flappy_bird import GameState

REPLAY_MEMORY_SIZE = 30000
EPSILON0 = 0.0001
EPSILON1 = 0.08
ANNEAL_ITERATIONS = 100000
GAMMA = 0.8


def train(output):
    game_state = GameState()

    # initialize replay memory buffer
    memory = ReplayMemory(REPLAY_MEMORY_SIZE)

    run = 0

    optimizer = optim.Adam(output.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    action_list = torch.zeros([2], dtype=torch.float32)

    # get state, reward, terminal from game_state
    action_list[0] = 1
    # act on the environment and get the env state
    img_state, reward, terminal = game_state.frame_step(action_list)

    # preprocess the img state to input to the nn
    # state = preprocess(img_state)

    state = preprocess(img_state)
    state = tensor_image(state)

    # stack frames
    state = torch.cat((state, state, state, state))
    # batch size
    state = state.unsqueeze(0)

    e = EPSILON1

    e_values = np.linspace(
        start=EPSILON1,
        stop=EPSILON0,
        num=ANNEAL_ITERATIONS
    )

    while True:
        action_list = torch.zeros([2], dtype=torch.float32)

        if torch.cuda.is_available():
            action_list = action_list.cuda()

        model_output = output(state)

        pred = model_output[0]

        if run < 1000:
            action_index = torch.randint(2, torch.Size([]), dtype=torch.int)

        # epsilon greedy strategy
        else:
            random_action = random.random() <= e
            action_index = torch.randint(2, torch.Size([]), dtype=torch.int) if random_action else torch.argmax(pred)

        if torch.cuda.is_available():
            action_index = action_index.cuda()

        # setting the action_index as the action chosen
        action_list[action_index] = 1

        next_img, reward, terminal = game_state.frame_step(action_list)
        # state1 = preprocess(next_img)
        state1 = preprocess(next_img)
        state1 = tensor_image(state1)

        update_frame = state.squeeze(0)[1:, :, :]

        new_frames = torch.cat((update_frame, state1), dim=0)
        state1 = new_frames.unsqueeze(0)

        action1 = action_list.unsqueeze(0)

        reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0)

        memory.add(state, action1, reward, state1, terminal)

        if run >= 1000:
            e = e_values[run]

            mini_batch = memory.sample(32)

            states = []
            actions = []
            rewards = []
            next_states = []

            for experience in mini_batch:
                s = experience[0]
                a = experience[1]
                r = experience[2]
                n = experience[3]

                states.append(s)
                actions.append(a)
                rewards.append(r)
                next_states.append(n)

            states_list = torch.cat(states, dim=0)
            actions_list = torch.cat(actions, dim=0)
            rewards_list = torch.cat(rewards, dim=0)
            next_states_list = torch.cat(next_states, dim=0)

            if torch.cuda.is_available():
                states_list = states_list.cuda()
                actions_list = actions_list.cuda()
                rewards_list = rewards_list.cuda()
                next_states_list = next_states_list.cuda()

            pred1 = output(next_states_list)

            q_values = []

            # for i in range(len(mini_batch)):
            #     is_terminal =  rewards_list[i]
            predicted = torch.cat(tuple(
                rewards_list[i] if mini_batch[i][4] else rewards_list[i] + GAMMA * torch.max(pred1[i]) for i in
                range(len(mini_batch))))

            q_target = torch.sum(output(states_list) * actions_list, dim=1)

            optimizer.zero_grad()
            predicted = predicted.detach()
            loss = criterion(q_target, predicted)
            loss.backward()
            optimizer.step()

        state = state1
        run += 1

        if run % 10000 == 0:
            torch.save(output, "trained_models/model_" + str(run) + ".pth")

        print("episode: ", run, "Epsilon value ", e)
        print("Q value: ", np.max(pred.cpu().detach().numpy()))


def test(model):
    game_state = GameState()

    action_list = torch.zeros([2], dtype=torch.float32)
    action_list[0] = 1
    image_data, reward, terminal = game_state.frame_step(action_list)
    image_data = preprocess(image_data)
    image_data = tensor_image(image_data)
    state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)

    while True:
        res = model(state)[0]

        action_list = torch.zeros([2], dtype=torch.float32)
        if torch.cuda.is_available():
            action_list = action_list.cuda()

        action_index = torch.argmax(res)
        if torch.cuda.is_available():
            action_index = action_index.cuda()
        action_list[action_index] = 1

        next_image, reward, terminal = game_state.frame_step(action_list)
        next_state = preprocess(next_image)
        next_state = tensor_image(next_state)
        state1 = torch.cat((state.squeeze(0)[1:, :, :], next_state)).unsqueeze(0)

        state = state1


def preprocess(image):
    image = image[0:288, 0:404]
    image_data = cv2.cvtColor(cv2.resize(image, (84, 84)), cv2.COLOR_BGR2GRAY)
    image_data[image_data > 0] = 255
    image_data = np.reshape(image_data, (84, 84, 1))
    return image_data


def tensor_image(image):
    image_tensor = image.transpose(2, 0, 1)
    image_tensor = image_tensor.astype(np.float32)
    image_tensor = torch.from_numpy(image_tensor)
    if torch.cuda.is_available():  # put on GPU if CUDA is available
        image_tensor = image_tensor.cuda()
    return image_tensor


def main(mode):
    cuda_is_available = torch.cuda.is_available()

    if mode == 'test':
        model = torch.load(
            'trained_models/model_3500.pth',
            map_location='cpu' if not cuda_is_available else None
        ).eval()

        if cuda_is_available:
            model = model.cuda()

        test(model)

    elif mode == 'train':
        if not os.path.exists('trained_models/'):
            os.mkdir('trained_models/')

        model = Dqn()

        if cuda_is_available:
            model = model.cuda()

        model.set_weights()

        train(model)


if __name__ == "__main__":
    main(sys.argv[1])
