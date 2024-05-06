"""
File: cnn_dqn_run.py
Description: Uses trained model to play a game
Modified By: Kevin Lizarazu
Date: 04/21/2024
"""
import gymnasium
import torch

from dqn.dqn_train import DQNModel

EPISODES = 10
PATH_TO_WEIGHTS = './dqn/models/model2_4000.pth'

# speeds up training with the use of gpu when using torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create the gymansium environment in preprocessed rgb array
env = gymnasium.make("FlappyBird-v0", render_mode="rgb_array")

def run():
    """
    Runs a trained model for a number of episodes
    """
    print("Running Flappy Bird with DQN...")

    # create, load trained weights, and set mode to evaluate model
    model = DQNModel(env.observation_space.shape[0], env.action_space.n)
    model.load_state_dict(torch.load(PATH_TO_WEIGHTS))
    model.eval()

    for ep in range(EPISODES):
        state, _ = env.reset()
        done = False
        total_rew = 0
        steps = 0

        while not done:
            env.render()
            state_tensor = torch.tensor(state).float().unsqueeze(0)
            action = model(state_tensor).argmax(dim=1).item()

            next_state, reward, done, _, _ = env.step(action)
            state = next_state
            total_rew += reward
            steps += 1

        print(f"EPISODE {ep}, TOTAL REWARD: {total_rew}, MEAN REWARD: {total_rew/steps}")