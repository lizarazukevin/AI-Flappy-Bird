"""
File: cnn_dqn_train.py
Description: CNN variant of DQN learning trains agent on array of game states
Modified By: Kevin Lizarazu
Date: 04/21/2024
"""
import random
import gymnasium
import numpy as np
import torch
import os
import wandb
import flappy_bird_gymnasium

GAMMA = 0.99
LEARNING_RATE = 10e-4
MEMORY_SIZE = 100000
BATCH_SIZE = 64
EPSILON_START = 0.5
EPSILON_END = 1e-4
EPSILON_DECAY = 0.9995
EPISODES = 100000

# speeds up training with the use of gpu when using torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create the gymansium environment in preprocessed rgb array
env = gymnasium.make("FlappyBird-v0", render_mode="rgb_array")

class DQNModel(torch.nn.Module):
    """
    Architecture for CNN model
    """
    def __init__(self, num_inputs, num_actions):
        super(DQNModel, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(num_inputs, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_actions)
        )

    def forward(self, state):
        return self.model(state)
    

class ExperienceMemory:
    """
    From stored states, fetches a batch to use for experience replay learning
    """
    def __init__(self, max_size):
        self.max_size = max_size
        self.memory = []
        self.index = 0

    def add(self, experience):
        if len(self.memory) < self.max_size:
            self.memory.append(None)
        self.memory[self.index] = experience
        self.index = (self.index + 1) % self.max_size

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

def train():
    """
    Trains a model using CNN architecture as part of our DQN methods, 
    """
    print("Training Flappy Bird with DQN...")

    # initiate wandb
    wandb.init(project="CNNFlappyBird", config={
        "learning_rate": LEARNING_RATE, 
        "epochs": EPISODES, 
        "batch_size": BATCH_SIZE
    })

    # create, load empty weights, and set to model to evaluate
    # two models to further stabilize training
    model = DQNModel(env.observation_space.shape[0], env.action_space.n)
    target_model = DQNModel(env.observation_space.shape[0], env.action_space.n)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    wandb.watch(model, log='all', log_freq=10)
    wandb.watch(target_model, log='all', log_freq=10)

    # optimizer and memory used for experience replay/optimization
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    memory = ExperienceMemory(MEMORY_SIZE)

    # starting epsilon, before decay
    epsilon = EPSILON_START

    for episode in range(EPISODES):
        state, _ = env.reset()
        done = False
        total_rew = 0
        steps = 0
        loss = None

        while not done:

            # exploration vs exploitation
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                state_tensor = torch.tensor(state).float().unsqueeze(0)
                action = model(state_tensor).argmax(dim=1).item()
                
            next_state, rew, done, _, _ = env.step(action)

            if rew == 1:
                rew = 10
            elif rew == -1:
                rew = -5

            memory.add((state, action, rew, next_state, done))

            # train enough iterations to sample from a full batch size
            if len(memory.memory) >= BATCH_SIZE:
                batch = memory.sample(BATCH_SIZE)
                states, actions, rewards, next_states, dones = zip(*batch)

                states_tensor = torch.tensor(np.array(states)).float()
                actions_tensor = torch.tensor(np.array(actions)).long()
                rewards_tensor = torch.tensor(np.array(rewards)).float()
                next_states_tensor = torch.tensor(np.array(next_states)).float()
                dones_tensor = torch.tensor(np.array(dones)).float()

                current_q_values = model(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
                next_q_values = target_model(next_states_tensor).max(dim=1).values
                target_q_values = rewards_tensor + (1 - dones_tensor) * next_q_values * GAMMA

                loss = torch.nn.functional.mse_loss(current_q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # decays epsilon until minimum exploration
                if episode > 300:
                    if epsilon > EPSILON_END:
                        epsilon *= EPSILON_DECAY

                if episode % 10 == 0:
                    target_model.load_state_dict(model.state_dict())

            state = next_state
            total_rew += rew
            steps += 1

        # log episode info to wandb
        wandb.log({
            'total_rewards': total_rew,
            'average_reward': total_rew/steps,
            'episode': episode,
            'loss': loss,
            'epsilon': epsilon,
            'learning_rate': LEARNING_RATE
        })

        # log episode info
        if episode % 100 == 0:
            print(f"EPISODE: {episode}, TOTAL REWARD: {total_rew}, MEAN REWARD: {total_rew/steps}")

        # every 1000th episode, saves the trained weights
        if episode % 1000 == 0:
            if not os.path.exists('./cnn_dqn/models'):
                os.makedirs('./cnn_dqn/models')
            torch.save(model.state_dict(), f"./cnn_dqn/models/model2_{episode}.pth")