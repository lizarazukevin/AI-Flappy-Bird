"""
File: agent.py
Description: Contains all classes of intelligent agents
Modified By: Kevin Lizarazu
Date: 04/21/2024
"""
import torch
import random
import numpy as np
import math

from collections import deque

from game.util import load_json
from game.model import LinearDQN, QTrainer
from game.game import FlappyGameDQN

# load JSON variables
conf = load_json("config.json")

EPISODES = conf["RL"]["EPISODES"]
LEARNING_RATE = conf["RL"]["LEARNING_RATE"]
GAMMA = conf["RL"]["GAMMA"]
EPSILON = conf["RL"]["EPSILON"]
MAX_MEMORY = conf["RL"]["MAX_MEMORY"]
BATCH_SIZE = conf["RL"]["BATCH_SIZE"]

WIDTH = conf["GAME"]["WIDTH"]
HEIGHT = conf["GAME"]["HEIGHT"]
FPS = conf["GAME"]["FPS"]
SCROLL_SPEED = conf["GAME"]["SCROLL_SPEED"]
ANIMATION_DELAY = conf["GAME"]["ANIMATION_DELAY"]
GRAVITY = conf["GAME"]["GRAVITY"]

ENV_STYLE = conf["GAME"]["ENV_STYLE"]
BIRD_STYLE = conf["GAME"]["BIRD_STYLE"]
BIRD_W = conf["GAME"]["BIRD_W"]
BIRD_H = conf["GAME"]["BIRD_H"]
FLOOR_STYLE = conf["GAME"]["FLOOR_STYLE"]
FLOOR_W = conf["GAME"]["FLOOR_W"]
FLOOR_H = conf["GAME"]["FLOOR_H"]
PIPE_COLOR = conf["GAME"]["PIPE_COLOR"]
PIPE_W = conf["GAME"]["PIPE_W"]
PIPE_H = conf["GAME"]["PIPE_H"]
PIPE_GAP = conf["GAME"]["PIPE_GAP"]
RAND_UPPER = conf["GAME"]["RAND_UPPER"]
RAND_LOWER = conf["GAME"]["RAND_LOWER"]

class DQNAgent():
    def __init__(self):
        self.game_counter = 0
        self.epsilon = EPSILON
        self.gamma = GAMMA
        self.lr = LEARNING_RATE
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = LinearDQN(6, 256, 2)
        self.trainer = QTrainer(self.model, self.lr, self.gamma)
    
    # Retrieves current state of observations
    # [pipe_bot (bool), pipe_top (bool), pipe_mid (int), dist_to_pipe (int), dist_to_ceiling (int), dist_to_floor (int)]
    def get_state(self, game):
        state = [-1, -1, -1, -1, -1, -1]

        # pipe stuff
        if len(game.pipes) == 2:
            state[0] = True if game.player.rect.bottom < game.pipes[0].rect.top else False
            state[1] = True if game.player.rect.top > game.pipes[1].rect.bottom else False
            state[2] = abs((game.player.rect.top + (BIRD_H // 2)) - (game.pipes[0].rect.top - (PIPE_GAP // 2)))
            state[3] = game.pipes[0].rect.right - game.player.rect.left
        elif len(game.pipes) == 4:
            state[0] = True if game.player.rect.bottom < game.pipes[2].rect.top else False
            state[1] = True if game.player.rect.top > game.pipes[3].rect.bottom else False
            state[2] = abs((game.player.rect.top + (BIRD_H // 2)) - (game.pipes[2].rect.top - (PIPE_GAP // 2)))
            state[3] = game.pipes[2].rect.right - game.player.rect.left

        # floor/ceiling
        state[4] = abs(game.player.rect.bottom)
        state[5] = abs(game.h - game.floor_h - game.player.rect.top)

        return np.array(state, dtype=int)
    


    # Past state information saved for future reference
    def remember(self, state, action, rew, next_state, done):
        self.memory.append((state, action, rew, next_state, done))

    # When game ends, sample of all memory is used to update model
    def train_long_mem(self):
        # Gather a sample in batch sizes, full memory if not enough
        if len(self.memory) > BATCH_SIZE:
            mem_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mem_sample = self.memory
        
        # unpacks and packages into lists for model update
        state, action, rew, next_state, done = zip(*mem_sample)
        self.trainer.train_step(np.array(state), np.array(action), np.array(rew), np.array(next_state), np.array(done))

    # Updates the model trained every step
    def train_short_mem(self, state, action, rew, next_state, done):
        self.trainer.train_step(state, action, rew, next_state, done)

    # Returns best action given current state and model
    def get_action(self, state):
        action = [0, 0]

        # exploration vs exploitation
        self.epsilon = 100 - self.game_counter
        if random.randint(0, 300) < self.epsilon:
            idx = random.randint(0, 1)
        else:
            pred_actions = self.model(torch.tensor(state, dtype=torch.float))
            idx = torch.argmax(pred_actions).item()

        action[idx] = 1
        return action

# Training loop runs as many games as it can, cieling is obtaining a score of 999
def train():
    print("Welcome to AI Flappy Bird + DQN Training")

    plot_scores = []
    plot_mean = []
    total_score = 0
    high_score = 0
    agent = DQNAgent()
    game = FlappyGameDQN(
        (WIDTH, HEIGHT),
        (FLOOR_W, FLOOR_H),
        (BIRD_W, BIRD_H),
        (PIPE_W, PIPE_H, PIPE_GAP, PIPE_COLOR),
        (RAND_LOWER, RAND_UPPER),
        ENV_STYLE, 
        BIRD_STYLE, 
        ANIMATION_DELAY,
        SCROLL_SPEED,
        GRAVITY, 
        FPS
    )
    
    # training loop
    while True:
        # get current state and take single game step with action
        curr_state = agent.get_state(game)
        action = agent.get_action(curr_state)
        rew, done, score =  game.play_step(action)

        # get new state and update model
        next_state = agent.get_state(game)
        agent.remember(curr_state, action, rew, next_state, done)
        agent.train_short_mem(curr_state, action, rew, next_state, done)

        # flappy bird collided, sample remembered states and update model
        if done:
            game.reset()
            agent.game_counter += 1
            agent.train_long_mem()

            # save model if new high score is achieved
            if high_score < score:
                high_score = score
                agent.model.save()

            print(f"Game: {agent.game_counter}, Score: {score}, Highest Score: {high_score}")
        
        # plotting business (game score and average game scores)


if __name__ == "__main__":
    train()
