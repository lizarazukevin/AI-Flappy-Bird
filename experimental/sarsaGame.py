"""
File: sarsaGame.py
Description: Contains executable code for training a SARSA instance of Flappy Bird
Modified By: Kevin Lizarazu
Date: 04/30/2024
"""
import numpy as np
import copy
import json

from collections import defaultdict

from game.util import load_json
from game.game import FlappyGameSARSA

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

MIN_EXPLORING_RATE = 0.01
MIN_LEARNING_RATE = 0.5

class SARSAAgent():
    def __init__(self, num_actions, bucket_range_per_feature, disc_factor=0.99):
        self.explore_count = 200

        self.disc_factor = disc_factor
        self.num_actions = num_actions
        self.update_params(0)

        # creates a qtable sized to number of actions
        self.q_table = defaultdict(lambda: np.zeros(num_actions))
        self.bucket_range_per_feature = bucket_range_per_feature

    def get_state(self, game):
        state = {
            "player_y": game.player.rect.y,
            "player_vel": game.player.y_vel,
            "next_pipe_dist_to_player": WIDTH//2,
            "next_pipe_top_y": (HEIGHT - FLOOR_H)//2 - PIPE_GAP,
            "next_pipe_bot_y": (HEIGHT - FLOOR_H)//2 + PIPE_GAP,
            "next_next_pipe_dist_to_player": WIDTH//2,
            "next_next_pipe_top_y": (HEIGHT - FLOOR_H)//2 - PIPE_GAP,
            "next_next_pipe_bot_y": (HEIGHT - FLOOR_H)//2 + PIPE_GAP
        }

        # pipe stuff
        if len(game.pipes) == 2:
            state['next_pipe_dist_to_player'] = game.pipes[0].rect.x - game.player.rect.x
            state['next_pipe_bot_y'] = game.pipes[0].rect.y
            state['next_pipe_top_y'] = game.pipes[1].rect.bottom
        if len(game.pipes) == 4:
            state['next_next_pipe_dist_to_player'] = game.pipes[2].rect.x - game.player.rect.x
            state['next_next_pipe_bot_y'] = game.pipes[2].rect.y
            state['next_next_pipe_top_y'] = game.pipes[3].rect.bottom

        return state

    def get_action(self, state, episode):
        action = [0, 0]

        # explore first
        # if episode < self.explore_count:
        #     action[np.random.choice(self.num_actions)] = 1
        #     return action

        # decrement epsilon slightly every iteration
        rand = np.random.random()
        self.epsilon *= 0.9

        if rand < self.epsilon:
            idx = np.random.choice(self.num_actions)
        else:
            state_idx = self.get_state_idx(state)
            idx = np.argmax(self.q_table[state_idx])

        action[idx] = 1
        return action

    def update_params(self, episode):
        self.lr = max(MIN_LEARNING_RATE, min(0.5, 0.99 ** (episode / 30)))
        self.epsilon = max(MIN_EXPLORING_RATE, min(0.5, 0.99 ** (episode / 30)))
    
    def exploit(self):
        self.epsilon = 0

    def update_policy(self, state, action, reward, next_state):
        state_idx = self.get_state_idx(state)
        next_state_id = self.get_state_idx(next_state)
        bestq = np.max(self.q_table[next_state_id])
        self.q_table[state_idx][action] += self.lr * (reward + self.disc_factor * bestq - self.q_table[state_idx][action])

    def get_state_idx(self, state):
        state = copy.deepcopy(state)
        state['next_next_pipe_bot_y'] -= state['player_y']
        state['next_next_pipe_top_y'] -= state['player_y']
        state['next_pipe_bot_y'] -= state['player_y']
        state['next_pipe_top_y'] -= state['player_y']

        state_key = [k for k, v in sorted(state.items())]

        state_idx = []
        for key in state_key:
            int(state[key] / self.bucket_range_per_feature[key])
        return tuple(state_idx)

class SAgent:
    """
    Agent clas for SARSA implementation
    """
    def __init__(self, actions, descr=None):
        """
        actions -> dim of possible action values
        discr -> rounds state space for discretization
        """
        self.actions = actions
        self.qTable = defaultdict(float)
        self.env = FlappyGameSARSA(
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
            FPS,
            descr
        )

    def get_action(self, state):
        """
        state -> current state
        """
        if np.random.rand() < self.epsilon:
            return np.random.choice([0, 1])
        
        qPrime = [self.qTable.get((state, action), 0) for action in self.actions]

        if qPrime[0] < qPrime[1]:
            return 1

        return 0

    def saveQTable(self):
        toSave = {key[0] + ' action ' + str(key[1]) : self.qTable[key] for key in self.qTable}
        with open('qTable.json', 'w') as f:
            json.dump(toSave, f)

    def loadQTable(self):
        def parseKey(key):
            state = key[:-9]
            action = int(key[-1])
            return (state, action)

        with open('qTable.json') as f:
            toLoad = json.load(f)
            self.qTable = {parseKey(key) : toLoad[key] for key in toLoad}

    def train(self, episodes=2000, epsilon=0.1, gamma=1, saveInterval=200):
        """
        episodes -> how many episodes to train
        epsilon -> exploitation vs exploration
        gamma -> discount factor
        save_interval -> save info every # episodes
        """
        
        for ep in range(episodes):
            if ep % 50 == 0 or ep == episodes - 1:
                print("Iter: ", ep)
            
            # could implement epsilong decay here
            score = 0
            totalRew = 0
            self.env.reset()
            gameEp = []
            state = self.env.get_state()
            action = self.get_action(state)

            # run the episode here

    def test(self, episodes):
        pass

    def saveOutput(self):
        pass


def train():
    print("Welcome to AI Flappy Bird + SARSA Training")

    reward_per_epoch = []
    lifetime_per_epoch = []
    exploring_rates = []
    learning_rates = []

    print_every_episode = 500
    NUM_EPISODE = 40000

    bucket_range_per_feature = {
        'next_next_pipe_bot_y': 40,
        'next_next_pipe_dist_to_player': 512,
        'next_next_pipe_top_y': 40,
        'next_pipe_bot_y': 20,
        'next_pipe_dist_to_player': 20,
        'next_pipe_top_y': 20,
        'player_vel': 4,
        'player_y': 16
    }

    agent = SARSAAgent(2, bucket_range_per_feature)
    game = FlappyGameSARSA(
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

    for episode in range(NUM_EPISODE):
        game.reset()

        if episode % print_every_episode == 0:
            agent.exploit()

        state = agent.get_state(game)
        cum_reward = 0
        t = 0

        # train loop
        while True:
            # observe the current game state
            action = agent.get_action(state, episode)
            rew, done, score = game.play_step(action)

            cum_reward += rew

            # get next state 
            next_state = agent.get_state(game)

            # update Q-values using SARSA update rule
            agent.update_policy(state, action, rew, next_state)
            state = next_state
            t += 1

            if done:
                break
        
        agent.update_params(episode)

        if episode % print_every_episode == 0:
            print("Episode {} finished after {} time steps, cumulated reward: {}, exploring rate: {}, learning rate: {}".format(
                episode,
                t,
                cum_reward,
                agent.epsilon,
                agent.lr
            ))
            reward_per_epoch.append(cum_reward)
            exploring_rates.append(agent.epsilon)
            learning_rates.append(agent.lr)
            lifetime_per_epoch.append(t)


if __name__ == "__main___":
    train()
