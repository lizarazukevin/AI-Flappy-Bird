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
from game.model import LinearSARSA, SARSATrainer
from game.game import FlappyGameSARSA

import matplotlib.pyplot as plt

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

def plot_progress(scores, averages):
    plt.figure(figsize=(10,5))
    plt.plot(scores, label="Scores per Episode")
    plt.plot(averages, label="Average Score")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.title("Training Progress")
    plt.legend()
    plt.grid(True)
    plt.show() 

        
def get_state(obs,value_checker):
    
    max_values = [195.0 ,258.0, 25.5, 581.0]
    min_values = [-21.0 ,-477.0, -6, -37]
    
    
    num_buckets = [20, 50, 10, 50]

    # Initialize state list
    state = []


    # Calculate the bucket index for each state dimension
    for i, (max_value, min_value, buckets) in enumerate(zip(max_values, min_values, num_buckets)):
        # Scale and normalize the observation
        scaled = (obs[i] - min_value) / (max_value - min_value) if max_value != min_value else 0
        
        # Calculate the bucket index for this dimension
        bucket_index = int(scaled * buckets)
        
        # Ensure the bucket index lies within the valid range
        bucket_index = min(bucket_index, buckets - 1)
        bucket_index = max(bucket_index, 0)

        # Append the bucket index to the state list
        state.append(bucket_index)

    # if obs[0] != -999 and obs[1] != -999:
    #     if value_checker[0]< obs[0]:
    #         value_checker[0] = obs[0]
    #     if value_checker[1]< obs[1]:
    #         value_checker[1] = obs[1]
    #     if value_checker[2]< obs[2]:
    #         value_checker[2] = obs[2]
    #     if value_checker[2]< obs[3]:
    #         value_checker[3] = obs[3]
        
        
    if obs[1] == -999 and obs[0] == -999:
        state[0] = 0
        state[1] = 0
    
        
    return tuple(state)
    
    state = (0, 0, 0, 0)
    
    return state

def maxAction(Q, state, actions=[0,1]):
    values = np.array([Q[state,a] for a in actions])
    action = np.argmax(values)

    return action

def choose_action(Q, state, epsilon):
    if state[0] == 0 and state[1] == 0:
        return choose_action_no_pipe_in_sight(state, epsilon)
    elif np.random.rand() < epsilon:
        return np.random.choice([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]) ## reduce chance of flapping too much
    else:
        return maxAction(Q, state) # Exploit
    
def choose_action_no_pipe_in_sight(state, epsilon):
    # print(epsilon)
    # if np.random.rand() < epsilon:
    #     return np.random.choice([0, 1])
    # else:
    if state[3] > 12:
        return 1
    else:
        return 0
    
def sum_q_values_for_specific_states(Q, fourth_index=10):
    sum_q_values = 0
    # Iterate over all keys in the Q-table
    for (state, action), value in Q.items():
        # Check if the third and fourth indices of the state match the specified values
        if state[3] == fourth_index:
            sum_q_values += value

    return sum_q_values

def main():
    
    value_checker = [0, 0, 0, 0]
    
    scores = []
    highest_scores = [] 
    average_scores = []  # List to store average scores progression

    max_value_container = [0]
        
    alpha = 0.1    # learning rate   low = slow learning, high = fast learning
    gamma = 0.9   # discount factor  low = immediate reward, high = future reward
    epsilon = 0.2  # exploration rate
    epsilon_decay = 0.9995  # Slower decay rate
    epsilon_min = 0.01  # Minimum value of epsilon

    # Environment

    env = FlappyGameSARSA(        
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
            FPS)


    num_buckets = [20, 50, 10, 50]
    action_space = [0, 1]  # Possible actions

    Q = {}
    for i in range(num_buckets[0]):
        for j in range(num_buckets[1]):
            for k in range(num_buckets[2]):
                for l in range(num_buckets[3]):
                    for action in action_space:
                        Q[(i, j, k, l), action] = 0.0
        
    num_episodes =2000
    highest_score = 0
    total_score = 0  # Total score accumulator
    for episode in range(num_episodes):
        print("episode", episode)
        obs = env.reset()
        if obs is None:
            print("Game object is None")
        
        # print("Observation:", obs)
        # print("Length of observation:", len(obs))
        state = get_state(obs,value_checker)
        
        action = choose_action(Q, state, epsilon)
        total_reward = 0

        # print(f"Initial Observation: {obs} -> Initial State: {state}")
        while True:
            next_obs, reward, done, score = env.play_step(action) # rew, done, self.score
            # print("obs",next_obs[0],next_obs[1],next_obs[2],next_obs[3])
            next_state = get_state(next_obs,value_checker)
            # if next_state[3] == 10:
            #     print("next_state[2] == 10 and next_state[3] == 10")
            next_action = choose_action(Q, next_state, epsilon)
            
            # SARSA update formula
            # SARSA Update
            Q[state, action] += alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action])
            state, action = next_state, next_action
            total_reward += reward
            if done:
                print(score)
                print(total_reward)
                scores.append(total_reward)
                total_score += total_reward
                average_scores.append(total_score / (episode + 1))  # Calculate the average score
                if total_reward > highest_score:
                    highest_score = total_reward
                    print(f"New highest score: {highest_score}")
                    np.save("best_q_table.npy", Q)
                    with open("highest_score.txt", "w") as f:
                        f.write(str(highest_score))
                highest_scores.append(highest_score)
                break
        specific_state_sum = sum_q_values_for_specific_states(Q)
        print(specific_state_sum)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)  # Decay epsilon    
                
        # if (episode + 1) % 100 == 0:
            # print(f"Q-Table on episode {episode + 1}:")
            # print(q_table)
    # env.close()
    print("final values", value_checker)
    plot_progress(scores, average_scores)
    quit()
    