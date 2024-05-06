"""
File: sarsaRun.py
Description: Uses trained model to play a game
Modified By: Inseong Lee
Date: 05/01/2024
"""

import numpy as np
import gymnasium
import matplotlib.pyplot as plt
import flappy_bird_gymnasium
import time
import json
from filelock import Timeout, FileLock

class sarsaRunner:

    def __init__(self):
        
        self.lock_path = "./sarsa/training_data/best_q_table.lock"

        self.num_buckets_distance = 80  # Number of buckets for horizontal distance to the next pipe
        self.num_buckets_velocity = 20  # Number of buckets for vertical velocity
        self.num_buckets_position = 60  # Number of buckets for vertical position
        self.num_buckets_ver_dis = 30
        self.action_space = [0, 1]  # Possible actions


    def get_state(self, observation):
        if isinstance(observation, tuple):
            state_array = observation[0]
        else:
            state_array = observation
            
        max_values = [1.0, 1.0, 0.75, 0.43]  # Maximum observed values for each feature
        min_values = [-0.181, -0.9, -0.18, 0]  # Minimum observed values for each feature

        # Calculate indices for discretization
        horizontal_distance_to_next_pipe = state_array[0]
        vertical_velocity = state_array[10]
        vertical_position = state_array[9]
        vertical_distance_to_center_of_pipe = (state_array[1] + state_array[2]) / 2

        dist_index = int(((horizontal_distance_to_next_pipe - min_values[0]) / (max_values[0] - min_values[0])) * self.num_buckets_distance)
        velocity_index = int(((vertical_velocity - min_values[1]) / (max_values[1] - min_values[1])) * self.num_buckets_velocity)
        vertical_pos = int(((vertical_position - min_values[2]) / (max_values[2] - min_values[2])) * self.num_buckets_position)
        vertical_dis = int(((vertical_distance_to_center_of_pipe - min_values[3]) / (max_values[3] - min_values[3])) * self.num_buckets_ver_dis)

        # Ensure indices are within bounds
        dist_index = min(max(dist_index, 0), self.num_buckets_distance - 1)
        velocity_index = min(max(velocity_index, 0), self.num_buckets_velocity - 1)
        vertical_pos = min(max(vertical_pos, 0), self.num_buckets_position - 1)
        vertical_dis = min(max(vertical_dis, 0), self.num_buckets_ver_dis - 1)

        # return (dist_index, velocity_index, vertical_pos, vertical_dis)
        state = (dist_index, velocity_index, vertical_pos, vertical_dis)
        return state

    
    def max_action(self, state, actions=[0,1]):
        values = np.array([self.Q[state, a] for a in actions])
        action = np.argmax(values)
        
        return action

    def run(self):
        print("Running Flappy Bird with SARSA...")
        filename = "best_q_table" #change filename
        self.Q = np.load(f"./sarsa/training_data/best_q_table.npy", allow_pickle=True).item()

        # Setup environment
        env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False)
        obs = env.reset()    
        done = False

        # # Replay using the best Q-table
        while not done:
            state = self.get_state(obs)
            action = self.max_action(state)
                    
            obs, reward, done,  _, info = env.step(action)
            env.render()  # This will display the current environment state
            time.sleep(1 / 60) 

        env.close()
        