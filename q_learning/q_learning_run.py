"""
File: q_learning_run.py
Description: Contains methods to run agent using Q-Learning
Modified By: Devashree Bhagwat
Date: 05/01/2024
"""

import json

class QLearning:
    def __init__(self):
        print("Running Flappy Bird with Q-Learning...")

        # Set training mode to False indicating the system is not currently learning
        self.training_mode = False
        # Set the discount factor for future rewards in the Q-learning algorithm
        self.discount_factor = 0.95
        # Set the learning rate for the Q-learning update rule
        self.learning_rate = 0.6
        # Define penalties for certain actions to influence the learning process
        self.penalties = {0: 0, 1: -1000}
        # Initialize an empty dictionary to store Q-values
        self.q_values = {}
        # Load Q-values from a file or start with an empty table if the file doesn't exist
        self.load_q_values()

    def load_q_values(self):
        # Try to load Q-values from a JSON file; if not found, use an empty Q-table
        try:
            with open("./q_learning/training_data/q_values.json", "r") as file:
                self.q_values = json.load(file)
        except FileNotFoundError:
            print("Q-values file not found, operating with an empty Q-table.")

    def initialize_q_values(self, state):
        # Ensure a Q-value entry exists for the given state, with default values [0, 0, 0]
        self.q_values.setdefault(state, [0, 0, 0])

    def choose_action(self, x, y, vel, pipe):
        # Determine the current state based on the position and velocity data
        current_state = self.compute_state(x, y, vel, pipe)
        # Choose an action based on the current Q-values, preferring the action with a higher Q-value
        return 0 if self.q_values[current_state][0] >= self.q_values[current_state][1] else 1

    def compute_state(self, x, y, vel, pipe):
        # Calculate state based on the relative position of the nearest pipes and the bird's velocity
        pipe0, pipe1 = pipe[0], pipe[1]
        if x - pipe[0]["x"] >= 50:
            pipe0 = pipe[1]
            if len(pipe) > 2:
                pipe1 = pipe[2]
        x0 = pipe0["x"] - x
        y0 = pipe0["y"] - y
        if -50 < x0 <= 0:
            y1 = pipe1["y"] - y
        else:
            y1 = 0
        # Quantize the x and y offsets to reduce the state space
        if x0 < -40:
            x0 = int(x0)
        elif x0 < 140:
            x0 = int(x0) - (int(x0) % 10)
        else:
            x0 = int(x0) - (int(x0) % 70)
        if -180 < y0 < 180:
            y0 = int(y0) - (int(y0) % 10)
        else:
            y0 = int(y0) - (int(y0) % 60)
        if -180 < y1 < 180:
            y1 = int(y1) - (int(y1) % 10)
        else:
            y1 = int(y1) - (int(y1) % 60)
        # Format state as a string and ensure Q-values are initialized for this state
        state = str(int(x0)) + "_" + str(int(y0)) + "_" + str(int(vel)) + "_" + str(int(y1))
        self.initialize_q_values(state)
        return state
