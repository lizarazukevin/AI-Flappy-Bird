"""
File: q_learning_train.py
Description: Contains methods to train agent using Q-Learning
Modified By: Devashree Bhagwat
Date: 05/01/2024
"""

import json
from datetime import datetime

class QLearning:
    # Initialize QLearning object with default values and load training data.
    def __init__(self):
        print("Training Flappy Bird with Q-Learning")

        self.training_mode = True  # Indicates whether the model is training
        self.discount_factor = 0.95  # Discount factor for future rewards
        self.learning_rate = 0.6  # Initial learning rate for Q-value updates
        self.penalties = {0: 0, 1: -1000}  # Penalties applied for each action
        self.learning_rate_decay = 0.00003  # Rate at which learning rate decays over time
        self.episode_count = 0  # Number of episodes the model has been trained on
        self.last_action = 0  # Last action taken
        self.last_state = "0_0_0_0"  # Initial state
        self.history = []  # List to store state transitions
        self.scores = []  # List to store scores per episode
        self.highest_score = 0  # Highest score achieved
        self.q_values = {}  # Dictionary to store Q-values
        self.load_q_values()  # Load Q-values from file
        self.load_training_data()  # Load training data from file

    # Load Q-values from a JSON file, initialize if file not found.
    def load_q_values(self):
        try:
            with open("./q_learning/training_data/q_values.json", "r") as file:
                self.q_values = json.load(file)
        except FileNotFoundError:
            self.initialize_q_values(self.last_state)

    # Initialize Q-values for a new or unseen state.
    def initialize_q_values(self, state):
        self.q_values.setdefault(state, [0, 0, 0])

    # Load training data from a JSON file.
    def load_training_data(self):
        print("Loading training data from JSON file...")
        try:
            with open("./q_learning/training_data/training_values.json", "r") as file:
                training_data = json.load(file)
                self.episode_count = training_data['episodes'][-1]
                self.scores = training_data['scores']
                self.learning_rate = max(self.learning_rate - self.learning_rate_decay * self.episode_count, 0.1)
                self.highest_score = max(self.scores)
        except FileNotFoundError:
            pass

    # Choose an action based on current state and Q-values.
    def choose_action(self, x, y, vel, pipe):
        current_state = self.compute_state(x, y, vel, pipe)
        self.history.append((self.last_state, self.last_action, current_state))
        self.trim_history()
        self.last_state = current_state
        self.last_action = 0 if self.q_values[current_state][0] >= self.q_values[current_state][1] else 1
        return self.last_action

    # Update Q-values using the latest score and the history of states/actions.
    def update_q_values(self, score):
        self.episode_count += 1
        self.scores.append(score)
        self.highest_score = max(score, self.highest_score)
        reversed_history = list(reversed(self.history))
        high_death = int(reversed_history[0][2].split("_")[1]) > 120
        time_step, was_flap = 0, True
        for state, action, next_state in reversed_history:
            time_step += 1
            reward = self.penalties[0] if time_step > 2 else self.penalties[1]
            if was_flap or high_death and action:
                reward = self.penalties[1]
                was_flap = False
                high_death = False
            self.q_values[state][action] = ((1 - self.learning_rate) * self.q_values[state][action] +
                                            self.learning_rate * (reward + self.discount_factor *
                                                                  max(self.q_values[next_state][0:2])))
        if self.learning_rate > 0.1:
            self.learning_rate = max(self.learning_rate - self.learning_rate_decay, 0.1)
        self.history = []

    # Compute the current state based on game variables.
    def compute_state(self, x, y, vel, pipe):
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
        state = str(int(x0)) + "_" + str(int(y0)) + "_" + str(int(vel)) + "_" + str(int(y1))
        self.initialize_q_values(state)
        return state

    # Trim history to prevent excessive memory use.
    def trim_history(self, max_length=1000000):
        if len(self.history) > max_length:
            trimmed_history = self.history[-max_length:]
            for state, action, next_state in reversed(trimmed_history):
                self.q_values[state][action] = ((1 - self.learning_rate) * self.q_values[state][action] +
                                                self.learning_rate * (self.penalties[0] + self.discount_factor *
                                                                      max(self.q_values[next_state][0:2])))
            self.history = self.history[-max_length:]

    # Handle the end of an episode, updating scores and history.
    def end_episode(self, score):
        self.episode_count += 1
        self.scores.append(score)
        self.max_score = max(score, self.highest_score)
        history = list(reversed(self.history))
        for move in history:
            state, action, new_state = move
            self.q_values[state][action] = (1 - self.learning_rate) * (self.q_values[state][action]) + \
                                           self.learning_rate * (self.penalties[0] + self.discount_factor *
                                                                 max(self.q_values[new_state][0:2]))
        self.moves = []

    # Save current Q-values to a JSON file.
    def save_q_values(self):
        print(f"Saving Q-table with {len(self.q_values)} states to file...")
        with open(f"./q_learning/training_data/q_values{datetime.now()}.json", "w") as file:
            json.dump(self.q_values, file)

    # Save current training data to a JSON file.
    def save_training_data(self):
        print(f"Saving training data with {self.episode_count} episodes to file...")
        with open(f"./q_learning/training_data/training_values{datetime.now()}.json", "w") as file:
            json.dump({'episodes': list(range(1, self.episode_count + 1)),
                       'scores': self.scores}, file)

