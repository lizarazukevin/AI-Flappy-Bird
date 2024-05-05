import flappy_bird_gymnasium
import gymnasium
import numpy as np
import matplotlib.pyplot as plt
import json
import sys
import time
from filelock import Timeout, FileLock
from datetime import datetime

class sarsaTrainer:    
    def __init__(self):
        
        self.lock_path = "best_q_table.lock"  
        self.scores = []
        self.highest_scores = [] 
        self.average_scores = []  # List to store average scores progression

            
        self.alpha = 0.3   # learning rate   low = slow learning, high = fast learning
        self.gamma = 0.96   # discount factor  low = immediate reward, high = future reward
        self.epsilon = 0.3  # exploration rate
        self.epsilon_decay = 0.995 # Slower decay rate
        self.epsilon_min = 0.01  # Minimum value of epsilon

        # Environment

        self.env = gymnasium.make("FlappyBird-v0",   use_lidar=False) #   render_mode="human",
        self.n_actions = 2  # should be 2: 0 (do nothing), 1 (flap)

        self.num_buckets_distance = 80  # Number of buckets for horizontal distance to the next pipe
        self.num_buckets_velocity = 20  # Number of buckets for vertical velocity
        self.num_buckets_position = 60  # Number of buckets for vertical position
        self.num_buckets_ver_dis = 30
        self.action_space = [0, 1]  # Possible actions


        self.alpha_decay = 0.95  # Learning rate decay factor
        self.alpha_min = 0.01  # Minimum value of learning rate
        self.score_threshold = 20  # Score threshold for decaying learning rate
                
        self.Q = {}
        for i in range(self.num_buckets_distance):
            for j in range(self.num_buckets_velocity):
                for j2 in range(self.num_buckets_position):
                    for k in range(self.num_buckets_ver_dis):
                        for action in self.action_space:
                            self.Q[(i, j,j2, k), action] = 0.0
            
    def adjust_learning_rate(self, highest_score, alpha):
        # Reduce the learning rate if the highest score surpasses multiple of score_threshold
        if highest_score // self.score_threshold > (highest_score - self.scores[-1]) // self.score_threshold:
            new_alpha = max(alpha * self.alpha_decay, self.alpha_min)
            print(f"Adjusting learning rate from {self.alpha} to {new_alpha}")
            return new_alpha
        return alpha

    def maxAction(self, state, actions=[0,1]):
        
        values = np.array([self.Q[state,a] for a in actions])
        action = np.argmax(values)

        return action

    def save_q_values(self):
        print("saving q value")
        q_table_str_keys = {str(key): value for key, value in self.Q.items()}
        print("saving q valu2")
        with open("../q_table.json", "w") as file:
            json.dump(q_table_str_keys, file)
        print("saving q value3")

    def get_state(self, observation):
        # Extract the state array from the observation tuple
        if isinstance(observation, tuple):
            state_array = observation[0]
        else:
            state_array = observation
            
        # # Returning the indices directly
        # return (dist_index, velocity_index, diff_top_index, diff_bottom_index)
        max_values = [1.0,    1.0,   0.75,  0.43]  # Maximum observed values for each feature
        min_values = [-0.181, -0.9, -0.18,    0]  # Minimum observed values for each feature

        # Extract features from the state array using the correct indices
        horizontal_distance_to_next_pipe = state_array[0]
        vertical_velocity = state_array[10]
        vertical_position = state_array[9]
        vertical_distance_to_center_of_pipe = (state_array[1]+ state_array[2])/2
        
        
        dist_index = int(((horizontal_distance_to_next_pipe - min_values[0]) / (max_values[0] - min_values[0])) * self.num_buckets_distance)
        dist_index = min(max(dist_index, 0), self.num_buckets_distance - 1) 

        velocity_index = int(((vertical_velocity - min_values[1]) / (max_values[1] - min_values[1])) * self.num_buckets_velocity)
        velocity_index = min(max(velocity_index, 0), self.num_buckets_velocity - 1)
        
        
        vertical_pos = int(((vertical_position - min_values[2]) / (max_values[2] - min_values[2])) * self.num_buckets_position)
        vertical_pos = min(max(vertical_pos, 0), self.num_buckets_position - 1)

        vertical_dis = int(((vertical_distance_to_center_of_pipe - min_values[3]) / (max_values[3] - min_values[3])) * self.num_buckets_ver_dis)
        vertical_dis = min(max(vertical_dis, 0), self.num_buckets_ver_dis - 1)
        

        return (dist_index, velocity_index, vertical_pos, vertical_dis)


    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()  # Explore
        else:
            return self.maxAction(state) # Exploit first 1.0 4th 0.5 if third is 00.4 jump, not no jump

    def train(self):
        # Training loop
        num_episodes =   1000
        highest_score = 0
        total_score = 0  # Total score accumulator
        for episode in range(num_episodes):
            obs = self.env.reset()

            state = self.get_state(obs)
            
            action = self.choose_action( state)
            total_reward = 0

            while True:
                next_obs, reward, done, _, info = self.env.step(action)
                next_state = self.get_state(next_obs)
                next_action = self.choose_action(next_state)
                
                # SARSA update formula
                # SARSA Update
                self.Q[state, action] += self.alpha * (reward + self.gamma * self.Q[next_state, next_action] - self.Q[state, action])
                state, action = next_state, next_action
                total_reward += reward
                if done:
                    self.scores.append(total_reward)
                    total_score += total_reward
                    self.average_scores.append(total_score / (episode + 1))  # Calculate the average score
                    if total_reward > highest_score:
                        print("on epispde:", episode)
                        highest_score = total_reward
                        self.alpha = self.adjust_learning_rate(highest_score, self.alpha)
                        print(f"New highest score: {highest_score}")
                        with open("highest_score.txt", "w") as f:
                            f.write(str(highest_score))
                    self.highest_scores.append(highest_score)
                    break
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)  # Decay epsilon    

        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S") 
        
        lock = FileLock(self.lock_path, timeout=10)
        with lock:  # Using the lock as a context manager ensures it's released properly
            np.save(f"./sarsa/best_q_table_{formatted_time}.npy", self.Q)
            # self.save_q_values()
        self.env.close()
        print("highest", highest_score)
        print("Last average score:", self.average_scores[-1])


        plt.figure(figsize=(12, 6))
        plt.plot(self.scores, label='Score per Episode', alpha=0.5)  # Plot the scores per episode
        plt.plot(self.highest_scores, label='Highest Score', color='r')  # Plot the highest score progression
        plt.plot(self.average_scores, label='Average Score', color='g')  # Plot the average score progression
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.title('Progression of Scores Over Episodes')
        plt.legend()
        plt.show()

        