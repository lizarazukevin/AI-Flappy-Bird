import flappy_bird_gymnasium
import gymnasium
import numpy as np
import matplotlib.pyplot as plt
import json
import sys
import time
from filelock import Timeout, FileLock

    
lock_path = "best_q_table.lock"  
scores = []
highest_scores = [] 
average_scores = []  # List to store average scores progression

    
alpha = 0.3   # learning rate   low = slow learning, high = fast learning
gamma = 0.85   # discount factor  low = immediate reward, high = future reward
epsilon = 0.3  # exploration rate
epsilon_decay = 0.99  # Slower decay rate
epsilon_min = 0.05  # Minimum value of epsilon

# Environment

env = gymnasium.make("FlappyBird-v0",   use_lidar=False) #   render_mode="human",
n_actions = 2  # should be 2: 0 (do nothing), 1 (flap)

num_buckets_distance = 60  # Number of buckets for horizontal distance to the next pipe
num_buckets_velocity = 10  # Number of buckets for vertical velocity
num_buckets_position = 40  # Number of buckets for vertical position
num_buckets_ver_dis = 25
action_space = [0, 1]  # Possible actions

        
Q = {}
for i in range(num_buckets_distance):
    for j in range(num_buckets_velocity):
        for j2 in range(num_buckets_position):
            for k in range(num_buckets_ver_dis):
                for action in action_space:
                    Q[(i, j,j2, k), action] = 0.0
        


def maxAction(Q, state, actions=[0,1]):
    
    values = np.array([Q[state,a] for a in actions])
    action = np.argmax(values)

    return action

    

def get_state(observation):
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
    
    
    dist_index = int(((horizontal_distance_to_next_pipe - min_values[0]) / (max_values[0] - min_values[0])) * num_buckets_distance)
    dist_index = min(max(dist_index, 0), num_buckets_distance - 1) 

    velocity_index = int(((vertical_velocity - min_values[1]) / (max_values[1] - min_values[1])) * num_buckets_velocity)
    velocity_index = min(max(velocity_index, 0), num_buckets_velocity - 1)
    
    
    vertical_pos = int(((vertical_position - min_values[2]) / (max_values[2] - min_values[2])) * num_buckets_position)
    vertical_pos = min(max(vertical_pos, 0), num_buckets_position - 1)

    vertical_dis = int(((vertical_distance_to_center_of_pipe - min_values[3]) / (max_values[3] - min_values[3])) * num_buckets_ver_dis)
    vertical_dis = min(max(vertical_dis, 0), num_buckets_ver_dis - 1)
    

    return (dist_index, velocity_index, vertical_pos, vertical_dis)


def choose_action(state):
    if np.random.rand() < epsilon:
        return env.action_space.sample()  # Explore
    else:
        return maxAction(Q, state) # Exploit first 1.0 4th 0.5 if third is 00.4 jump, not no jump

best_actions = []
current_episode_actions = []
# Training loop
num_episodes =   2000000
highest_score = 0
total_score = 0  # Total score accumulator
for episode in range(num_episodes):
    obs = env.reset()

    state = get_state(obs)
    
    action = choose_action( state)
    total_reward = 0

    while True:
        next_obs, reward, done, _, info = env.step(action)
        next_state = get_state(next_obs)
        next_action = choose_action(next_state)
        
        # SARSA update formula
        # SARSA Update
        Q[state, action] += alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action])
        state, action = next_state, next_action
        total_reward += reward
        if done:
            scores.append(total_reward)
            total_score += total_reward
            average_scores.append(total_score / (episode + 1))  # Calculate the average score
            if total_reward > highest_score:
                print("on epispde:", episode)
                highest_score = total_reward
                print(f"New highest score: {highest_score}")
                with open("highest_score.txt", "w") as f:
                    f.write(str(highest_score))
            highest_scores.append(highest_score)
            break
    epsilon = max(epsilon_min, epsilon * epsilon_decay)  # Decay epsilon    

lock = FileLock(lock_path, timeout=10)
with lock:  # Using the lock as a context manager ensures it's released properly
    np.save("best_q_table.npy", Q)
env.close()
print("highest", highest_score)
print("Last average score:", average_scores[-1])


plt.figure(figsize=(12, 6))
plt.plot(scores, label='Score per Episode', alpha=0.5)  # Plot the scores per episode
plt.plot(highest_scores, label='Highest Score', color='r')  # Plot the highest score progression
plt.plot(average_scores, label='Average Score', color='g')  # Plot the average score progression
plt.xlabel('Episode')
plt.ylabel('Score')
plt.title('Progression of Scores Over Episodes')
plt.legend()
plt.show()

    