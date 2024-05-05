import numpy as np
import gymnasium
import matplotlib.pyplot as plt
import flappy_bird_gymnasium
import time
import json
from filelock import Timeout, FileLock

lock_path = "best_q_table.lock"

num_buckets_distance = 60  # Number of buckets for horizontal distance to the next pipe
num_buckets_velocity = 10  # Number of buckets for vertical velocity
num_buckets_position = 40  # Number of buckets for vertical position
num_buckets_ver_dis = 25
action_space = [0, 1]  # Possible actions

def load_q_table(filename="best_q_table.npy"):
    return np.load(filename, allow_pickle=True).item()

def max_action(Q, state, actions=[0,1]):
    values = np.array([Q[state,a] for a in actions])
    action = np.argmax(values)

    return action

def get_state(observation):
    if isinstance(observation, tuple):
        state_array = observation[0]
    else:
        state_array = observation
    max_values = [1.0,    1.0,   0.75,  0.43]  # Maximum observed values for each feature
    min_values = [-0.181, -0.9, -0.18,    0]  # Minimum observed values for each feature

    # Extract features from the state array using the correct indices
    horizontal_distance_to_next_pipe = state_array[0]
    vertical_velocity = state_array[10]
    vertical_position = state_array[9]
    vertical_distance_to_center_of_pipe = (state_array[1]+ state_array[2])/2
    
    
    # Discretize horizontal distance to the next pipe
    dist_index = int(((horizontal_distance_to_next_pipe - min_values[0]) / (max_values[0] - min_values[0])) * num_buckets_distance)
    dist_index = min(max(dist_index, 0), num_buckets_distance - 1) 

    # Discretize vertical velocity
    velocity_index = int(((vertical_velocity - min_values[1]) / (max_values[1] - min_values[1])) * num_buckets_velocity)
    velocity_index = min(max(velocity_index, 0), num_buckets_velocity - 1)
    
    
    vertical_pos = int(((vertical_position - min_values[2]) / (max_values[2] - min_values[2])) * num_buckets_position)
    vertical_pos = min(max(vertical_pos, 0), num_buckets_position - 1)
    
    vertical_dis = int(((vertical_distance_to_center_of_pipe - min_values[3]) / (max_values[3] - min_values[3])) * num_buckets_ver_dis)
    vertical_dis = min(max(vertical_dis, 0), num_buckets_ver_dis - 1)
    
    return (dist_index, velocity_index, vertical_pos, vertical_dis)



def load_best_actions(filename='best_actions.json'):
    with open(filename, 'r') as f:
        best_actions = json.load(f)
    return best_actions

def replay_best_episode():
    Q = load_q_table()

    # Setup environment
    env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False)
    obs = env.reset()    
    done = False

    # # Replay using the best Q-table
    while not done:
        state = get_state(obs)
        action = max_action(Q, state)
        

        if action ==1:
            if isinstance(obs, tuple):
                state_array = obs[0]
            else:
                state_array = obs
                
        obs, reward, done,  _, info = env.step(action)
        env.render()  # This will display the current environment state
        time.sleep(1 / 60) 

    env.close()
    
if __name__ == "__main__":
    replay_best_episode()