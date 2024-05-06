"""
File: manualGame.py
Description: Default game mode that uses taps to jump, no AI
Modified By: Kevin Lizarazu
Date: 03/20/2024
"""
import random
import numpy as np

from datetime import datetime

from game.util import load_json, save_json
from game.game import ManualGame

# load JSON variables
conf = load_json("config.json")

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

def main():
    print("Playing Manual Flappy Bird...")

    # game env created
    game = ManualGame(
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

    # record of player runs
    rec = {
        "run_inputs": [], 
        "run_scores": [], 
        "run_time": [], 
        "random_seed": datetime.now().timestamp(),
        "total_runs": 0, 
        "FPS": FPS
        }
    random.seed(rec["random_seed"])
    inputs = []

    while True:
        action, fps_count, score, done = game.play_step()

        # checks done condition
        # 0 -> not done, 
        # 1 -> done and record, 
        # 2 -> done but don't record, 
        # -1 -> completely done
        if done == -1:
            break

        if done == 1:
            rec["run_inputs"].append(inputs)
            rec["run_scores"].append(score)
            rec["run_time"].append(fps_count)
            rec["total_runs"] += 1
            inputs = []

        # captures jump action and frame
        if np.array_equal(action, [1, 0]):
            inputs.append([action, fps_count])
    
    # game exits, save for replay
    save_json(rec)
    quit()
    

if __name__ == "__main__":
    main()