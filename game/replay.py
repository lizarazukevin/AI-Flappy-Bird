"""
File: replay.py
Description: Replays a previous game session from JSON file
Modified By: Kevin Lizarazu
Date: 03/20/2024
"""
"""
File: manualGame.py
Description: Default game mode that uses taps to jump, no AI
Modified By: Kevin Lizarazu
Date: 03/20/2024
"""
import random
import numpy as np
import pygame

from datetime import datetime

from game.util import load_json, save_json
from game.game import FlappyGame

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
    # load desired session
    data = load_json(conf["LOAD_DIR"])
    random.seed(data["random_seed"])

    # game env created
    game = FlappyGame (
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
        data["FPS"],
        replay=True
    )

    # iterate through all games played
    for run in range(data["total_runs"]):
        print(f"Replaying Game: {run + 1}, Score: {data['run_scores'][run]}, Game Length: {data['run_time'][run]}")

        input_counter = 0
        game.reset()

        # at every frame feeds an action taken, jump only if it matches the record's input
        for frame in range(data["run_time"][run]):
            if input_counter < len(data["run_inputs"][run]) and data["run_inputs"][run][input_counter][1] == frame:
                game.play_step(action=data["run_inputs"][run][input_counter][0])
                input_counter += 1
            else:
                game.play_step()

    print("Finished Replay!")
    quit()
    

if __name__ == "__main__":
    main()
