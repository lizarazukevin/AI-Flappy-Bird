"""
File: replay.py
Description: Replays a previous game session from JSON file
Modified By: Kevin Lizarazu
Date: 03/20/2024
"""
import pygame.time
import random

from collections import deque

from game.player import Player
from game.environment import *
from game.object import *

# Load config macros
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
PIPE_STYLE = conf["GAME"]["PIPE_STYLE"]
PIPE_W = conf["GAME"]["PIPE_W"]
PIPE_H = conf["GAME"]["PIPE_H"]
PIPE_GAP = conf["GAME"]["PIPE_GAP"]
PIPE_FREQ = conf["GAME"]["PIPE_FREQ"]
PIPE_INTERVAL = conf["GAME"]["PIPE_INTERVAL"]
RAND_UPPER = conf["GAME"]["RAND_UPPER"]
RAND_LOWER = conf["GAME"]["RAND_LOWER"]

# Game env initialization
pygame.init()
pygame.display.set_caption("AI Flappy Bird")
window = pygame.display.set_mode((WIDTH, HEIGHT))


# Handles the replay of recorded game sessions
def main():
    # clock object to control ticks to maintain FPS
    clock = pygame.time.Clock()

    # environment setup
    env = Environment(WIDTH, HEIGHT, ENV_STYLE)

    # floor object created (scaled by 1.5)
    floor = Floor(0, HEIGHT - FLOOR_H, FLOOR_W, FLOOR_H, FLOOR_STYLE)
    ground_scroll = 0

    # pipe parameters and storage (scaled by 1.5)
    last_pipe = pygame.time.get_ticks()
    pipes = deque()
    pass_pipe = False

    # player creation
    player = Player(WIDTH // 2 - 34, (HEIGHT - FLOOR_H) // 2, BIRD_W, BIRD_H, BIRD_STYLE, ANIMATION_DELAY)

    # load desired session
    data = load_json(conf["LOAD_DIR"])
    random.seed(data["random_seed"])
    total_runs = data["total_runs"]

    # runs the loop in state-machine fashion
    curr_run = -1
    fps_counter = 0
    input_counter = 0
    total_inputs = 0
    total_run_fps = 0
    mode = "START"
    run = True

    while run:
        # maintains game running at specified FPS
        clock.tick(data["FPS"])

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break

            # searches for events, left click begins playing run of the session
            # runs can be skipped by simply clicking
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # only replays if current run does not exceed total runs
                if curr_run < total_runs - 1:
                    curr_run += 1
                else:
                    run = False
                    break

                total_inputs = len(data["run_inputs"][curr_run])
                total_run_fps = data["run_time"][curr_run]
                fps_counter = 0
                input_counter = 0

                pipes = deque()
                last_pipe = pygame.time.get_ticks()
                pass_pipe = False

                player.reset(WIDTH // 2 - 34, (HEIGHT - FLOOR_H) // 2)
                player.score = 0

                mode = "ALIVE"

        # simple start loop iterates through player sprite animation
        if mode == "START":
            last_pipe = pygame.time.get_ticks()
            player.start_loop()
        else:
            # searches for next input in relation to recorded time frame
            if input_counter < total_inputs:
                action, at_fps = data["run_inputs"][curr_run][input_counter]
                if fps_counter == at_fps:
                    player.jump(GRAVITY)
                    input_counter += 1

            # control scoring, increment when player passes pipe
            if len(pipes):
                if player.rect.left > pipes[0].rect.left \
                        and player.rect.right < pipes[0].rect.right \
                        and not pass_pipe:
                    pass_pipe = True

                if pass_pipe and player.rect.left > pipes[0].rect.right:
                    player.score += 1
                    pass_pipe = False

            # generate new pipes at random heights
            time_now = pygame.time.get_ticks()
            if time_now - last_pipe > PIPE_FREQ:
                pipe_height = random.randint(RAND_LOWER, RAND_UPPER)
                pipe_u = Pipe(WIDTH, (HEIGHT // 2) + pipe_height, PIPE_W, PIPE_H, PIPE_GAP, PIPE_STYLE, True)
                pipe_d = Pipe(WIDTH, (HEIGHT // 2) + pipe_height, PIPE_W, PIPE_H, PIPE_GAP, PIPE_STYLE)
                pipes.append(pipe_u)
                pipes.append(pipe_d)
                last_pipe = time_now

            # dequeues and updates pipes as they transition across window
            pipe_off = False
            for pipe in pipes:
                if pipe.loop(SCROLL_SPEED):
                    pipe_off = True
            if pipe_off:
                pipes.popleft()
                pipes.popleft()

            # update player sprite while still alive
            player.loop(FPS, GRAVITY)

            # control ground scrolling
            ground_scroll -= SCROLL_SPEED
            if abs(ground_scroll) > 35:
                ground_scroll = 0

            # only increase if still alive
            if fps_counter < total_run_fps:
                fps_counter += 1
            else:
                mode = "START"
                env.set_highscore(player)

        # draw all updated sprites
        draw(window, env, player, floor, pipes, ground_scroll)

    print("Finished Replay!")
    pygame.quit()
    quit()


if __name__ == "__main__":
    print("Welcome to the Flappy Bird Replay Station!")
    main()
