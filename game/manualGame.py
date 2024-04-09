"""
File: manualGame.py
Description: Default game mode that uses taps to jump, no AI
Modified By: Kevin Lizarazu
Date: 03/20/2024
"""
import pygame.time
import random

from collections import deque
from datetime import datetime

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


# Handles game objects and environment changes
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

    # record of entire session
    play_record = {"run_inputs": [], "run_scores": [], "run_time": [], "random_seed": datetime.now().timestamp(),
                   "total_runs": 0, "FPS": FPS}
    random.seed(play_record["random_seed"])
    fps_counter = 0
    inputs = []

    # runs the loop in state-machine fashion
    mode = "START"
    run = True

    while run:
        # maintains game running at specified FPS
        clock.tick(FPS)

        # game over condition
        if mode == "END":
            # handles aftereffects of collision
            if handle_collision(player, [floor]):
                player.landed()
            else:
                player.end_loop()

            # searches for input to restart the game
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    break

                # press r key to replay
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r and player.y_vel == 0:
                    inputs = []
                    pipes = deque()
                    last_pipe = pygame.time.get_ticks()
                    player.reset(WIDTH // 2 - 34, (HEIGHT - FLOOR_H) // 2)
                    player.score = 0
                    mode = "START"

            # updates sprites
            draw(window, env, player, floor, pipes, ground_scroll)
            continue

        # searching for user input events while game is still active
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break

            if event.type == pygame.KEYDOWN:
                # starts game to active
                if mode == "START":
                    mode = "ALIVE"
                    fps_counter = 0
                # space key is pressed for jump, records frame and action
                if event.key == pygame.K_SPACE and mode == "ALIVE":
                    player.jump(GRAVITY)
                    inputs.append(("jump", fps_counter))

        # simple start loop iterates through player sprite animation
        if mode == "START":
            last_pipe = pygame.time.get_ticks()
            player.start_loop()
        # identifies collisions with floor/pipe objects or if off-screen, ends run
        elif handle_collision(player, [floor, *pipes]) or player.rect.bottom < 0:
            mode = "END"
            env.set_highscore(player)
            player.hit_object()
            play_record["run_inputs"].append(inputs)
            play_record["run_time"].append(fps_counter)
            play_record["run_scores"].append(player.get_score())
            play_record["total_runs"] += 1
        # active game, no collisions, must update all sprites
        else:
            # when pipes exist, controls scoring based on if player passes through
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

        # draw all updated sprites
        draw(window, env, player, floor, pipes, ground_scroll)
        fps_counter += 1

    save_json(play_record, conf["SAVE_DIR"])
    pygame.quit()
    quit()


if __name__ == "__main__":
    main()
