"""
File: manualGame.py
Description: Default game mode that uses taps to jump, no AI
Modified By: Kevin Lizarazu
Date: 03/20/2024
"""
import pygame.time
import random

from collections import deque

from game.player import Player
from game.environment import *
from game.object import *

# Initializing game macros and environment window
pygame.init()
pygame.display.set_caption("AI Flappy Bird")
window = pygame.display.set_mode((WIDTH, HEIGHT))


# Populates and runs the game on a loop
def main():
    clock = pygame.time.Clock()

    # environment setup
    ground_scroll = 0
    env = Environment()

    # floor object created (scaled by 1.5)
    floor_w, floor_h = 504, 168
    floor = Floor(0, HEIGHT - floor_h, floor_w, floor_h)

    # pipe objects created (scaled by 1.5)
    pipe_w, pipe_h = 78, 480
    pipes = deque()
    last_pipe = pygame.time.get_ticks()
    pass_pipe = False

    # player creation
    player = Player(WIDTH // 2 - 34, (HEIGHT - floor_h) // 2, 51, 36)

    mode = "START"
    run = True

    while run:
        clock.tick(FPS)

        # game over condition
        if mode == "END":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    break

                # press r key to replay
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r and player.y_vel == 0:
                    pipes = deque()
                    last_pipe = pygame.time.get_ticks()
                    player.reset(WIDTH // 2 - 34, (HEIGHT - floor_h) // 2)
                    env.score = 0
                    mode = "START"

            # collision with floor
            if handle_collision(player, [floor]):
                player.landed()
            else:
                player.end_loop()

            # draw(window, env, player, floor, offset_x, pipes_bottom, pipes_top, True)
            draw(window, env, player, floor, pipes, ground_scroll, done=True)

            continue

        # searching for key events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break

            # handles event action for player jumping
            if event.type == pygame.KEYDOWN:
                mode = "ALIVE"
                if event.key == pygame.K_SPACE and mode == "ALIVE":
                    player.jump()

        # remaining actions for each mode
        if mode == "START":
            player.PLAYER_VEL = 3
            player.start_loop()
        elif handle_collision(player, [floor, *pipes]) or player.rect.bottom < 0:
            mode = "END"
            player.hit_object()
        else:
            # control scoring, increment when player passes pipe
            if len(pipes):
                if player.rect.left > pipes[0].rect.left \
                        and player.rect.right < pipes[0].rect.right \
                        and not pass_pipe:
                    pass_pipe = True

                if pass_pipe and player.rect.left > pipes[0].rect.right:
                    env.score += 1
                    pass_pipe = False

            # generate new pipes at random heights
            time_now = pygame.time.get_ticks()
            if time_now - last_pipe > PIPE_FREQ:
                pipe_height = random.randint(-80, 80)
                pipe_u = Pipe(WIDTH, (HEIGHT // 2) + pipe_height, pipe_w, pipe_h, True)
                pipe_d = Pipe(WIDTH, (HEIGHT // 2) + pipe_height, pipe_w, pipe_h)
                pipes.append(pipe_u)
                pipes.append(pipe_d)
                last_pipe = time_now

            # update player sprite
            player.loop()

            # update and dequeue pipes if travel off the screen
            pipe_off = False
            for pipe in pipes:
                if pipe.loop():
                    pipe_off = True
            if pipe_off:
                pipes.popleft()
                pipes.popleft()

            # control ground scrolling
            ground_scroll -= SCROLL_SPEED
            if abs(ground_scroll) > 35:
                ground_scroll = 0

        # draw all updated sprites
        draw(window, env, player, floor, pipes, ground_scroll)

    pygame.quit()
    quit()


if __name__ == "__main__":
    main()
