"""
File: game.py
Description: Contains all necessary flappy bird game environment elements
Modified By: Kevin Lizarazu
Date: 04/21/2024
"""
import pygame
import random
import numpy as np
import math

from game.util import load_env_sprites
from game.player import Player
from game.object import Floor, Pipe

from enum import Enum
from collections import namedtuple, deque

# initialize pygame
pygame.init()
pygame.display.set_caption("AI Flappy Bird")
pygame_icon = pygame.image.load("./assets/sprites/birds/red/redbird-downflap.png")
pygame.display.set_icon(pygame_icon)

BG_STYLE = {
    "day": 0,
    "night": 1
}

PIPE_STYLE = {
    "green": 1,
    "red": 2
}

font = pygame.font.Font('./game/arial.ttf', 25)

class BaseGame:
    def __init__(self, window_dims, floor_dims, bird_dims, pipe_dims, rand_dims, env_style, bird_style, animation_delay, scroll_speed, gravity, fps):
        self.w, self.h = window_dims
        self.floor_w, self.floor_h = floor_dims
        self.bird_w, self.bird_h = bird_dims
        self.pipe_w, self.pipe_h, self.pipe_gap, self.pipe_color = pipe_dims
        self.rand_lower, self.rand_upper = rand_dims
        self.env_style = env_style
        self.scroll_speed = scroll_speed
        self.gravity = gravity
        self.fps = fps

        # based off of pipe spacing at 60 fps
        self.pipe_freq = (1500 * 60) / self.fps

        self.window = pygame.display.set_mode((self.w, self.h))
        self.clock = pygame.time.Clock()

        self.bg_anchors, self.bg_sprites = load_env_sprites(self.w, self.h)
        self.floor = Floor(0, self.h - self.floor_h, self.floor_w, self.floor_h)
        self.player = Player(self.w // 2 - 34, (self.h - self.floor_h) // 2, self.bird_w, self.bird_h, bird_style, animation_delay)

        self.reset()

    # Resets game environment
    def reset(self):
        self.player.reset(self.w // 2 - 34, (self.h - self.floor_h) // 2)
        self.score = 0
        self.fps_counter = 0
        self.pipes = deque()
        self.pass_pipe = False
        self.last_pipe = pygame.time.get_ticks()
    
    # At every frame an action is taken, resulting in reward, done, score
    def play_step(self, action):
        raise NotImplementedError("Method 'play_step' must be implemented in derived classes.")
    
    # Handles collisions in the environment
    def is_collision(self):
        # player cannot fly higher than cieling or lower than floor
        if self.player.rect.bottom < 0 or self.player.rect.bottom > self.h - self.floor_h:
            return True
        
        # check if player collides with any pipes
        for pipe in self.pipes:
            if pygame.sprite.collide_mask(self.player, pipe):
                return True

        return False
    
    # Draws to window the background, floor, pipes, and player
    def _render(self):
        for anchor in self.bg_anchors:
            self.window.blit(self.bg_sprites[BG_STYLE[self.env_style]], anchor)

        for pipe in self.pipes:
            self.window.blit(pipe.image, (pipe.rect.x, pipe.rect.y))

        self.window.blit(self.floor.image, (self.floor.rect.x, self.floor.rect.y))
        self.window.blit(self.player.sprite, (self.player.rect.x, self.player.rect.y))

        text = font.render("Score: " + str(self.score), True, (255, 255, 255))
        self.window.blit(text, (0, 0))

        pygame.display.flip()
    
    # Updates positions of all sprites
    def _update(self, action):
        raise NotImplementedError("Method '_update' must be implemented in derived classes.")

# Game environment used for SARSA training
class FlappyGameSARSA(BaseGame):
    def __init__(self, window_dims, floor_dims, bird_dims, pipe_dims, rand_dims, env_style, bird_style, animation_delay, scroll_speed, gravity, fps):
        super().__init__(window_dims, floor_dims, bird_dims, pipe_dims, rand_dims, env_style, bird_style, animation_delay, scroll_speed, gravity, fps)
        self.states_dim = 8
        self.actions_dim = 2

    def get_state():
        pass

    # At every frame an action is taken, resulting in reward, done, score
    def play_step(self, action):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # update player and object sprites
        self._update(action)

        # render each frame
        self.fps_counter += 1
        # print(f"At frame {self.fps_counter}, height is: {self.player.rect.top} and {'DEAD' if self.player.hit else 'ALIVE'}")
        self.clock.tick(self.fps)
        self._render()

        # determine score and rewards
        # living reward -> +0.1
        # crash reward -> -1
        # pass pipe reward -> 1
        rew, done = -0.1, False
        if self.is_collision():
            self.player.hit_object()
            done = True
            rew -= 1
            # print(f"At frame {self.fps_counter}, Reward: {rew}")
            return rew, done, self.score

        if len(self.pipes):
            if self.player.rect.left > self.pipes[0].rect.left \
                and self.player.rect.right < self.pipes[0].rect.right \
                and not self.pass_pipe:
                self.pass_pipe = True
            if self.pass_pipe and self.player.rect.left > self.pipes[0].rect.right:
                self.pass_pipe = False
                self.score += 1
                rew += 1
    
        return rew, done, self.score

    # Updates positions of all sprites
    def _update(self, action):
        # action is jump, else do nothing 
        if np.array_equal(action, [1, 0]) and not self.player.hit:
            self.player.jump(self.gravity)
        
        # determines which sprite update loop for player
        if self.player.hit:
            if self.player.rect.bottom > self.h - self.floor_h:
                self.player.landed()
            else:
                self.player.end_loop()
            return
        else:
            self.player.loop(self.gravity)

        # pipe updates --> generation and removal
        time_now = pygame.time.get_ticks()
        if time_now - self.last_pipe > self.pipe_freq:
            pipe_y = random.randint(self.rand_lower, self.rand_upper)
            pipe_u = Pipe(self.w, (self.h // 2) + pipe_y, self.pipe_w, self.pipe_h, self.pipe_gap, PIPE_STYLE[self.pipe_color], True)
            pipe_d = Pipe(self.w, (self.h // 2) + pipe_y, self.pipe_w, self.pipe_h, self.pipe_gap, PIPE_STYLE[self.pipe_color])
            self.pipes += [pipe_d, pipe_u]
            self.last_pipe = time_now

        pipe_off = False
        for pipe in self.pipes:
            if pipe.loop(self.scroll_speed):
                pipe_off = True
                break
        
        if pipe_off:
            self.pipes.popleft()
            self.pipes.popleft()

        # handle ground scroll
        self.floor.loop(self.scroll_speed)

# Game environment used for DQN training
class FlappyGameDQN(BaseGame):
    def __init__(self, window_dims, floor_dims, bird_dims, pipe_dims, rand_dims, env_style, bird_style, animation_delay, scroll_speed, gravity, fps):
        super().__init__(window_dims, floor_dims, bird_dims, pipe_dims, rand_dims, env_style, bird_style, animation_delay, scroll_speed, gravity, fps)

    # At every frame an action is taken, resulting in reward, done, score
    def play_step(self, action):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # update player and object sprites
        self._update(action)

        # render each frame
        self.fps_counter += 1
        # print(f"At frame {self.fps_counter}, height is: {self.player.rect.top} and {'DEAD' if self.player.hit else 'ALIVE'}")
        self.clock.tick(self.fps)
        self._render()

        # determine score and rewards
        # living reward -> +0.1
        # crash reward -> -1
        # pass pipe reward -> 1
        rew, done = 0.1, False
        if self.is_collision():
            self.player.hit_object()
            done = True
            rew -= 1000
            # print(f"At frame {self.fps_counter}, Reward: {rew}")
            return rew, done, self.score

        # increaase living reward if too close to floor/ceiling -> [-100, -1]
        # rew -= 100 / math.exp(self.player.rect.bottom)
        # rew -= 100 / math.exp(self.h - self.floor_h - self.player.rect.top)

        if len(self.pipes):
            if self.player.rect.left > self.pipes[0].rect.left \
                and self.player.rect.right < self.pipes[0].rect.right \
                and not self.pass_pipe:
                self.pass_pipe = True
            if self.pass_pipe and self.player.rect.left > self.pipes[0].rect.right:
                self.pass_pipe = False
                self.score += 1
                rew += 20
            
            # lessen reward closer to pipe -> [-1, 0]
            # if len(self.pipes) == 2:
            #     # print("pipes present")
            #     rew += (1 / (1 + math.exp((self.pipes[0].rect.right - self.player.rect.left - 200) / 36)))
            # elif len(self.pipes) == 4:
            #     rew += (1 / (1 + math.exp((self.pipes[2].rect.right - self.player.rect.left - 200) / 36)))
        
        # print(f"At frame {self.fps_counter}, Reward: {rew}")
        return rew, done, self.score

    # Updates positions of all sprites
    def _update(self, action):
        # action is jump, else do nothing 
        if np.array_equal(action, [1, 0]) and not self.player.hit:
            self.player.jump(self.gravity)
        
        # determines which sprite update loop for player
        if self.player.hit:
            if self.player.rect.bottom > self.h - self.floor_h:
                self.player.landed()
            else:
                self.player.end_loop()
            return
        else:
            self.player.loop(self.gravity)

        # pipe updates --> generation and removal
        time_now = pygame.time.get_ticks()
        if time_now - self.last_pipe > self.pipe_freq:
            pipe_y = random.randint(self.rand_lower, self.rand_upper)
            pipe_u = Pipe(self.w, (self.h // 2) + pipe_y, self.pipe_w, self.pipe_h, self.pipe_gap, PIPE_STYLE[self.pipe_color], True)
            pipe_d = Pipe(self.w, (self.h // 2) + pipe_y, self.pipe_w, self.pipe_h, self.pipe_gap, PIPE_STYLE[self.pipe_color])
            self.pipes += [pipe_d, pipe_u]
            self.last_pipe = time_now

        pipe_off = False
        for pipe in self.pipes:
            if pipe.loop(self.scroll_speed):
                pipe_off = True
                break
        
        if pipe_off:
            self.pipes.popleft()
            self.pipes.popleft()

        # handle ground scroll
        self.floor.loop(self.scroll_speed)

# Game environment for manual gameplay
class ManualGame(BaseGame):
    def __init__(self, window_dims, floor_dims, bird_dims, pipe_dims, rand_dims, env_style, bird_style, animation_delay, scroll_speed, gravity, fps, replay=False):
        super().__init__(window_dims, floor_dims, bird_dims, pipe_dims, rand_dims, env_style, bird_style, animation_delay, scroll_speed, gravity, fps)
        self.mode = "START"
        self.replay = replay

        self.reset()

    # At every frame an action is taken, resulting in reward, done, score
    def play_step(self, action=[0, 0]):
        action = list(action)

        # key events searched every play step
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                done = -1
                return action, self.fps_counter, self.score, done
            
            if not self.replay and event.type == pygame.KEYDOWN:
                if self.mode == "START":
                    self.mode = "ALIVE"
                    self.last_pipe = pygame.time.get_ticks()
                if event.key == pygame.K_SPACE and self.mode == "ALIVE":
                    action[0] = 1
                if event.key == pygame.K_r and self.mode == "END":
                    self.mode = "START"
                    self.reset()

        # update player and object sprites
        self._update(action)

        # only record the frames that bird is alive
        if self.mode == "ALIVE":
            self.fps_counter += 1

        # render each frame
        self.clock.tick(self.fps)
        self._render()

        done = 0
        if self.is_collision() and not self.player.hit:
            self.mode = "END"
            self.player.hit_object()
            done = 1
        else:
            done = 2

        if len(self.pipes):
            if self.player.rect.left > self.pipes[0].rect.left \
                and self.player.rect.right < self.pipes[0].rect.right \
                and not self.pass_pipe:
                self.pass_pipe = True
            if self.pass_pipe and self.player.rect.left > self.pipes[0].rect.right:
                self.pass_pipe = False
                self.score += 1

        return action, self.fps_counter - 1, self.score, done

    # Updates positions of all sprites
    def _update(self, action):
        # start game animation
        if self.mode == "START" and not self.replay:
            self.player.start_loop()
            return

        # action is jump, else do nothing 
        if np.array_equal(action, [1, 0]) and not self.player.hit:
            self.player.jump(self.gravity)
        
        # determines which sprite update loop for player
        if self.player.hit:
            if self.player.rect.bottom > self.h - self.floor_h:
                self.player.landed()
            else:
                self.player.end_loop()
            return
        else:
            self.player.loop(self.gravity)

        # pipe updates --> generation and removal
        time_now = pygame.time.get_ticks()
        if time_now - self.last_pipe > self.pipe_freq:
            pipe_y = random.randint(self.rand_lower, self.rand_upper)
            pipe_u = Pipe(self.w, (self.h // 2) + pipe_y, self.pipe_w, self.pipe_h, self.pipe_gap, PIPE_STYLE[self.pipe_color], True)
            pipe_d = Pipe(self.w, (self.h // 2) + pipe_y, self.pipe_w, self.pipe_h, self.pipe_gap, PIPE_STYLE[self.pipe_color])
            self.pipes += [pipe_d, pipe_u]
            self.last_pipe = time_now

        pipe_off = False
        for pipe in self.pipes:
            if pipe.loop(self.scroll_speed):
                pipe_off = True
                break
        
        if pipe_off:
            self.pipes.popleft()
            self.pipes.popleft()

        # handle ground scroll
        self.floor.loop(self.scroll_speed)