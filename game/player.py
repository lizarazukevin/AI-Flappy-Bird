"""
File: player.py
Description: Holds various properties for the player object
Modified By: Kevin Lizarazu
Date: 03/20/2024
"""
import pygame

from game.util import load_sprite


class Player(pygame.sprite.Sprite):

    def __init__(self, x, y, width, height, bird_style, animation_delay):
        super().__init__()
        self.rect = pygame.Rect(x, y, width, height)
        self.x_vel = 0
        self.y_vel = 0
        self.animation_count = 0
        self.fall_count = 0
        self.hit = False
        self.mask = None
        self.sprites = load_sprite("birds", bird_style, scale=1.3)
        self.sprite = self.sprites[0]
        self.animation_delay = animation_delay

    # Repositions player for replayability
    def reset(self, x, y):
        self.rect.x = x
        self.rect.y = y
        self.y_vel = 0
        self.sprite = pygame.transform.rotate(self.sprite, 90)
        self.hit = False
        self.score = 0

    # Action for jumping, resets fall count
    def jump(self, gravity):
        self.y_vel = -gravity * 6
        self.fall_count = 0

    # Updates values when landing on an object
    def landed(self):
        self.fall_count = 0
        self.y_vel = 0

    # Updates the position of the player
    def move(self, dx, dy):
        self.rect.x += dx
        self.rect.y += dy

    # Hit an object, player nosedives
    def hit_object(self):
        self.hit = True
        self.animation_count = 0
        self.sprite = pygame.transform.rotate(self.sprites[1], -90)

    # Handles player movement given current x and y velocities
    def loop(self, gravity):
        self.y_vel += min(1, (self.fall_count / 60) * gravity)
        self.move(self.x_vel, self.y_vel)
        self.fall_count += 1
        self.update_sprite()

    # Animation loop for starting the game
    def start_loop(self):
        self.update_sprite()

    # Bird actions after getting hit
    def end_loop(self):
        self.y_vel += 1
        self.move(self.x_vel, self.y_vel)

    # Updates the player obj_sprite and mask depending on animation count and delay
    def update_sprite(self):
        sprite_index = (self.animation_count // self.animation_delay) % len(self.sprites)
        self.sprite = pygame.transform.rotate(self.sprites[sprite_index], -2 * self.y_vel)
        self.mask = pygame.mask.from_surface(self.sprite)
        self.animation_count += 1