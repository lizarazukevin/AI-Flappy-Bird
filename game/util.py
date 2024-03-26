"""
File: util.py
Description: Macros and helpful general methods (macros to be converted to json)
Modified By: Kevin Lizarazu
Date: 03/20/2024
"""
import pygame

from os import listdir
from os.path import isfile, join

# Game-related macros
WIDTH, HEIGHT = 432, 768
FPS = 60
DAY, NIGHT = 0, 1
FLOOR, PIPE_G, PIPE_R = 0, 1, 2
PIPE_GAP = 150
PIPE_FREQ = 1500

# Map related macros
SCROLL_SPEED = 4
MAP_HEIGHT = HEIGHT
START_WIDTH = WIDTH
PIPE_INTERVAL = 300


# Loads an obj_sprite sheet from image-holding directory
def load_sprite(dir1, dir2=None, scale=1.0):

    # deals with single or embedded folders for sprites
    if dir2:
        path = join("assets", "sprites", dir1, dir2)
    else:
        path = join("assets", "sprites", dir1)

    images = [f for f in listdir(path) if isfile(join(path, f))]

    sprites = []
    for img in images:
        sprite = pygame.image.load(join(path, img)).convert_alpha()
        width, height = sprite.get_width(), sprite.get_height()
        surface = pygame.Surface((width, height), pygame.SRCALPHA, 32)
        rect = pygame.Rect(0, 0, width, height)
        surface.blit(sprite, (0, 0), rect)
        sprites.append(pygame.transform.scale_by(surface, scale))

    return sprites


# Loads background image and positions to draw them
def load_env_sprites():
    sprites = load_sprite("env", scale=1.5)

    # figure out the anchor points to print these at
    anchors_all = []
    for i in range(len(sprites)):
        _, _, width, height = sprites[i].get_rect()

        # fill the entire window
        anchors = []
        for j in range((WIDTH // width) + 1):
            pos = (j * width, 0)
            anchors.append(pos)

        anchors_all.append(anchors)

    return anchors_all, sprites


# Draws the players, images, and objects onto the world environment
def draw(window, env, player, floor, pipes, ground_scroll, done=False):
    env.draw(window)

    for p in pipes:
        p.draw(window, p.rect.x)

    floor.draw(window, ground_scroll)
    player.draw(window)

    if done:
        env.draw_score(window)

    pygame.display.update()

