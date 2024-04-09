"""
File: util.py
Description: Macros and helpful general methods (macros to be converted to json)
Modified By: Kevin Lizarazu
Date: 03/20/2024
"""
import pygame
import json

from os import listdir
from os.path import isfile, join
from datetime import datetime


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
def load_env_sprites(w_width, w_height):
    sprites = load_sprite("env", scale=1.5)

    # figure out the anchor points to print these at
    anchors_all = []
    for i in range(len(sprites)):
        _, _, width, height = sprites[i].get_rect()

        # fill the entire window
        anchors = []
        for j in range((w_width // width) + 1):
            pos = (j * width, 0)
            anchors.append(pos)

        anchors_all.append(anchors)

    return anchors_all, sprites


# Draws the players, images, and objects onto the world environment
def draw(window, env, player, floor, pipes, ground_scroll):
    env.draw(window)

    for p in pipes:
        p.draw(window, p.rect.x)

    floor.draw(window, ground_scroll)
    player.draw(window, )

    # draws current and high score
    env.draw_score(window, player.get_score(), window.get_width() // 4, 690)
    env.draw_score(window, env.get_highscore(), 3 * window.get_width() // 4, 690)

    pygame.display.update()


# Loads information about JSON file here
def load_json(filename):
    with open(filename) as f:
        data = json.load(f)

    return data


# Saves dictionary object to JSON format
def save_json(play_record, filename=None):
    if not filename:
        filename = datetime.now().strftime("%Y%m%d%H%M%S.json")
    with open("run_sessions/" + filename, "w") as f:
        json.dump(play_record, f)
