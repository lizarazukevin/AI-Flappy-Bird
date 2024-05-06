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

    anchors = []
    _, _, width, height = sprites[0].get_rect()
    for i in range((w_width // width) + 1):
        pos = (i * width, 0)
        anchors.append(pos)

    return anchors, sprites


# Loads information about JSON file here
def load_json(filename):
    with open(filename) as f:
        data = json.load(f)

    return data


# Saves dictionary object to JSON format
def save_json(play_record, filename=None):
    if not filename:
        filename = datetime.now().strftime("%Y%m%d%H%M%S.json")
    with open("./manual_run_sessions/" + filename, "w") as f:
        json.dump(play_record, f)
