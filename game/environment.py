"""
Project: AI Flappy Bird
Class: CS5804
Modified By: Kevin Lizarazu
Date: 03/20/2024
"""
from game.util import *


class Environment:
    def __init__(self):
        self.bg_anchors, self.bg_images = load_env_sprites()
        self.numbers = load_sprite("numbers")
        self.score = 0

    # Draws environment elements and objects on window
    def draw(self, window, offset_x):
        for anchor in self.bg_anchors[NIGHT]:
            window.blit(self.bg_images[NIGHT], (anchor[0] - offset_x, anchor[1]))
