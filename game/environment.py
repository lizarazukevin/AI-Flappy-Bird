"""
Project: AI Flappy Bird
Class: CS5804
Modified By: Kevin Lizarazu
Date: 03/20/2024
"""
from game.util import *


class Environment:
    def __init__(self, width, height, env_style):
        self.bg_anchors, self.bg_images = load_env_sprites(width, height)
        self.numbers = load_sprite("numbers", scale=1.5)
        self.highScore = 0
        self.env_style = env_style

    # Draws environment elements and objects on window
    def draw(self, window):
        for anchor in self.bg_anchors[self.env_style]:
            window.blit(self.bg_images[self.env_style], (anchor[0], anchor[1]))

    # Sets new high score
    def set_highscore(self, player):
        self.highScore = max(self.highScore, player.get_score())

    # Getter for high score
    def get_highscore(self):
        return self.highScore

    # Draws current score in the environment (maybe show at the end)
    def draw_score(self, window, score, x, y):
        hund, tens, ones = score // 100, score // 10 % 10, score % 10

        if not hund == 0:
            window.blit(self.numbers[hund], (x - 50 - 12, y))
            window.blit(self.numbers[tens], (x - 12, y))
            window.blit(self.numbers[ones], (x + 50 - 12, y))
        elif not tens == 0:
            window.blit(self.numbers[tens], (x - 25 - 12, y))
            window.blit(self.numbers[ones], (x + 25 - 12, y))
        else:
            window.blit(self.numbers[ones], (x - 12, y))
