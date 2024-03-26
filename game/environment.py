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
        self.numbers = load_sprite("numbers", scale=1.5)
        self.score_off = [-50, 0, 50]
        self.score = 0

    # Draws environment elements and objects on window
    def draw(self, window):
        for anchor in self.bg_anchors[NIGHT]:
            window.blit(self.bg_images[NIGHT], (anchor[0], anchor[1]))

    # Draws current score in the environment (maybe show at the end)
    def draw_score(self, window):
        nums = [self.score // 100, self.score // 10 % 10, self.score % 10]
        for i in range(len(nums)):
            window.blit(self.numbers[nums[i]], (WIDTH // 2 + self.score_off[i] - 12, 80))
