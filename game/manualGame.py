"""
Project: AI Flappy Bird
Class: CS5804
Modified By: Kevin Lizarazu
Date: 03/20/2024
Description: Manual mode of playing Flappy Bird
"""
import os

import pygame

from os import listdir
from os.path import isfile, join

from game.player import Player

# Initializing game macros and env
pygame.init()
pygame.display.set_caption("AI Flappy Bird")
WIDTH, HEIGHT = 1000, 768
FPS = 60
PLAYER_VEL = 5

window = pygame.display.set_mode((WIDTH, HEIGHT))


# Loads background image and positions to draw them
def get_background(name):
    path = join(os.getcwd(), "assets", "sprites", "env", name)
    bg_img = pygame.transform.scale_by(pygame.image.load(path), 1.5)
    _, _, width, height = bg_img.get_rect()

    # Fill the entire window
    anchors = []
    for i in range(WIDTH // width + 1):
        pos = (i * width, 0)
        anchors.append(pos)

    return anchors, bg_img


# Draws the players, images, and objects onto the world environment
def draw(anchors, bg_img):

    for a in anchors:
        window.blit(bg_img, a)

    pygame.display.update()


# Populates and runs the game on a loop
def main():
    clock = pygame.time.Clock()
    anchors, bg_img = get_background("background-day.png")
    player = Player(10, 10, 10, 10)

    run = True
    while run:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break

        draw(anchors, bg_img)

    pygame.quit()
    quit()


if __name__ == "__main__":
    main()
