"""
File: object.py
Description: Objects in game that can be collided with
Modified By: Kevin Lizarazu
Date: 03/20/2024
"""
import pygame.sprite

from game.util import *


# General class for objects
class Object(pygame.sprite.Sprite):
    def __init__(self, x, y, width, height, name=None):
        super().__init__()
        self.rect = pygame.Rect(x, y, width, height)
        self.image = pygame.Surface((width, height), pygame.SRCALPHA)
        self.obj_sprite = load_sprite("objects", scale=1.5)
        self.width = width
        self.height = height
        self.name = name

    def draw(self, window, dx):
        window.blit(self.image, (dx, self.rect.y))


# Constructor for a Floor object
class Floor(Object):
    def __init__(self, x, y, width, height):
        super().__init__(x, y, width, height)
        self.image.blit(self.obj_sprite[FLOOR], (0, 0))
        self.mask = pygame.mask.from_surface(self.image)


# Constructor for a Pipe object
class Pipe(Object):
    def __init__(self, x, y, width, height, flip=False):
        super().__init__(x, y, width, height)
        self.flip = flip
        self.obj_sprite = self.obj_sprite[PIPE_G]

        if flip:
            self.obj_sprite = pygame.transform.flip(self.obj_sprite, False, True)
            self.rect.bottomleft = [x, y - (PIPE_GAP // 2)]
        else:
            self.rect.topleft = [x, y + (PIPE_GAP // 2)]

        self.image.blit(self.obj_sprite, (0, 0))
        self.mask = pygame.mask.from_surface(self.image)

    # Update behavior for pipes, returns True if off-screen
    def loop(self):
        self.rect.x -= SCROLL_SPEED

        if self.rect.right < 0:
            return True
        return False


# Returns the exact object player collides with
def handle_collision(player, objects):
    # find collisions
    floor = False
    collided_objects = []
    for obj in objects:
        if pygame.sprite.collide_mask(player, obj):
            collided_objects.append(obj)

            if type(obj) == Floor:
                floor = True

    # in collisions find if it hit the floor
    if floor:
        player.landed()
        return True

    return collided_objects
