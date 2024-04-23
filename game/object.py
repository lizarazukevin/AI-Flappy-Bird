"""
File: object.py
Description: Objects in game that can be collided with
Modified By: Kevin Lizarazu
Date: 03/20/2024
"""
import pygame.sprite

from game.util import load_sprite


# General class for objects
class Object(pygame.sprite.Sprite):
    def __init__(self, x, y, width, height, name=None):
        super().__init__()
        self.rect = pygame.Rect(x, y, width, height)
        self.image = pygame.Surface((width, height), pygame.SRCALPHA)
        self.floor_sprite, self.green_sprite, self.red_sprite = load_sprite("objects", scale=1.5)
        self.width = width
        self.height = height
        self.name = name

    def draw(self, window, dx):
        window.blit(self.image, (dx, self.rect.y))


# Constructor for a Floor object
class Floor(Object):
    def __init__(self, x, y, width, height):
        super().__init__(x, y, width, height)
        self.image.blit(self.floor_sprite, (0, 0))
        self.mask = pygame.mask.from_surface(self.image)


# Constructor for a Pipe object
class Pipe(Object):
    def __init__(self, x, y, width, height, pipe_gap, pipe_style, flip=False):
        super().__init__(x, y, width, height)
        self.flip = flip

        self.pipe_sprite = self.green_sprite
        if pipe_style == "red":
            self.pipe_sprite = self.red_sprite

        if flip:
            self.pipe_sprite = pygame.transform.flip(self.pipe_sprite, False, True)
            self.rect.bottomleft = [x, y - (pipe_gap // 2)]
        else:
            self.rect.topleft = [x, y + (pipe_gap // 2)]

        self.image.blit(self.pipe_sprite, (0, 0))
        self.mask = pygame.mask.from_surface(self.image)

    # Update behavior for pipes, returns True if off-screen
    def loop(self, scroll_speed):
        self.rect.x -= scroll_speed

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
