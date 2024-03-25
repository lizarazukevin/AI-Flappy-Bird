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
        self.obj_sprite = load_sprite("objects")
        self.width = width
        self.height = height
        self.name = name

    def draw(self, window, offset_x):
        window.blit(self.image, (self.rect.x - offset_x, self.rect.y))


class Floor(Object):
    def __init__(self, x, y, width, height):
        super().__init__(x, y, width, height)
        self.image.blit(self.obj_sprite[FLOOR], (0, 0))
        self.mask = pygame.mask.from_surface(self.image)


class Pipe(Object):
    def __init__(self, x, y, width, height, flip=False):
        super().__init__(x, y, width, height)
        self.flip = flip
        self.obj_sprite = self.obj_sprite[PIPE_G]

        if flip:
            self.obj_sprite = pygame.transform.rotate(self.obj_sprite, 180)

        self.image.blit(self.obj_sprite, (0, 0))
        self.mask = pygame.mask.from_surface(self.image)


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

# # Handle vertical collisions
# def vertical_collision(player, objects):
#     collided_objects = []
#     for obj in objects:
#         if pygame.sprite.collide_mask(player, obj):
#             if player.y_vel > 0 and type(obj) == Floor:
#                 player.rect.bottom = obj.rect.top
#                 player.landed()
#             elif player.y_vel < 0:
#                 player.rect.top = obj.rect.bottom
#                 player.hit_top()
#             collided_objects.append(obj)
#
#     return collided_objects
#
#
# # Handles horizontal collisions
# def horizontal_collision(player, objects, dx):
#     player.move(dx, 0)
#     player.update()
#
#     collided_object = None
#     for obj in objects:
#         if pygame.sprite.collide_mask(player, obj):
#             collided_object = obj
#             break
#
#     player.move(-dx, 0)
#     player.update()
#     return collided_object
#
#
# # Handle all collisions
# def handle_collisions(player, objects):
#     player.x_vel = 0
#     right_coll = horizontal_collision(player, objects, player.PLAYER_VEL * 2)
#     vertical_coll = vertical_collision(player, objects)
#
#     if right_coll or vertical_coll:
#         for obj in vertical_coll:
#             if obj and type(obj) == Floor:
#                 player.landed()
#         return True
#
#     return False
