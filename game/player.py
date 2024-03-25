"""
File: player.py
Description: Holds various properties for the player object
Modified By: Kevin Lizarazu
Date: 03/20/2024
"""
from game.util import *


class Player(pygame.sprite.Sprite):

    def __init__(self, x, y, width, height):
        super().__init__()
        self.rect = pygame.Rect(x, y, width, height)
        self.x_vel = 0
        self.y_vel = 0
        self.animation_count = 0
        self.fall_count = 0
        self.hit = False
        self.mask = None
        self.sprite = None
        self.GRAVITY = 0.8
        self.ANIMATION_DELAY = 5
        self.PLAYER_VEL = 3
        self.SPRITES = load_sprite("birds", "red")

    # Repositions player for replayability
    def reset(self, x, y):
        self.rect.x = x
        self.rect.y = y
        self.sprite = pygame.transform.rotate(self.sprite, 90)
        self.hit = False

    # Action for jumping, resets fall count
    def jump(self):
        self.y_vel = -self.GRAVITY * 8
        self.fall_count = 0

    # Updates values when landing on an object
    def landed(self):
        self.fall_count = 0
        self.y_vel = 0

    # Sets a constant displacement for horizontal movement
    def move_right(self, vel):
        self.x_vel = vel

    # Updates the position of the player
    def move(self, dx, dy):
        self.rect.x += dx
        self.rect.y += dy

    # Hit an object, player nosedives
    def hit_object(self):
        self.hit = True
        self.sprite = pygame.transform.rotate(self.sprite, -90)
        self.x_vel = 0

    # Handles player movement given current x and y velocities
    def loop(self):
        self.move_right(self.PLAYER_VEL)
        self.y_vel += min(1, (self.fall_count / FPS) * self.GRAVITY)

        self.move(self.x_vel, self.y_vel)

        self.fall_count += 1
        self.update_sprite()

    # Animation loop for starting the game
    def start_loop(self):
        self.update_sprite()

    # Updates the player obj_sprite and mask depending on animation count and delay
    def update_sprite(self):
        if self.hit:
            return

        sprite_index = (self.animation_count // self.ANIMATION_DELAY) % len(self.SPRITES)
        self.sprite = self.SPRITES[sprite_index]
        self.mask = pygame.mask.from_surface(self.sprite)
        self.animation_count += 1

    # Draws player on the game window
    def draw(self, window, offset_x):
        window.blit(self.sprite, (self.rect.x - offset_x, self.rect.y))
