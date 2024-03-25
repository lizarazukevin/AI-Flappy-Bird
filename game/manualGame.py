"""
File: manualGame.py
Description: Default game mode that uses taps to jump, no AI
Modified By: Kevin Lizarazu
Date: 03/20/2024
"""
from game.player import Player
from game.environment import *
from game.object import *

# Initializing game macros and environment window
pygame.init()
pygame.display.set_caption("AI Flappy Bird")
window = pygame.display.set_mode((WIDTH, HEIGHT))


# Populates and runs the game on a loop
def main():
    clock = pygame.time.Clock()
    offset_x = 0

    # floor object created (scaled by 1.5)
    floor_w, floor_h = 504, 168
    floor = [Floor(i * floor_w, HEIGHT - floor_h, floor_w, floor_h)
             for i in range(MAP_WIDTH // floor_w + 1)]

    # pipe objects created (scaled by 1.5)
    pipe_w, pipe_h = 78, 480
    pipes_bottom = [Pipe(START_WIDTH + (i * PIPE_INTERVAL), 400, pipe_w, pipe_h)
                    for i in range((MAP_WIDTH - START_WIDTH) // (pipe_w + PIPE_INTERVAL))]

    pipes_top = [Pipe(START_WIDTH + (i * PIPE_INTERVAL), -300, pipe_w, pipe_h, True)
                 for i in range((MAP_WIDTH - START_WIDTH) // (pipe_w + PIPE_INTERVAL))]

    player = Player(floor_w // 2, (HEIGHT - floor_h) // 2, 51, 36)
    env = Environment()

    mode = "START"
    run = True

    while run:
        clock.tick(FPS)

        # game over condition
        if mode == "END":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    break

                # press r key to replay
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r and player.y_vel == 0:
                    offset_x = 0
                    player.reset(floor_w // 2, (HEIGHT - floor_h) // 2)
                    mode = "START"

            if handle_collision(player, floor):
                player.landed()
            else:
                player.loop()

            draw(window, env, player, floor, offset_x, pipes_bottom, pipes_top)

            continue

        # game continues condition
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break

            # handles event action for player jumping
            if event.type == pygame.KEYDOWN:
                mode = "ALIVE"
                if event.key == pygame.K_SPACE and mode == "ALIVE":
                    player.jump()

        # remaining actions after events have been observed
        if mode == "START":
            player.PLAYER_VEL = 3
            player.start_loop()
        elif handle_collision(player, [*floor, *pipes_bottom, *pipes_top]):
            mode = "END"
            player.PLAYER_VEL = 0
            player.hit_object()
        else:
            player.loop()

        draw(window, env, player, floor, offset_x, pipes_bottom, pipes_top)

        # controls the camera offset, follows player
        if (player.rect.right - offset_x >= WIDTH - floor_w) and \
                player.x_vel > 0 and \
                player.rect.right < MAP_WIDTH - floor_w + 10:
            offset_x += player.x_vel

    pygame.quit()
    quit()


if __name__ == "__main__":
    main()
