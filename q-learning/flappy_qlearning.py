from itertools import cycle
from collections import deque
import copy
import random
import sys
import pygame
from pygame.locals import *
from config import config

if config['train']:
    from q_learning_train import QLearning
else:
    from q_learning_run import QLearning

# Initialize the QLearning agent from configuration
Agent = QLearning()

# Print current mode based on agent's training mode
print("Training agent..." if Agent.training_mode else "Running agent...")

#  Constants for the game
FPS = 30
SCREENWIDTH = 288
SCREENHEIGHT = 512
PIPEGAPSIZE = 100
BASEY = SCREENHEIGHT * 0.79 # Base Y position for the ground

# Load images and hitmasks
IMAGES, HITMASKS = {}, {}
STATE_HISTORY = deque(maxlen=70) # Deque to store state history, limited to 70 items
REPLAY_BUFFER = [] # Buffer to store replay events

# Lists of resources for players and pipes
PLAYERS_LIST = (
    (
        'images/redbird-upflap.png',
        'images/redbird-midflap.png',
        'images/redbird-downflap.png',
    ),
)

BACKGROUNDS_LIST = (
    'images/background-day.png',
    'images/background-night.png',
)

PIPES_LIST = (
    'images/pipe-green.png',
    'images/pipe-red.png',
)


xrange = range

def showWelcomeAnimation():
    """Show initial welcome animation."""
    playery = int((SCREENHEIGHT - IMAGES['player'][0].get_height()) / 2)
    playerIndexGen = cycle([0, 1, 2, 1])
    return {
        'playery': playery,
        'basex': 0,
        'playerIndexGen': playerIndexGen,
    }


def mainGame(movementInfo):
    """Handle the main game loop."""
    score = playerIndex = loopIter = 0
    playerIndexGen = movementInfo['playerIndexGen']
    playerx, playery = int(SCREENWIDTH * 0.2), movementInfo['playery']

    basex = movementInfo['basex']
    baseShift = IMAGES['base'].get_width() - IMAGES['background'].get_width()

    # Generate new pipes at the start
    newPipe1 = getRandomPipe()
    newPipe2 = getRandomPipe()
    upperPipes = [
        {'x': SCREENWIDTH + 200, 'y': newPipe1[0]['y']},
        {'x': SCREENWIDTH + 200 + (SCREENWIDTH / 2), 'y': newPipe2[0]['y']},
    ]
    lowerPipes = [
        {'x': SCREENWIDTH + 200, 'y': newPipe1[1]['y']},
        {'x': SCREENWIDTH + 200 + (SCREENWIDTH / 2), 'y': newPipe2[1]['y']},
    ]

    # Game dynamics initialization
    pipeVelX = -4
    playerVelY = -9
    playerMaxVelY = 10
    playerAccY = 1
    playerFlapAcc = -9
    playerFlapped = False


    if len(STATE_HISTORY) < 20:
        STATE_HISTORY.clear()
    resume_from_history = len(STATE_HISTORY) > 0 if Agent.training_mode else None
    initial_len_history = len(STATE_HISTORY)
    resume_from = 0
    current_score = STATE_HISTORY[-1][5] if resume_from_history else None
    print_score = False

    # Game loop
    while True:
        # If resuming from previously saved game state
        if resume_from_history:
            # Check if there are more states to resume from
            if resume_from < initial_len_history:
                # On the first iteration, set all game variables to their saved states
                if resume_from == 0:
                    playerx, playery, playerVelY, lowerPipes, upperPipes, score, playerIndex = \
                        STATE_HISTORY[resume_from]
                else:
                    # For subsequent iterations, only update the pipe positions
                    lowerPipes, upperPipes = STATE_HISTORY[resume_from][3], STATE_HISTORY[resume_from][4]
                resume_from += 1
        else:
            # Save the current state if the score has reached a pre-defined threshold (configurable)
            if Agent.training_mode and config['resume_score'] and score >= config['resume_score']:
                    STATE_HISTORY.append([playerx, playery, playerVelY, copy.deepcopy(lowerPipes),
                                          copy.deepcopy(upperPipes), score, playerIndex])

        # Event handling for user inputs and system events
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                # Quit the game and save the current Q-values and training data
                if print_score:
                    print('')
                if Agent.training_mode:
                    Agent.save_q_values()
                    Agent.save_training_data()
                pygame.quit()
                sys.exit()
            # Player input: space bar or up arrow key causes the bird to flap
            if event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
                if playery > -2 * IMAGES['player'][0].get_height():
                    playerVelY = playerFlapAcc
                    playerFlapped = True
        # Agent's action based on current state
        if Agent.choose_action(playerx, playery, playerVelY, lowerPipes):
            if playery > -2 * IMAGES['player'][0].get_height():
                playerVelY = playerFlapAcc
                playerFlapped = True
        # Check for collision
        crashTest = checkCrash({'x': playerx, 'y': playery, 'index': playerIndex},
                               upperPipes, lowerPipes)
        if crashTest[0]:
            # If a crash occurs, handle the end-of-game logic
            if print_score:
                print('')
            if resume_from_history:
                if score > current_score:
                    # If the current score is greater than the score at the time of resuming, update Q-values
                    Agent.update_q_values(score)
                else:
                    # Otherwise, store the move history for potential replay
                    REPLAY_BUFFER.append(copy.deepcopy(Agent.moves))
                # If there are enough replays stored or the score is high, retrain from these replays
                if score > current_score or len(REPLAY_BUFFER) >= 50:
                    random.shuffle(REPLAY_BUFFER)
                    for _ in range(5):
                        if REPLAY_BUFFER:
                            Agent.moves = REPLAY_BUFFER.pop()
                            Agent.update_q_values(current_score)
                    # Clear history and buffer after retraining
                    STATE_HISTORY.clear()
                    REPLAY_BUFFER.clear()
            else:
                # Update Q-values
                Agent.update_q_values(score)
            return {
                'y': playery,
                'groundCrash': crashTest[1],
                'basex': basex,
                'upperPipes': upperPipes,
                'lowerPipes': lowerPipes,
                'score': score,
                'playerVelY': playerVelY
            }

        # Score calculation based on the bird passing through the pipes
        playerMidPos = playerx + IMAGES['player'][0].get_width() / 2
        for pipe in upperPipes:
            pipeMidPos = pipe['x'] + IMAGES['pipe'][0].get_width() / 2
            if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                score += 1
                if score % config['print_score'] == 0:
                    print_score = True
                    print(f"\r {'Training' if Agent.training_mode else 'Running'} agent, "
                          f"score reached: {score:,}", end="")

                if config['max_score'] and score >= config['max_score']:
                    # End the game if max score is reached
                    if print_score:
                        print('')
                    Agent.end_episode(score)
                    STATE_HISTORY.clear()
                    REPLAY_BUFFER.clear()
                    print(f"Max score of {config['max_score']} reached at episode {Agent.episode_count}...")
                    return {
                        'y': playery,
                        'groundCrash': crashTest[1],
                        'basex': basex,
                        'upperPipes': upperPipes,
                        'lowerPipes': lowerPipes,
                        'score': score,
                        'playerVelY': playerVelY,
                    }

        # Update player index for animation every 3 iterations
        if (loopIter + 1) % 3 == 0:
            playerIndex = next(playerIndexGen)
        loopIter = (loopIter + 1) % 30 # Increment loop iterator and reset every 30 iterations

        # Move the base to create a continuous scroll effect
        basex = -((-basex + 100) % baseShift)

        # Gravity effect: increase downward velocity unless a flap has occurred
        if playerVelY < playerMaxVelY and not playerFlapped:
            playerVelY += playerAccY
        if playerFlapped:
            playerFlapped = False  # Reset flapping after processing

        # Update the player's vertical position
        playerHeight = IMAGES['player'][playerIndex].get_height()
        playery += min(playerVelY, BASEY - playery - playerHeight)

        # Continue moving pipes leftward if resuming from saved state has finished
        if resume_from >= initial_len_history:
            for uPipe, lPipe in zip(upperPipes, lowerPipes):
                uPipe['x'] += pipeVelX
                lPipe['x'] += pipeVelX

        # Generate new pipes when the first pipe is about to exit the screen on the left
        if 0 < upperPipes[0]['x'] < 5:
            newPipe = getRandomPipe()
            upperPipes.append(newPipe[0])  # Append a new upper pipe
            lowerPipes.append(newPipe[1])  # Append a new lower pipe

        # Remove pipes that have moved off the screen to maintain performance
        if upperPipes[0]['x'] < -IMAGES['pipe'][0].get_width():
            upperPipes.pop(0)  # Remove the first upper pipe
            lowerPipes.pop(0)  # Remove the first lower pipe

        # Rendering the game state
        if config['show_game']:
            SCREEN.blit(IMAGES['background'], (0, 0)) # Draw the background

            # Draw all pipes
            for uPipe, lPipe in zip(upperPipes, lowerPipes):
                SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
                SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

            SCREEN.blit(IMAGES['base'], (basex, BASEY)) # Draw the moving base
            showScore(score)   # Show the current score

            # Draw the player at the current position
            playerSurface = IMAGES['player'][playerIndex]
            SCREEN.blit(playerSurface, (playerx, playery))

            pygame.display.update() # Update the display
            FPSCLOCK.tick(FPS)  # Maintain the game's frames per second


def showGameOverScreen(crashInfo):
    # Extract the player's y-coordinate at the time of the crash from the crash info.
    playery = crashInfo['y']
    playerHeight = IMAGES['player'][0].get_height()

    # Infinite loop to keep the game in the game over state until an action is taken.
    while True:
        for event in pygame.event.get():
            # If the window close button is clicked or the escape key is pressed,
            # save the Q-values, training data, and exit the game
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                if Agent.training_mode:
                    Agent.save_q_values()
                    Agent.save_training_data()
                pygame.quit()
                sys.exit()
            # Allow restarting only if the player sprite is at or below the ground level.
            # This check ensures that the game restarts only from a valid game over state.
            if event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
                if playery + playerHeight >= BASEY - 1:
                    return
        return


def getRandomPipe():
    # Determine the vertical position for the new pipe's gap
    gapY = random.randrange(0, int(BASEY * 0.6 - PIPEGAPSIZE))
    gapY += int(BASEY * 0.2)
    pipeHeight = IMAGES['pipe'][0].get_height()
    pipeX = SCREENWIDTH + 10 # Position the pipe just outside the right side of the screen

    return [
        {'x': pipeX, 'y': gapY - pipeHeight},
        {'x': pipeX, 'y': gapY + PIPEGAPSIZE},
    ]


def showScore(score):
    scoreDigits = [int(x) for x in list(str(score))] # Split the score into individual digits
    totalWidth = 0 # Total width of all digits

    for digit in scoreDigits:
        totalWidth += IMAGES['numbers'][digit].get_width()

    Xoffset = (SCREENWIDTH - totalWidth) / 2

    for digit in scoreDigits:
        SCREEN.blit(IMAGES['numbers'][digit], (Xoffset, SCREENHEIGHT * 0.1)) # Draw each digit
        Xoffset += IMAGES['numbers'][digit].get_width() # Move the offset for the next digit


def checkCrash(player, upperPipes, lowerPipes):
    pi = player['index']
    playerRect = pygame.Rect(player['x'], player['y'], IMAGES['player'][pi].get_width(),
                             IMAGES['player'][pi].get_height())

    # Check collision with the ground
    if playerRect.bottom >= BASEY:
        return [True, True]

    # Check for collision with each pair of pipes
    for uPipe, lPipe in zip(upperPipes, lowerPipes):
        # Define hitboxes for upper and lower pipes
        uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], IMAGES['pipe'][0].get_width(), IMAGES['pipe'][0].get_height())
        lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], IMAGES['pipe'][1].get_width(), IMAGES['pipe'][1].get_height())
        # Perform pixel perfect collision detection
        if pixelCollision(playerRect, uPipeRect, HITMASKS['player'][pi], HITMASKS['pipe'][0]) or \
                pixelCollision(playerRect, lPipeRect, HITMASKS['player'][pi], HITMASKS['pipe'][1]):
            return [True, False]

    return [False, False]  # No collision detected


def pixelCollision(rect1, rect2, hitmask1, hitmask2):
    """
    Checks for pixel-level collisions between two rectangles using hitmasks.
    """
    rect = rect1.clip(rect2)
    if not rect.width or rect.height:
        return False

    # Check every pixel within the overlap area for a collision
    for x in xrange(rect.width):
        for y in xrange(rect.height):
            if hitmask1[rect.x - rect1.x + x][rect.y - rect1.y + y] and hitmask2[rect.x - rect2.x + x][rect.y - rect2.y + y]:
                return True
    return False


def getHitmask(image):
    """
    Creates a hitmask for an image.
    """
    mask = []
    for x in xrange(image.get_width()):
        mask.append([])
        for y in xrange(image.get_height()):
            # Boolean value for whether pixel is opaque
            mask[x].append(bool(image.get_at((x, y))[3]))
    return mask

def main():
    global SCREEN, FPSCLOCK
    pygame.init()
    FPSCLOCK = pygame.time.Clock()
    SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
    pygame.display.set_caption('Flappy Bird')

    # Load static resources
    IMAGES['numbers'] = (
        pygame.image.load('images/0.png').convert_alpha(),
        pygame.image.load('images/1.png').convert_alpha(),
        pygame.image.load('images/2.png').convert_alpha(),
        pygame.image.load('images/3.png').convert_alpha(),
        pygame.image.load('images/4.png').convert_alpha(),
        pygame.image.load('images/5.png').convert_alpha(),
        pygame.image.load('images/6.png').convert_alpha(),
        pygame.image.load('images/7.png').convert_alpha(),
        pygame.image.load('images/8.png').convert_alpha(),
        pygame.image.load('images/9.png').convert_alpha()
    )

    IMAGES['gameover'] = pygame.image.load('images/gameover.png').convert_alpha()
    IMAGES['message'] = pygame.image.load('images/message.png').convert_alpha()
    IMAGES['base'] = pygame.image.load('images/base.png').convert_alpha()

    # Main game loop
    while True:
        # Randomly select the background and player images
        randBg = random.randint(0, len(BACKGROUNDS_LIST) - 1)
        IMAGES['background'] = pygame.image.load(BACKGROUNDS_LIST[randBg]).convert()
        randPlayer = random.randint(0, len(PLAYERS_LIST) - 1)
        IMAGES['player'] = (
            pygame.image.load(PLAYERS_LIST[randPlayer][0]).convert_alpha(),
            pygame.image.load(PLAYERS_LIST[randPlayer][1]).convert_alpha(),
            pygame.image.load(PLAYERS_LIST[randPlayer][2]).convert_alpha(),
        )
        # Randomly select the pipe images
        pipeindex = random.randint(0, len(PIPES_LIST) - 1)
        IMAGES['pipe'] = (
            pygame.transform.rotate(pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(), 180),
            pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(),
        )
        # Create hitmasks for pipes and player
        HITMASKS['pipe'] = tuple(getHitmask(IMAGES['pipe'][i]) for i in range(2))
        HITMASKS['player'] = tuple(getHitmask(IMAGES['player'][i]) for i in range(3))

        # Start game and handle game over
        movementInfo = showWelcomeAnimation()
        crashInfo = mainGame(movementInfo)
        showGameOverScreen(crashInfo)


if __name__ == '__main__':
    main()
