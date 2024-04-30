"""
Project: AI Flappy Bird
Class: CS5804
Modified By: Kevin Lizarazu
Date: 03/20/2024
"""

import os, sys
import argparse

from game import manualGame, dqnGame, replay

# Load in desired paths
sys.path.append('.')
sys.path.append('./assets')
sys.path.append('./game')


# Will be used to parse arguments and specify types
def parse_args():
    parser = argparse.ArgumentParser(prog="AI Flappy Bird")

    # Add different modes to train agent
    parser.add_argument('--mode', type=str, default=None)

    args = parser.parse_known_args()[0]
    return args


def main():
    print("Welcome to AI Flappy Bird!")

    args = parse_args()

    if args.mode == 'dqn':
        dqnGame.train()
    elif args.mode == 'replay':
        replay.main()
    else:
        manualGame.main()


if __name__ == "__main__":
    main()
