"""
Project: AI Flappy Bird
Class: CS5804
Modified By: Kevin Lizarazu
Date: 03/20/2024
"""

import os, sys
import argparse

from game import manualGame, replay
from dqn import dqn_train, dqn_run
from q_learning import flappy_qlearning
from sarsa import sarsaRun, sarsaTrain

# Load in desired paths
sys.path.append('.')
sys.path.append('./assets')
sys.path.append('./game')


# Will be used to parse arguments and specify types
def parse_args():
    parser = argparse.ArgumentParser(prog="AI Flappy Bird")

    # Add different modes to train agent
    parser.add_argument('--mode', type=str, default=None)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--run', action='store_true')

    args = parser.parse_known_args()[0]
    return args


def main():
    args = parse_args()

    if args.mode == 'dqn':
        dqn_train.train() if args.train else dqn_run.run()
    # elif args.mode == 'dqn':
    #     dqn_train.train()
    elif args.mode == 'qlearning':
        flappy_qlearning.main(args.train)
    elif args.mode == 'sarsa':
        if args.train:
            sarsa = sarsaTrain.sarsaTrainer()
            sarsa.train()
        else:
            sarsa = sarsaRun.sarsaRunner()
            sarsa.run()
    elif args.mode == 'replay':
        replay.main()
    else:
        manualGame.main()


if __name__ == "__main__":
    main()

