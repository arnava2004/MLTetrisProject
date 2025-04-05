import sys
import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim.adam

from qlnn import QLTetrisNeuralNetwork
from TetrisEnv import TetrisEnvironment, TetrisBlocks

def parseArguments() -> argparse.Namespace:
    parse: argparse.ArgumentParser = argparse.ArgumentParser()
    parse.add_argument("-train", metavar="save_path", help="-train <save_path> : Train a model and save it to save_path")
    parse.add_argument("-model", metavar="load_path", help="-model <load_path> : Load a model from load_path instead of random initialization")
    parse.add_argument("-loop", action="store_true", help="-loop : Run the model on repeat")
    args: argparse.Namespace = parse.parse_args()
    if len(sys.argv) == 1:
        parse.print_help()
    return args

def trainTetrisModel(model: QLTetrisNeuralNetwork, env, name: str):

    learningRate = 0.01
    discountRate = 0.95
    criterion = nn.MSELoss()
    episodes = 5000
    exploreEpisodes = 4000
    maxMoves = -1
    maxMoveHistory = 2000
    displayEvery = 50
    batchSize = 256

    model.trainModel(env, learningRate, discountRate, criterion, episodes, exploreEpisodes, maxMoves, maxMoveHistory, displayEvery, batchSize, name)

def runTetrisModel(model: QLTetrisNeuralNetwork, env):
    env.reset()
    done = False
    
    while not done:
        nextStates = env.get_next_states()
        bestAction = model.bestMoveFinder(nextStates)
        reward, done = env.play(bestAction[0], bestAction[1], True)

env = TetrisEnvironment()
args = parseArguments()

if args.model == None:
    numStates = env.get_state_size()
    layers = [nn.Linear(numStates, 32), nn.ReLU(), nn.Linear(32,32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1)]
    # layers = [nn.Linear(numStates, 1)]
    model = QLTetrisNeuralNetwork(layers)
else:
    print(f"Loading Model: {args.model}")
    model = torch.load(args.model, weights_only=False)
    for name, param in model.named_parameters():
        print(f"Layer: {name}, Data: {param.data}")

if args.train != None:
    trainTetrisModel(model, env, args.train)

while args.loop:
    runTetrisModel(model, env)