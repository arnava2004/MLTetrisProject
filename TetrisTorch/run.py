import sys
import argparse
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim.adam

from qlnn import QLTetrisNeuralNetwork
from TetrisEnv import TetrisEnvironment, TetrisBlocks

STATPATH = "Model Stats\\"
MODELPATH = "Models\\"

def parseArguments() -> argparse.Namespace:

    parse: argparse.ArgumentParser = argparse.ArgumentParser()
    parse.add_argument("-train", metavar="save_path", help="-train <save_path> : Train a model and save it to save_path")
    parse.add_argument("-model", metavar="load_path", help="-model <load_path> : Load a model from load_path instead of random initialization")
    parse.add_argument("-stats", metavar="num_samples", help="-stats <num_samples> : Run the model and record sample statistics")
    parse.add_argument("-loop", action="store_true", help="-loop : Run the model on repeat")
    args: argparse.Namespace = parse.parse_args()

    if len(sys.argv) == 1:
        parse.print_help()

    return args

def loadModel(file: str | None) -> QLTetrisNeuralNetwork:

    if file == None:
        numStates = env.get_state_size()
        # layers = [nn.Linear(numStates, 32), nn.ReLU(), nn.Linear(32,32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1)]
        layers = [nn.Linear(numStates, 1)]
        model = QLTetrisNeuralNetwork(layers)
        return model
    else:
        print(f"Loading Model: {file}")
        model = torch.load(MODELPATH + file, weights_only=False)
        for name, param in model.named_parameters():
            print(f"Layer: {name}, Data: {param.data}")
        return model


def trainTetrisModel(model: QLTetrisNeuralNetwork, env, name: str):

    learningRate = 0.01
    discountRate = 0.95
    criterion = nn.MSELoss()
    episodes = 5000
    exploreEpisodes = 4000
    maxMoves = -1
    maxMoveHistory = 2000
    displayEvery = 500
    batchSize = 256

    model.trainModel(env, learningRate, discountRate, criterion, episodes, exploreEpisodes, maxMoves, maxMoveHistory, displayEvery, batchSize, MODELPATH + name)

def runTetrisModel(model: QLTetrisNeuralNetwork, env, display) -> tuple[int, int, float]:

    env.reset()
    done = False
    moves = 0
    startTime = time.time()
    
    while not done:
        nextStates = env.get_next_states()
        bestAction = model.bestMoveFinder(nextStates)
        reward, done = env.play(bestAction[0], bestAction[1], display)
        moves += 1

    deltaTime = time.time() - startTime
    score = env.get_game_score()
    return score, moves, deltaTime

def testModel(model: QLTetrisNeuralNetwork, env, name: str, numSamples: int):

    statFile = open(f"{STATPATH}{name}.stats", mode='w')
    samples = []

    for i in range(1, numSamples + 1):
        print(f"Sample: {i}/{numSamples}", end='\r')
        score, moves, deltaTime = runTetrisModel(model, env, False)
        samples.append((score, moves, deltaTime))

    statFile.write(f"Model: {name}\n")
    for layer, param in model.named_parameters():
        statFile.write(f"{layer}: {param.data}\n")
    
    scores = [x[0] for x in samples]
    moves = [x[1] for x in samples]
    times = [x[2] for x in samples]
    meanScore = sum(scores) / numSamples

    statFile.write("\nSample Stats:\n\n")

    statFile.write(f"   Mean Score: {meanScore: .1f}\n")
    statFile.write(f"   Std. Score: {(sum([(x - meanScore) ** 2 for x in scores]) / (numSamples - 1)) ** 0.5: .1f}\n")
    statFile.write(f"   Min Score: {min(scores)}\n")
    statFile.write(f"   Max Score: {max(scores)}\n\n")

    statFile.write(f"   Average Moves: {sum(moves) / numSamples: .1f}\n")
    statFile.write(f"   Average Time: {sum(times) / numSamples: .3f}s\n")

    statFile.write(f"{'_' * 50}\n\n")
    sampleNum = 1
    for score, moves, deltaTime in samples:
        statFile.write(f"Game {sampleNum}   Score: {score}, Moves: {moves}, Time: {deltaTime}\n")
        sampleNum += 1


env = TetrisEnvironment()
blockdrop = TetrisEnvironment([TetrisBlocks.SINGLEBLOCK], width=4,height=4)
args = parseArguments()
model = loadModel(args.model)

if args.train != None:
    trainTetrisModel(model, env, args.train)

if args.stats != None:
    samples = int(args.stats)
    testModel(model, env, args.model, samples)

while args.loop:
    runTetrisModel(model, env, True)