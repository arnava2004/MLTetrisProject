import sys
import argparse
import numpy as np
from matplotlib import pyplot as plt

FILEPATH = "Model Training\\"

def parseArguments() -> argparse.Namespace:
    parse: argparse.ArgumentParser = argparse.ArgumentParser()
    parse.add_argument("file", help="Plot training data from target file")
    parse.add_argument("-type", default="Moves", help="Moves, Scores, or Loss")
    if len(sys.argv) == 1:
        parse.print_help()
    args: argparse.Namespace = parse.parse_args()
    return args

def extractData(file: str):
    data = np.loadtxt(file, delimiter=',')
    moves = data[:, 0]
    scores = data[:, 1]
    losses = data[:, 2]
    return {"Moves": moves, "Score": scores, "Loss": losses}

def setupPlotFigure(xTitle: str, yTitle: str, title: str) -> None:
    plt.figure(figsize=(8,5))
    plt.xlabel(xTitle)
    plt.ylabel(yTitle)
    plt.title(title)
    plt.grid(True)

def plotData(x, y, color: str, name: str):
    plt.plot(x, y, color=color, label=name)

args = parseArguments()
if args.file != None:
    type = args.type
    data = extractData(FILEPATH + args.file)[type]
    step = 100
    x = range(step - 1, len(data), step)
    y = [max(data[i-step+1:i+1]) for i in x]
    setupPlotFigure("Episodes", type, f"Plot of max {type} every {step} episodes")
    plotData(x, y, "cyan", "")
    plt.show()