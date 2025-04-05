import random
import numpy as np
import torch
import torch.nn as nn
from collections import deque

class QLTetrisNeuralNetwork(nn.Module):

    def __init__(self, layers: list[nn.Module]):
        super(QLTetrisNeuralNetwork, self).__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def bestMoveFinder(self, actionStates: dict):
        values = [float(self.forward(torch.tensor(s, dtype=torch.float32)).detach()) for a, s in list(actionStates.items())]
        index = np.argmax(values)
        return list(actionStates.items())[index][0]
    
    def trainModel(self, env, learningRate = 0.001, discountRate = 0.9, criterion = nn.MSELoss(), episodes = 3000, exploreEpisodes = 1000, maxMoves = 300, maxMoveHistory = 1000, displayEvery = 50, batchSize = 128, name = "tetrisbot"):

        optimizer = torch.optim.Adam(self.parameters(), learningRate)
        moveHistory = deque(maxlen=maxMoveHistory)
        epsilon = 0.99
        highestScore = 0

        for i in range(episodes):

            curState = env.reset()
            done = False
            moves = 0
            display = (i % displayEvery) == 0

            while not done and moves != maxMoves:
                nextStates = env.get_next_states()
                bestAction = self.bestMoveFinder(nextStates) if random.random() > epsilon else random.choice(list(nextStates.keys()))
                reward, done = env.play(bestAction[0], bestAction[1], display)
                moveHistory.append((curState, nextStates[bestAction], reward, done)) # S, S', r, terminate
                curState = nextStates[bestAction]
                moves += 1
                print(f"Number of moves: {moves}", end="\r")

            epsilon -= (0.99) / exploreEpisodes
            score = env.get_game_score()
            if score > highestScore:
                highestScore = score
                torch.save(self, name)

            # for name, param in self.named_parameters():
            #     print(f"Layer: {name}, Data: {param.data}")

            # if (len(moveHistory) > batchSize):
            batch = random.sample(moveHistory, min(len(moveHistory), batchSize))
            x = torch.tensor([s for s, ns, r, d in batch], dtype=torch.float32)
            y = torch.tensor([r if d else r + discountRate * self.forward(torch.tensor(ns, dtype=torch.float32)).detach() for s, ns, r, d in batch], dtype=torch.float32)

            optimizer.zero_grad()
            x = self.forward(x)
            x = torch.squeeze(x)
            loss = criterion(x, y)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()
            print(f"Episode {i}: {moves} moves, Loss: {float(loss)}, Score: {score}")

