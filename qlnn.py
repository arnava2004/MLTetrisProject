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
    
    def trainModel(self, env, learningRate = 0.001, discountRate = 0.9, criterion = nn.MSELoss(), episodes = 3000, exploreEpisodes = 1000, maxMoves = 300, maxMoveHistory = 1000, displayEvery = 50, batchSize = 128, name = "tetrisbot", priorityExp = False):

        optimizer = torch.optim.Adam(self.parameters(), learningRate)
        moveHistory = deque(maxlen=maxMoveHistory)
        epsilon = 0.99
        highestScore = 0
        episodeData = []

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

            # if (len(moveHistory) > batchSize):
            batch = random.sample(moveHistory, min(len(moveHistory), batchSize))
            if (priorityExp):
                tempDif = [abs((r if done else r + discountRate * self.forward(torch.tensor(ns, dtype=torch.float32)).detach()) - self.forward(torch.tensor(s, dtype=torch.float32)).detach()) for s, ns, r, d in moveHistory]
                batch = random.choices(moveHistory, tempDif, k=min(batchSize, len(moveHistory)))
            x = torch.tensor([s for s, ns, r, d in batch], dtype=torch.float32)
            y = torch.tensor([r if d else r + discountRate * self.forward(torch.tensor(ns, dtype=torch.float32)).detach() for s, ns, r, d in batch], dtype=torch.float32)

            optimizer.zero_grad()
            x = self.forward(x)
            x = torch.squeeze(x)
            loss = criterion(x, y)
            loss.backward()
            # nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()
            episodeData.append((moves, score, float(loss)))
            print(f"Episode {i}: {moves} moves, Loss: {float(loss)}, Score: {score}")

        return episodeData


class DuelingQLNN(QLTetrisNeuralNetwork):
    
    def __init__(self, sharedLayers, valueLayer, advantageLayer, actionSpace):
        super().__init__(sharedLayers)
        self.value = valueLayer
        self.advantage = advantageLayer
        self.actions = actionSpace
    
    def forward(self, x):
        x = super().forward(x)
        val = self.value(x)
        adv = self.advantage(x)
        avgAdv = (1 / len(self.actions)) * sum(adv)
        q = val + (adv - avgAdv)
        return q

    def bestMoveFinder(self, state, actions):
        values = self.forward(torch.tensor(state, dtype=torch.float32)).detach()
        indices = [i for i in range(len(self.actions)) if self.actions[i] in actions]
        index = np.argmax([values[i] for i in range(len(self.actions)) if self.actions[i] in actions])
        index = indices[index]
        return self.actions[index]
    
    def trainModel(self, env, learningRate = 0.001, discountRate = 0.9, criterion = nn.MSELoss(), episodes = 3000, exploreEpisodes = 1000, maxMoves = 300, maxMoveHistory = 1000, displayEvery = 50, batchSize = 128, name = "tetrisbot", priorityExp = False):

        priorityExp = False
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
                bestAction = self.bestMoveFinder(curState, list(nextStates.keys())) if random.random() > epsilon else random.choice(list(nextStates.keys()))
                reward, done = env.play(bestAction[0], bestAction[1], display)
                moveHistory.append((curState, bestAction, nextStates[bestAction], reward, done)) # S, a, S', r, terminate
                curState = nextStates[bestAction]
                moves += 1
                print(f"Number of moves: {moves}", end="\r")

            epsilon -= (0.99) / exploreEpisodes
            score = env.get_game_score()
            if score > highestScore:
                highestScore = score
                torch.save(self, name)

            # if (len(moveHistory) > batchSize):
            # tempDif = [abs((r if done else r + discountRate * self.forward(torch.tensor(ns, dtype=torch.float32)).detach()) - self.forward(torch.tensor(s, dtype=torch.float32)).detach()) for s, ns, r, d in moveHistory]
            batch = random.sample(moveHistory, min(len(moveHistory), batchSize))
            # batch = random.choices(moveHistory, tempDif, k=min(batchSize, len(moveHistory)))
            x = torch.tensor([s for s, a, ns, r, d in batch], dtype=torch.float32)
            a = torch.tensor([self.actions.index(a) for s, a, ns, r, d in batch], dtype=torch.int64)
            y = torch.tensor([r if d else r + discountRate * self.forward(torch.tensor(ns, dtype=torch.float32)).max().detach() for s, a, ns, r, d in batch], dtype=torch.float32)

            optimizer.zero_grad()
            x = self.forward(x)
            x = x.gather(1, a.reshape(-1, 1))
            x = torch.squeeze(x)
            loss = criterion(x, y)
            loss.backward()
            # nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()
            print(f"Episode {i}: {moves} moves, Loss: {float(loss)}, Score: {score}")