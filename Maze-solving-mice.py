import numpy as np

PossibleActions = {'U': (-1,0), 'D': (1,0), 'L': (0,-1), 'R': (0,1)}

#First let's first document how the mouse should be able to moveself.
#Let's say that at any point, it shouls be able to move up, down, left or rightself.
#Now defining them according to the direction

class Maze(object):
    def __init__(self):
        self.maze = np.zeros((6,6)) # 6x6 maze - exit at 5,5
        self.maze[5, :5] = 1
        self.maze[:4, 5] = 1
        self.maze[2, 2:] = 1
        self.maze[3,2] = 1
        self.maze[0,0] = 2
        self.robotPosition = (0,0)
        self.steps = 0
        self.constructAllowedStates()

    def printMaze(self):
        print('@@------------------------------------@@')
        for row in self.maze:
            for col in row:
                if col == 0:
                    print('', end='\t')
                elif col == 1:
                    print('X', end='\t')
                elif col == 2:
                    print('R', end='\t')
            print('\n')
        print('@@------------------------------------@@')

    def isAllowedMove(self, state, action):
        y, x = state
        y += PossibleActions[action][0]
        x += PossibleActions[action][1]
        if y < 0 or x < 0 or y > 5 or x > 5:
            return False

        if self.maze[y,x] == 0 or self.maze[y,x] == 2:
            return True
        else:
            return False

    def constructAllowedStates(self):
        allowedStates= {}
        for y, row in enumerate(self.maze):
            for x, col in enumerate(row):
                if self.maze[(y,x)] != 1:
                    allowedStates[(y,x)] = []
                    for action in PossibleActions:
                        if self.isAllowedMove((y,x), action):
                            allowedStates[(y,x)].append(action)
        self.allowedStates = allowedStates

    def updateMaze(self, action):
        y,x = self.robotPosition
        self.maze[y,x] = 0
        y += PossibleActions[action][0]
        x += PossibleActions[action][1]
        self.robotPosition = (y,x)
        self.maze[y,x] = 2
        self.steps += 1

    def isGameOver(self):
        if self.robotPosition == (5,5):
            return True
        else:
            return False

    def getStateAndReward(self):
        reward = self.giveReward()
        return self.robotPosition, reward

    def giveReward(self):
        if self.robotPosition == (5,5):
            return 0
        else:
            return -1

import numpy as np



class Agent(object):
    def __init__(self, maze, alpha=0.15, randomFactor=0.2):
        self.stateHistory = [((0,0), 0)]
        self.G = {}  # present value of expected future rewards
        self.randomFactor = randomFactor
        self.alpha = alpha
        self.initReward(maze.allowedStates)

    def chooseAction(self, state, allowedMoves):
        maxG = -10e15
        nextMove = None
        randomN = np.random.random()
        if randomN < self.randomFactor:
            nextMove = np.random.choice(allowedMoves)
        else:
            for action in allowedMoves:
                newState = tuple([sum(x) for x in zip(state, PossibleActions[action])])
                if self.G[newState] >= maxG:
                    maxG = self.G[newState]
                    nextMove = action
        return nextMove

    def printG(self):
        for i in range(6):
            for j in range(6):
                if (i,j) in self.G.keys():
                    print('%.6f' % self.G[(i,j)], end='\t')
                else:
                    print('X', end='\t\t')
            print('\n')

    def updateStateHistory(self, state, reward):
        self.stateHistory.append((state, reward))

    def initReward(self, allowedStates):
        for state in allowedStates:
            self.G[state] = np.random.uniform(low=-1.0, high=-0.1)

    def learn(self):
        target = 0 # we only learn when we beat the maze

        for prev, reward in reversed(self.stateHistory):
            self.G[prev] = self.G[prev] + self.alpha * (target - self.G[prev])
            target += reward

        self.stateHistory = []
        self.randomFactor -= 10e-5


import matplotlib.pyplot as plt

if __name__ == '__main__':
    maze = Maze()
    robot = Agent(maze, alpha=0.1, randomFactor=0.25)
    moveHistory = []
    for i in range(5000):
        if i % 1000 == 0:
            print(i)
        while not maze.isGameOver():
            state, _ = maze.getStateAndReward()
            action = robot.chooseAction(state, maze.allowedStates[state])
            Maze.printMaze(maze)
            maze.updateMaze(action)
            state, reward = maze.getStateAndReward()
            robot.updateStateHistory(state, reward)
            if maze.steps > 1000:
                maze.robotPosition = (5,5)
        robot.learn()
        moveHistory.append(maze.steps)
        maze = Maze()

    maze = Maze()
    robot = Agent(maze, alpha=0.99, randomFactor=0.25)
    moveHistory2 = []
    for i in range(5000):
        if i % 1000 == 0:
            print(i)
        while not maze.isGameOver():
            state, _ = maze.getStateAndReward()
            action = robot.chooseAction(state, maze.allowedStates[state])
            maze.updateMaze(action)
            state, reward = maze.getStateAndReward()
            robot.updateStateHistory(state, reward)
            if maze.steps > 1000:
                maze.robotPosition = (5,5)
        robot.learn()
        moveHistory2.append(maze.steps)
        maze = Maze()

    plt.subplot(211)
    plt.semilogy(moveHistory, 'b--')
    plt.legend(['alpha=0.1'])
    plt.subplot(212)
    plt.semilogy(moveHistory2, 'r--')
    plt.legend(['alpha=0.99'])
    plt.show()
