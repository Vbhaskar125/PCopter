from ple import PLE
import numpy as np
from ple.games import pixelcopter
import tensorflow as tf
import ExperienceMemory
from collections import namedtuple


Experience = namedtuple('Experience',
                                'player_vel player_dist_to_ceil next_gate_block_top next_gate_dist_to_player next_gate_block_bottom player_dist_to_floor player_y action reward lives')


class RandomAgent:
    def __init__(self, actions):
        self.actions = actions

    def pickAction(self):
        return self.actions[np.random.randint(0, len(self.actions))]


class CopterAgent():
    def __init__(self,plenv,game,Qnetwork):
        self.game=game
        self.plenv=plenv
        self.plenv.init()
        self.actions=plenv.getActionSet()
        self.memory=ExperienceMemory.ExperienceMemory(500)
        self.actions = self.plenv.getActionSet()
        self.Qnetwork=Qnetwork

    def start_episode(self):
        self.plenv.reset_game()


    def sampleRandomRun(self,trainEpisodes):
        ra = RandomAgent(self.plenv.getActionSet())
        xqq=0
        for x in xrange(trainEpisodes):
            if self.plenv.game_over():
                xqq+=1
                print(xqq)
                self.Qnetwork.train(self.memory.sessionSample())
                self.start_episode()
            #print(self.memory.countSession())
            nextAction = ra.pickAction()
            reward = self.plenv.act(nextAction)
            ljk=self.game.lives
            delta=self.game.getGameState()
            try:
                self.memory.add(delta, nextAction, reward, ljk)
            except:
                print("exception")
                pass





    def pickRandomAction(self,reward, obs):
        #predaction=self.convnetpred(obs)
        return self.actions[np.random.randint(0, len(self.actions))]

    def trainSession(self,numberOfEpisodes):
        ra=RandomAgent(self.game.getActionSet())
        self.plenv.resetGame()
        self.start_episode()
        for x in xrange(0,numberOfEpisodes):
            if self.plenv.game_over():
                self.plenv.resetGame()
                self.start_episode()
            randomAct=ra.pickAction()
            rewrd=self.plenv.act(randomAct)
            self.memory.add(self.game.getGameState(),randomAct,rewrd,self.game.lives)
        self.Qnetwork.train(self.memory.sessionSample())





    def trainPoolMemory(self):
        pass




    def play(self):
        self.plenv.reset_game()
        self.start_episode()
        while not self.plenv.game_over():
            gameState=self.game.getGameState()
            lives=self.game.lives
            nextAction=self.Qnetwork.predict(gameState,lives)
            reward=self.plenv.act(nextAction)
            self.memory.add(self.game.getGameState(),nextAction,reward,self.game.lives)


    def testFunction(self):
       ppty= self.Qnetwork.prepData(data=self.memory.sessionSample())
       print ppty
