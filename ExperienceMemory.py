import MemUnit
import numpy as np
from collections import namedtuple

Experience = namedtuple('Experience',
                                'player_vel player_dist_to_ceil next_gate_block_top next_gate_dist_to_player next_gate_block_bottom player_dist_to_floor player_y action reward lives')


class ExperienceMemory:
    def __init__(self,size):
        Experience = namedtuple('Experience',
                                'player_vel player_dist_to_ceil next_gate_block_top next_gate_dist_to_player next_gate_block_bottom player_dist_to_floor player_y action reward lives')
        self.size=size
        self.sessionMemory=MemUnit.memunit(size)
        #self.poolMemory=MemUnit.memunit(5*size)

    def add(self,gameState,action,score,lives):
        terminal=0
        act=0
        reward=0
        if(lives==0):
            terminal=1
        if(action==119):
            act=1
        if(score>0):
            reward=1
        elif(score==0):
            reward=0;
        elif(score<0):
            reward=-1
        MemElement = Experience(gameState['player_vel'], gameState['player_dist_to_ceil'],
                                gameState['next_gate_block_top'], gameState['next_gate_dist_to_player'],
                                gameState['next_gate_block_bottom'], gameState['player_dist_to_floor'],
                                gameState['player_y'], act, reward, lives)
        self.sessionMemory.addExp(MemElement)
        #self.poolMemory.addExp(MemElement)

    def sessionSample(self):
        return self.sessionMemory.sample()

    #def poolSample(self,size):
        #return self.poolMemory(size)
    def countSession(self):
        return self.sessionMemory.count()

