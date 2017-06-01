from collections import deque, namedtuple
import numpy as np


class memunit:
    def __init__(self,size):
        self.mem=deque(maxlen=size)

    def addExp(self,memElement):
        self.mem.append(memElement)

    def count(self):
        return len(self.mem)

    def sample(self):
        return self.mem

    def randomSample(self,numberOfSamples):
        delta=0
        a=[]
        for x in xrange(0,numberOfSamples):
            step=np.random.randn(1)
            step=(step*1000)%self.count()
            delta=(delta+step)%self.count()
            a.append(self.mem[delta])
        return  a



