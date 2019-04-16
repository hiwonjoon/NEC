import os
import numpy as np

class ReplayBuffer(object):
    def __init__(self,maxlen):
        self.storage = []

        self.counter = 0
        self.maxlen = maxlen

    def append(self,s,a,R):
        ticket, self.counter = self.counter, self.counter + 1
        if ticket < self.maxlen:
            self.storage.append((s,a,R))
        else:
            self.storage[ticket%self.maxlen] = (s,a,R)

    def sample(self,size):
        idxes = np.random.randint(self.counter,size=size) % self.maxlen

        b_s,b_a,b_R = zip(*[self.storage[i] for i in idxes])
        b_s,b_a,b_R = np.array(b_s), np.array(b_a), np.array(b_R)

        return b_s,b_a,b_R
