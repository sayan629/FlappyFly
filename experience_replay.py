from collections import deque
import random

class ReplayMemory():
    
    # Create FIFO queue - experience replay memory
    def __init__(self, maxlen, seed = None):
        self.memory  = deque([], maxlen = maxlen)