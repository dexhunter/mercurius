from mercurius.strategy.base import expert
from mercurius.strategy import tools
import numpy as np
import logging

class ubah(expert):
    '''Uniform Buy and Hold'''
    def __init__(self):
        super(ubah, self).__init__()


    def get_b(self, data, last_b):
        return last_b

if __name__ == '__main__':
    tools.run(ubah(),plot=True)
