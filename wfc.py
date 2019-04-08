from collections import defaultdict, deque, namedtuple

import numpy as np
import random
import logging
import converter
import sys

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)


RIGHT = (1, 0)
UP = (0, 1)
LEFT = (-1, 0)
DOWN = (0, -1)

DIRS = [RIGHT,UP,LEFT,DOWN]

class WaveFunction():
    '''modified from 
    https://github.com/robert/wavefunction-collapse/blob/master/main.py'''
    @staticmethod
    def mk(size, weights):
        coefficients = WaveFunction.init_coefficients(size, weights.keys())
        return WaveFunction(coefficients, weights)

    @staticmethod
    def init_coefficients(size, tiles):
        x, y = size
        return np.array([set(tiles) for _ in range(int(x*y))], dtype=set).reshape(x, y)

    def __init__(self, coefficients, weights):
        self.coeffs = coefficients
        self.weights = weights

    def q_collapsed(self):
        for coeff in self.coeffs.flat:
            if len(coeff) > 1:
                return False

        return True

    def collapse(self, pos):
        possibilities = tuple(self.coeffs[pos])
        weights = tuple(self.weights[tile] for tile in possibilities)

        choice = random.choices(possibilities, weights=weights)
        logging.debug(f'{pos}:{choice}')
        self.coeffs[pos] = choice

    def shannon_entropy(self, tile, cache=None):
        if cache.tile == tile:
            return cache.entropy

        sum_weights = 0
        sum_log_weights = 0
        for possibility in tile:
            weight = self.weights[possibility]
            sum_weights += weight
            sum_log_weights += weight * np.log(weight)

        return np.log(sum_weights) - (sum_log_weights / sum_weights)

Cache = namedtuple('Cache', 'entropy tile')

class Model():
    '''modified from 
    https://github.com/robert/wavefunction-collapse/blob/master/main.py'''

    def __init__(self, size, weights, compatibilities):
        self.size = size
        self.compats = compatibilities

        self.wf = WaveFunction.mk(size, weights)

    def run(self):
        i = 0
        while not self.wf.q_collapsed():
            self.iterate()
            i += 1
            logging.debug(f'iter:{i}')

        return self.wf.coeffs

    def iterate(self):
        entropy, pos = self.minimum_entropy()
        logging.debug(f'chosen-tile:{pos, entropy}')
        self.wf.collapse(pos)
        self.propagate(pos)

    def propagate(self, pos):
        stack = deque([pos])

        while len(stack) > 0:
            c_pos = stack.pop()
            c_tiles = self.wf.coeffs[c_pos]

            removal = set()
            for other, dir in neighbours(self.wf.coeffs, c_pos):
                for o_tile in other:
                    if not any([(c_tile, o_tile, dir) in self.compats for c_tile in c_tiles]):
                        o_pos = add(c_pos, dir)
                        removal.add((o_tile, o_pos))
                        stack.append(o_pos)

            for o_tile, o_pos in removal:
                self.wf.coeffs[o_pos].remove(o_tile)

    def minimum_entropy(self):
        cache = Cache(None, None)
        minimum = None
        for x, row in enumerate(self.wf.coeffs):
            for y, tile in enumerate(row):
                if len(self.wf.coeffs[(x,y)]) == 1:
                    continue
                entropy = self.wf.shannon_entropy(tile, cache) - np.random.random() / 100
                cache = Cache(entropy, tile)
                
                if minimum is None or entropy < minimum:
                    minimum = entropy
                    m_tile = (x, y)

        return (minimum, m_tile)


def add(a, b):
    return (a[0] + b[0], a[1] + b[1])

def neighbours(arr, pos):
    x, y = pos = np.array(pos)
    width, height = arr.shape

    if x < width-1: yield (arr[add(pos, RIGHT)], RIGHT)
    if y < height-1: yield (arr[add(pos, UP)], UP)
    if x > 0: yield (arr[add(pos, LEFT)], LEFT)
    if y > 0: yield (arr[add(pos, DOWN)], DOWN)

def parse_array(arr):
    compatibilities = set()
    width, height = arr.shape

    weights = defaultdict(int)
    for x, row in enumerate(arr):
        for y, tile in enumerate(row):
            weights[tile] += 1

            for adj_tile, dir in neighbours(arr, (x,y)):
                compatibilities.add((tile, adj_tile, dir))

    logging.debug(f'weights:{weights}')
    logging.debug(f'compatibilities_total:{len(compatibilities)}')
    logging.debug(f'compatibilities_memory:{sys.getsizeof(compatibilities)}')

    return compatibilities, weights

if __name__ == "__main__":
    from PIL import Image

    img = Image.open('examples/loz.png')
    enc, hmap = converter.map2hash(img, 16)
    compats, weights = parse_array(hmap)

    wsize = (16, 16)
    wfc = Model(wsize, weights, compats)
    output = wfc.run()

    image = converter.hash2map(output, enc, 16)
