from collections import defaultdict

import numpy as np
import logging
import converter

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)


RIGHT = (1, 0)
UP = (0, 1)
LEFT = (-1, 0)
DOWN = (0, -1)

DIRS = [RIGHT,UP,LEFT,DOWN]

class WaveFunction():
    '''modified from https://github.com/robert/wavefunction-collapse/blob/master/main.py'''
    @staticmethod
    def mk(size, weights):
        coefficients = WaveFunction.init_coefficients(size, weights.keys())
        return WaveFunction(coefficients, weights)

    @staticmethod
    def init_coefficients(size, tiles):
        x, y = size
        return np.array([set(tiles) for _ in range(int(x*y))]).reshape(x, y)

    def __init__(self, coefficients, weights):
        self.coefficients = coefficients
        self.weights = weights

class Model():
    '''modified from https://github.com/robert/wavefunction-collapse/blob/master/main.py'''
    def __init__(self, output_size, weights, compatibilities):
        self.output_size = output_size
        self.compats = compatibilities

        self.wavefunction = WaveFunction.mk(output_size, weights)

def add(a, b):
    return (a[0] + b[0], a[1] + b[1])

def neighbours(arr, pos):
    x, y = pos = np.array(pos)
    width, height = arr.shape

    if x < width-1: yield arr[add(pos, RIGHT)]
    if y < height-1: yield arr[add(pos, UP)]
    if x > 0: yield arr[add(pos, LEFT)]
    if y > 0: yield arr[add(pos, DOWN)]

def parse_array(arr):
    compatibilities = set()
    width, height = arr.shape

    weights = defaultdict(int)
    for x, row in enumerate(arr):
        for y, tile in enumerate(row):
            weights[tile] += 1

            for adj_tile, dir in zip(neighbours(arr, (x,y)), DIRS):
                compatibilities.add((tile, adj_tile, dir))

    logging.debug(f'weights:{weights}')
    logging.debug(f'compatibilitiess_total:{len(compatibilities)}')

    return compatibilities, weights

if __name__ == "__main__":
    from PIL import Image

    img = Image.open('examples/cave.png')
    enc, hmap = converter.map2hash(img, 16)
    compats, weights = parse_array(hmap)

    wsize = (32, 32)
    wfc = Model(wsize, weights, compats)
    print(wfc.wavefunction)