from PIL import Image
from hashlib import blake2b
import numpy as np
import logging

def map2hash(img, size):
    '''Arguments:
    img -- an Image object of (.PNG, .BMP) of tiles.
    size -- tile size, tiles must be a factor of the image size.
    '''
    width, height = img.size
    arr = np.asarray(img, dtype=np.uint8)
    logging.info(f'img-size:{width, height}')
    
    twidth, theight = int(width/size), int(height/size) #TILE_WIDTH, TILE_HEIGHT
    hashmap = np.zeros((twidth, theight), dtype=np.uint64)
    encoding = dict()

    for row in range(twidth):
        for col in range(theight):
            ypos, xpos = size*col, size*row
            tile = img.crop(box=(ypos, xpos, ypos+size, xpos+size))

            tarr = np.array(tile, dtype=np.uint8)
            tstr = tarr.tobytes()
            thash = blake2b(tstr, digest_size=8).digest()
            tbytes = int.from_bytes(thash, 'big')

            encoding[tbytes] = tstr
            hashmap[col, row] = tbytes

    logging.info(f'total-tiles:{twidth*theight}')
    logging.info(f'finished-hashmap:{hashmap.shape}')

    return encoding, hashmap