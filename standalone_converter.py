'''
This script requires WaveFunctionCollapse.exe from
https://github.com/mxgmn/WaveFunctionCollapse
'''

from PIL import Image
import numpy as np
import logging
import pickle
import subprocess
from hashlib import blake2b
import xml.etree.ElementTree as ET

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)

def tilemap_to_bitmap(image, size):
    '''Arguments:
    image -- an image of (.PNG, .JPG, .BMP) of tiles.
    size -- tile size, tiles must be a factor of the image size.
    '''
    image = image.convert('RGB')
    w, h = image.size
    arr = np.asarray(image, dtype=np.uint8)
    print(arr.shape)

    tiles = dict()
    encoding = dict()
    print((int(h/size), int(w/size), 3))
    bmp = np.zeros((int(h/size), int(w/size), 3), dtype=np.uint8)

    for row in range(0, int(h/size)):
        for count, corner in enumerate(range(0, int(w/size))):
            tile = image.crop((size*corner, size*row, size*corner+size, size*row+size))

            #Convert to array, and bytestr
            tarr = np.asarray(tile, dtype=np.uint8)
            tstr = tarr.tobytes()

            #colorhash
            r, g, b = blake2b(tstr, digest_size=3).digest()
            bmp[row, corner, 0] = r
            bmp[row, corner, 1] = g
            bmp[row, corner, 2] = b

            tiles[tstr] = None
            encoding[tstr] = (r, g, b)
        else:
            #logging.debug(count+1)
            pass

    bimg = Image.fromarray(bmp)
    bimg.save("samples/PYOUT.png")
    logging.info(f'TILEMAP({bmp.shape})->debug/PYOUT.png')
    
    with open('debug/encoding.pkl', 'wb') as file:
        pickle.dump(encoding, file)
    logging.info(f'ENCODING->debug/encoding.pkl')

    tiles = list(tiles.keys())
    t = int(np.ceil(np.sqrt(len(tiles))))
    base = Image.new('RGB', (t*size,t*size), color=0)

    num = 0
    for r in range(t):
        for c in range(t):
            try: 
                tarr = np.frombuffer(tiles[num], dtype=np.uint8).reshape(size,size,3)
                grid = Image.fromarray(tarr)
                base.paste(grid, box=(c*size, r*size))
            except IndexError as e: 
                logging.info(e)
            num += 1

    for tile in tiles:
        r,g,b = blake2b(tile, digest_size=3).digest()
        #logging.debug((r,g,b))

    base.save(f"debug/tileset.png")
    logging.info(f'TILESET({t})->debug/tileset.png')

    return encoding, base

def bmp_to_tilemap(bmp, enc):
    bmp = bmp.convert('RGB')

    arr = np.asarray(bmp, dtype=np.uint8)
    size = int(np.sqrt(len(tuple(enc.keys())[0])/3)) #yikes :D

    colours = {}
    for tile, colour in enc.items():
        timg = Image.fromarray(np.frombuffer(tile, dtype=np.uint8).reshape(size,size,3))
        assert timg.size == (size,size)

        colours[colour] = timg

    width, height, cs = arr.shape
    base = Image.new('RGB', (height*size, width*size), color=0)

    for r in range(width):
        for c in range(height):
            base.paste(colours[tuple(arr[r, c])], box=(c*size, r*size))

    base.save(f"debug/wfcout.png")
    logging.info(f'WFCTILEMAP({width}-{height})->debug/wfcout.png')

    return base

def run_wfc(N, width, height):
    tree = ET.parse('samples.xml')
    for child in tree.getroot():
        child.attrib['name'] = 'PYOUT'
        child.attrib['N'] = str(N)
        child.attrib['height'] = str(height)
        child.attrib['width'] = str(width)
        logging.info(repr(child.attrib))
    
    tree.write('samples.xml')
    subprocess.run('.\WaveFunctionCollapse.exe',
                    shell = True)

def tilemap_to_wfc(path, size, N=3, wfc_size=(32,32)):
    '''Arguments:
    path -- path to tilemap (.png)
    size -- size of tiles
    N = adj NxN
    wfc_size = output width, height as tuple (width, height)
    '''
    logging.info(f'opening image from {path}')
    image = Image.open(path)
    encoding, bitmap = tilemap_to_bitmap(image, size)

    run_wfc(N, *wfc_size)

    bmp = Image.open('1 PYOUT 0.png')
    return bmp_to_tilemap(bmp, encoding)


if __name__ == "__main__":
    from inspect import cleandoc

    tilemap = tilemap_to_wfc('examples/vania.png', 16, N=2, wfc_size=(128,16))
    tilemap.save('output.png')
    tilemap.show()
    