import argparse
from io import BytesIO
from PIL import Image
import numpy as np
import os
import subprocess
import struct

def save_image():
    """ Save an image from a static raw as bmp.
    """

    size = 32
    if size == 32:
        with open("/home/richard/research/p3/digits/mnist-data/test32-10.raw", 'rb') as f:
            data = f.read()
    else:
        with open("/home/richard/research/p3/digits/mnist-data/t10k-images.idx3-ubyte", 'rb') as f:
            data = f.read()

    it = data.__iter__()

    for i in range(2*4):
        _ = bytes([it.__next__()])
    for i in range(2*4):
        _ = it.__next__()    

    data = read_bytes(it, size)

    img = Image.new('L', (size,size), color=255)
    pixels = img.load()

    for i in range(size):
        for j in range(size):
            pixels[j, i] = 255-data[i][j]

    img.save("public.bmp")
    

def raw_to_raw(data, dim=28):
    """ 
    Convert a raw image to 32x32, run through P3, and return both images.
    """

    orig_img = Image.new('L', (32,32), color=255)
    orig_pixels = orig_img.load()
    
    # Create raw MNIST image
    for i in range(dim):
        for j in range(dim):
            orig_pixels[j+2,i+2] = 255-data[i][j]

    for row in data:
        print(row)

    # normal
    if 1:
        ret = []
        for i in range(orig_img.size[0]):
            for j in range(orig_img.size[1]):
                ret.append(orig_pixels[i,j])
        return ret

    # pixel
    elif 0:
        size = 2
        
        #np.set_printoptions(edgeitems=16)
        #np.set_printoptions(linewidth=200)

        pixels = np.array(orig_img)
        #pixels = np.arange(32*32)
        #pixels = pixels.reshape((32,32))
        #print(pixels)

        for x in range(int(32/size)):
            for y in range(int(32/size)):
                ystart = y*size
                xstart = x*size
                
                average = np.sum(pixels[ystart:ystart+size,xstart:xstart+size])
                average = average*1/(size**2)
                average = int(average)
                
                pixels[ystart:ystart+size,xstart:xstart+size] = average

        pixels = pixels.reshape((32*32))
        ret = pixels.tolist()
        return ret

    # p3
    else:
        p = subprocess.Popen(
            '../jpeg-9a/cjpeg -grayscale -dct int -q 100 '.split(), 
            stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    
        # Send raw MNIST image to P3 process
        orig_img.save(p.stdin,'BMP')
        orig_img.save("test.jpg")
        return

        # Read P3 image
        #p3_img = Image.open(p.stdout)
        #p3_pixels = p3_img.load()
        
        raw = p.communicate()[0]
        file_jpgdata = BytesIO(raw)
        img = Image.open(file_jpgdata)
        ret = list(img.getdata())                
        return ret


def read_bytes(it, dim=28):
    """ 
    Read and return the next (dim x dim) image.
    """

    out = []
    for i in range(dim):
        inner = []
        for j in range(dim):
            inner.append(it.__next__())
        out.append(inner)
    return out


def main():
    """ 
    Read either a (training/test) file and write output
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="infile")
    parser.add_argument("outfile", help="outfile")
    args = parser.parse_args()

    if not args.dataset:
        print("Missing -dataset")
        return
    elif not args.outfile:
        print("Missing -outfile")
        return

    # Open dataset
    with open(args.dataset, 'rb') as f:
        data = f.read()
    one_percent = (len(data)-8-8)/(28*28)/100

    it = data.__iter__()

    out_bytes = b''    

    # Set up the start of the output file (Magic numbers)
    for i in range(2*4):
        out_bytes += bytes([it.__next__()])
        
    # Skip the bits specifying 28x28...
    for i in range(2*4):
        _ = it.__next__()
    
    # Cause we use 32x32 now, bitches!
    out_bytes += bytes([0,0,0,32])
    out_bytes += bytes([0,0,0,32])

    '''
    image = read_bytes(it)
    out_bytes += bytes(raw_to_raw(image))
    return
    '''
    
    count = 0

    # Write output
    with open(args.outfile, "wb") as f:
        f.write(out_bytes)

        while True:
            try:
                count += 1
                if not count%one_percent:
                    print(" {}%".format(count/one_percent), end='\r')

                # Read next image
                image = read_bytes(it)
            
                # run image through P3
                raw = raw_to_raw(image)
                if 0:
                    img = Image.new('L', (32,32), color=255)
                    pixels = img.load()

                    #for i in range(32):
                    #    print(raw[i*32:(i+1)*32])

                    for i in range(32):
                        for j in range(32):
                            pixels[i, j] = 255-raw[32*i+j]
                    img.save("out.jpg")
                    return
                else:
                    out_bytes = bytes(raw)
                    f.write(out_bytes)

            except Exception as e:
                print(e)
                print("Done")
                break

    # Write output
    #with open(args.outfile, "wb") as f:
    #    f.write(out_bytes)
    

def save_images(img_num=0):
    folder = '/home/richard/research/p3/digits/mnist-data/32/'
    names = ["norm/"] 
    names = ["p3-1/", "p3-10/", "p3-20/", "pix-2/", "pix-4/", "pix-8/", "pix-16/"] # turns out the pix are rotated
    fname = "train-images-new.idx3-ubyte"

    for name in names:
        with open(folder+name+fname, 'rb') as f:
            data = f.read()
    
        it = data.__iter__()
    
        for i in range(4*4):
            _ = it.__next__()
        
        for i in range(img_num):
            _ = read_bytes(it, 32)

        raw = read_bytes(it, 32)

        img = []
        for i in range(32):
            for j in range(32):
                img.append(raw[j][i])

        s = ""
        for i,e in enumerate(img):
            s += "{:3} ".format(e)
            if i % 32 == 31:
                s += '\n'
        #print(s)

        img = Image.new('L', (32,32), color=255)
        pixels = img.load()

        for i in range(32):
            for j in range(32):
                pixels[i, j] = raw[j][i]
        img.save("/home/richard/svn/p3image/images/mnist-"+name[:-1]+".jpg")

        

if __name__ == "__main__":
    #main()
    save_images(5)


'''
def raw_to_bmp(data):
    """ Convert raw data to bmps
    """
    img = Image.new('L', (28,28))
    pixels = img.load()
    
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            pixels[j,i] = 255-data[i][j]
            
    img.save("test.bmp")

def bmp_to_jpg():
    """ Convert a jpeg at a static address to bmp
    """
    os.system('../jpeg-9a/cjpeg -dct int -q 100 -outfile out.jpg ./test.bmp')

def jpg_to_raw():
    """ Convert a jpg at a static address to raw data
    """ 
    im = Image.open("out.jpg")
    out = []

    for j in range(im.size[0]):
        for i in range(im.size[1]):
            out.append(im.getpixel((i,j))[0])
        
    return out
'''
