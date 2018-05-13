import argparse
from io import BytesIO
from PIL import Image
import numpy as np
import os
import scipy.misc
import struct
import subprocess

def raw_to_raw_pixel(data):
    size = 16

    # Make original bmp
    img = Image.new('RGB', (32,32))
    pixels = img.load()
    for x in range(32):
        for y in range(32):
            pixels[x,y] = tuple(data[y][x])

    pixels = np.array(img)
    #scipy.misc.imsave('orig.jpg', pixels)

    for x in range(int(32/size)):
        for y in range(int(32/size)):
            ystart = y*size
            xstart = x*size
            
            red = np.sum(pixels[ystart:ystart+size,xstart:xstart+size,0])
            blue = np.sum(pixels[ystart:ystart+size,xstart:xstart+size,1])
            green = np.sum(pixels[ystart:ystart+size,xstart:xstart+size,2])

            ared = int(red*1/(size**2))
            ablue = int(blue*1/(size**2))
            agreen = int(green*1/(size**2))

            #print(pixels[ystart:ystart+size,xstart:xstart+size,0])
            #print(pixels[ystart:ystart+size,xstart:xstart+size,1])
            #print(pixels[ystart:ystart+size,xstart:xstart+size,2])
            #print(ared, ablue, agreen)
            
            pixels[ystart:ystart+size,xstart:xstart+size] = [ared, ablue, agreen]
            #print(pixels[0,0]) 
            #print(pixels[0,1]) 
            #print(pixels[1,0]) 
            #print(pixels[1,1]) 
            #return


            '''
            average = np.sum(pixels[ystart:ystart+size,xstart:xstart+size])
            average = average*1/(size**2)
            average = int(average)
            pixels[ystart:ystart+size,xstart:xstart+size] = average
            '''

    #pixels = pixels.reshape((32,32,3))
    #scipy.misc.imsave('outfile.jpg', pixels)

    #np.set_printoptions(edgeitems=16)
    #np.set_printoptions(linewidth=200)
    #print(pixels)

    #return None

    out = []
    for c in range(3):
        for y in range(32):
            for x in range(32):
                out.append(pixels[x,y][c])

    #out = pixels.reshape((3*32*32))

    return out

def raw_to_raw_p3(data):
    # Set up cjpeg command
    p = subprocess.Popen(
        '../jpeg-9a/cjpeg -dct int -q 100 '.split(), 
        stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    # Make original bmp
    img = Image.new('RGB', (32,32))
    pixels = img.load()
    for x in range(32):
        for y in range(32):
            pixels[x,y] = tuple(data[y][x])

    # Run the cjpeg command 
    img.save(p.stdin,'BMP')
    
    # Get the new pixel values
    raw = p.communicate()[0]
    file_jpgdata = BytesIO(raw)
    img = Image.open(file_jpgdata)
    img.save("temp.jpg")
    img = list(img.getdata())

    out = []

    for c in range(3):
        for i in range(len(img)):
            out.append(img[i][c])

    return out


def raw_to_raw_norm(data):
    # Make original bmp
    img = Image.new('RGB', (32,32))
    pixels = img.load()
    for x in range(32):
        for y in range(32):
            pixels[x,y] = tuple(data[y][x])

    out = []

    for c in range(3):
        for y in range(32):
            for x in range(32):
                out.append(pixels[x,y][c])

    return out


def raw_to_raw(data):
    return raw_to_raw_pixel(data)


def read_image(it):
    label = it.__next__()

    image = []
    
    for j in range(32):
        row = []
        for i in range(32):
            row.append([])
        image.append(row)

    for c in range(3):
        for y in range(32):
            for x in range(32):
                image[y][x].append(it.__next__())

    return label, image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("outfolder", help="unique part of outfile")
    args = parser.parse_args()

    dataset_folder = "cifar-10-batches-bin"
    trainsets = ["data_batch_1.bin", "data_batch_2.bin", "data_batch_3.bin", 
                "data_batch_4.bin", "data_batch_5.bin"]
    testset =   ["test_batch.bin"]
    
    for set_num, name in enumerate(trainsets):
        print("Training set {}/5".format(set_num+1))

        # Read file
        with open("{}/{}".format(dataset_folder, name),"rb") as f:
            data = f.read()

        # Make iterator
        it = data.__iter__()

        with open("{}/train_data_{}.raw".format(args.outfolder, set_num+1), "wb") as images_f:
            with open("{}/train_labels_{}.raw".format(args.outfolder, set_num+1), "wb") as labels_f:

                # Modify data
                for image_num in range(10000):
                    label, image = read_image(it)
                    image = raw_to_raw(image)
                    images_f.write(bytes(image))
                    labels_f.write(bytes([label]))
                    print("    {}%".format(int(image_num/100)), end='\r')


    '''
    for name in [trainsets[-1]]:
        print("Validation set 1/1")

        # Read file
        with open("{}/{}".format(dataset_folder, name),"rb") as f:
            data = f.read()

        # Make iterator
        it = data.__iter__()

        with open("{}/valid_data.raw".format(args.outfolder), "wb") as images_f:
            with open("{}/valid_labels.raw".format(args.outfolder), "wb") as labels_f:

                # Modify data
                for image_num in range(10000):
                    label, image = read_image(it)
                    image = raw_to_raw(image)
                    images_f.write(bytes(image))
                    labels_f.write(bytes([label]))
                    print("    {}%".format(int(image_num/100)), end='\r')
    '''

    for name in testset:
        print("Test set 1/1")

        # Read file
        with open("{}/{}".format(dataset_folder, name),"rb") as f:
            data = f.read()

        # Make iterator
        it = data.__iter__()

        with open("{}/test_data.raw".format(args.outfolder), "wb") as images_f:
            with open("{}/test_labels.raw".format(args.outfolder), "wb") as labels_f:

                # Modify data
                for image_num in range(10000):
                    label, image = read_image(it)
                    image = raw_to_raw(image)
                    images_f.write(bytes(image))
                    labels_f.write(bytes([label]))
                    print("    {}%".format(int(image_num/100)), end='\r')


def read_bytes(it):
    image = []
    
    for j in range(32):
        row = []
        for i in range(32):
            row.append([])
        image.append(row)

    for c in range(3):
        for y in range(32):
            for x in range(32):
                image[y][x].append(it.__next__())

    return image

def save_images(img_num):
    folder = "/home/richard/research/p3/cifar/cifar-data/"
    names = ["norm/"] 
    names = ["norm/", "p3-1/", "p3-10/", "p3-20/"]
    names = ["pix-2/"]
    #names = ["pix-2/", "pix-4/", "pix-8/", "pix-16/"]
    fname = "train_data_1.raw"

    for name in names:
        # Read file
        with open(folder+name+fname,"rb") as f:
            data = f.read()

        # Make iterator
        it = data.__iter__()
        
        for i in range(img_num):
            _ = read_bytes(it)

        raw = read_bytes(it)
        #raw = raw_to_raw_pixel(raw)
        

        img = Image.new('RGB', (32,32))
        pixels = img.load()

        for i in range(32):
            for j in range(32):
                pixels[j, i] = tuple(raw[i][j])
        img.show()
        #img.save("/home/richard/svn/p3image/images/cifar-"+name[:-1]+".jpg")
        #img.save("temp.jpg")

if __name__ == "__main__":
    #main()
    save_images(1)
