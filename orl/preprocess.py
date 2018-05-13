

def read_image(name, byteorder='>'):
    '''Reads a PGM image
    ORL images are 112x92 grayscale
    http://stackoverflow.com/questions/7368739/numpy-and-16-bit-pgm'''
    import numpy as np
    import re

    with open(name, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(buffer,
                         dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                         count=int(width)*int(height),
                         offset=len(header)).reshape((int(height), int(width)))


def display(im):
    '''Displays am image with pyplot'''
    from matplotlib import pyplot
    pyplot.imshow(im, pyplot.cm.gray)
    pyplot.show()


def process(mod_func):
    '''Reads ORL images.
    Splits the images into training, testing, and validation sets.
    Runs the images through mod_func
    Writes the sets and their labels in the MNIST format to data'''
    import random
    random.seed(1)

    # Images
    magic_1 = (2048 + 1).to_bytes(4,'big')
    magic_3 = (2048 + 3).to_bytes(4,'big')
    small_len = (2*40).to_bytes(4,'big')
    big_len = (8*40).to_bytes(4,'big')
    row_len = (92).to_bytes(4,'big')
    col_len = (112).to_bytes(4,'big')
    
    # init with mnist random numbers
    test = magic_3+small_len+row_len+col_len
    valid = magic_3+small_len+row_len+col_len
    train = magic_3+big_len+row_len+col_len
    testvalid_l = magic_1+small_len
    train_l = magic_1+big_len

    # Read and assign images from each folder
    for fol in range(1,41):
        print(fol)
        # Read images
        images = []
        for img in range(1,11):
            img_loc = "orl_faces/s{}/{}.pgm".format(fol,img)
            data = read_image(img_loc, "<")
            data = mod_func(data)
            images.append(data)

        # Shuffle and assign
        random.shuffle(images)
        for i in range(2):
            test += images.pop()
            testvalid_l += (fol-1).to_bytes(1,'big')
        #for i in range(2):
        #    valid += images.pop()
        for i in range(8):
            train += images.pop()
            train_l += (fol-1).to_bytes(1,'big')
        
    # Write
    with open("data/test-images.idx3-ubyte", 'wb') as f:
        f.write(test)
    #with open("data/valid-images.idx3-ubyte", 'wb') as f:
    #    f.write(valid)
    with open("data/train-images.idx3-ubyte", 'wb') as f:
        f.write(train)
    with open("data/test-labels.idx1-ubyte", 'wb') as f:
        f.write(testvalid_l)
    #with open("data/valid-labels.idx1-ubyte", 'wb') as f:
    #    f.write(testvalid_l)
    with open("data/train-labels.idx1-ubyte", 'wb') as f:
        f.write(train_l)


def mod_p3(a):
    '''Saves as image and then runs it through P3'''
    from io import BytesIO
    from PIL import Image
    import subprocess 

    # Save original image
    img = Image.fromarray(a)

    p = subprocess.Popen(
        '../../bin/cjpeg-1 -dct int -q 100 -grayscale'.split(),
        stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    img.save(p.stdin,"BMP")

    # Load and format new image
    raw = p.communicate()[0]
    file_jpgdata = BytesIO(raw)
    img = Image.open(file_jpgdata)

    data = list(img.getdata())
    return bytes(data)


def mod_null(a):
    from PIL import Image

    img = Image.fromarray(a)
    data = list(img.getdata())
    return bytes(data)


def mod_pixel(a):
    '''Pixelization of image'''
    from PIL import Image
    import numpy as np

    size = 2
    np.set_printoptions(edgeitems=4)

    img = Image.fromarray(a)
    #display(img)
    w, h = img.size # PIL is columns x rows (x,y)
    pixels = np.array(img.getdata(), dtype="int32")
    pixels = pixels.reshape(h,w) # numpy is rows x columns (y,x)

    # Ug, ok. Now let's average and pixelize
    for x in range(int(w/size)+1):
        for y in range(int(h/size)):
            ystart = y*size
            xstart = x*size

            average = np.sum(pixels[ystart:ystart+size,xstart:xstart+size])
            average = average*1/(size**2)
            average = int(average)

            pixels[ystart:ystart+size,xstart:xstart+size] = average

    #import scipy.misc
    #scipy.misc.imsave('x.png', pixels)    

    pixels = pixels.reshape(h*w)
    pixels = pixels.tolist()

    return bytes(pixels)


def read_bytes(it):
    """ 
    Read and return the next (dim x dim) image.
    """

    out = []
    for i in range(112):
        inner = []
        for j in range(92):
            inner.append(it.__next__())
        out.append(inner)
    return out


def save_images(img_num=0):
    from PIL import Image

    folder = './data/'
    names = ["norm/", "p3-1/", "p3-10/", "p3-20/", "pix-2/", "pix-4/", "pix-8/", "pix-16/"]
    #names = ["youtube-small/"]
    names = ["norm/"]
    fname = "train-images.idx3-ubyte"

    for name in names:
        with open(folder+name+fname, 'rb') as f:
            data = f.read()
    
        it = data.__iter__()
    
        for i in range(4*4):
            _ = it.__next__()
        
        for i in range(img_num):
            _ = read_bytes(it)

        raw = read_bytes(it)

        """
        img = []
        for i in range(32):
            for j in range(32):
                img.append(raw[j][i])

        s = ""
        for i,e in enumerate(img):
            s += "{:3} ".format(e)
            if i % 32 == 31:
                s += '\n'
        print(s)
        """

        img = Image.new('L', (92,112), color=255)
        pixels = img.load()

        for i in range(112):
            for j in range(92):
                pixels[j, i] = raw[i][j]
        #img.show()
        #img.save("test.png")
        img.save("/home/richard/svn/p3image/images/att-"+name[:-1]+".jpg")


def prepslideshow():
    from PIL import Image
    import random
    random.seed(1)


    test = []
    train = []
    for fol in range(1,41):
        #print(fol)

        images = []

        for img in range(1,11):
            images.append("orl_faces/s{}/{}.pgm".format(fol,img))

        # Shuffle and assign
        random.shuffle(images)
        for i in range(2):
            test.append(images.pop())
        for i in range(8):
            train.append(images.pop())

    white = Image.new('L', (1280, 720), color=255)

    i = 0
    for img_loc in train:
        data = read_image(img_loc, "<")
        
        img = Image.new('L', (1280, 720), color=255)
        pixels = img.load()

        for y in range(len(data)):
            for x in range(len(data[0])):
                pixels[594+x, 304+y] = data[y][x]

        img.save("youtube-images/train{}.png".format(i))
        i += 1
        white.save("youtube-images/train{}.png".format(i))
        i += 1

        
    i = 0
    for img_loc in test:
        data = read_image(img_loc, "<")
        
        img = Image.new('L', (1280, 720), color=255)
        pixels = img.load()

        for y in range(len(data)):
            for x in range(len(data[0])):
                pixels[594+x, 304+y] = data[y][x]

        img.save("youtube-images/test{}.png".format(i))
        i += 1
        white.save("youtube-images/test{}.png".format(i))
        i += 1


def readslideshow(fname):
    from PIL import Image

    left = int(1280/2-92)
    bottom = int(720/2-112)
    right = int(1280/2+92)
    top = int(720/2+112)

    img = Image.open(fname)
    img = img.crop((left, bottom, right, top))
    img = img.resize((92,112))
    data = list(img.getdata())
    for i in range(len(data)):
        data[i] = data[i][0]
    data = bytes(data)
    return data

def processslideshow():
    from PIL import Image

    # Images
    magic_1 = (2048 + 1).to_bytes(4,'big')
    magic_3 = (2048 + 3).to_bytes(4,'big')
    small_len = (5*2*40).to_bytes(4,'big')
    big_len = (5*8*40).to_bytes(4,'big')
    row_len = (92).to_bytes(4,'big')
    col_len = (112).to_bytes(4,'big')

    if 0:
        small_len = (2*40).to_bytes(4,'big')
        big_len = (8*40).to_bytes(4,'big')
        row_len = (92).to_bytes(4,'big')
        col_len = (112).to_bytes(4,'big')

    # init with mnist random numbers
    test = magic_3+small_len+row_len+col_len
    valid = magic_3+small_len+row_len+col_len
    train = magic_3+big_len+row_len+col_len
    test_l = magic_1+small_len
    train_l = magic_1+big_len


    # test images
    with open("data/test-images.idx3-ubyte",'wb') as f:

        # write metadata
        f.write(test)

        for i in range(2*40):
        
            for offset in range(13,18):
                # load image
                frame = 2*30*i+offset
                fname = "output/test/frame{:04}.png".format(frame)
                data = readslideshow(fname)

                # write image
                f.write(data)

    # #train images
    with open("data/train-images.idx3-ubyte",'wb') as f:

        # write metadata
        f.write(train)

        for i in range(8*40):
        
            for offset in range(13,18):
                # load image
                frame = 2*30*i+offset
                fname = "output/train/frame{:05}.png".format(frame)
                data = readslideshow(fname)

                # write image
                f.write(data)

    # test labels
    with open("data/test-labels.idx1-ubyte", 'wb') as f:
        f.write(test_l)
        
        for i in range(40):
            val = (i).to_bytes(1,'big')
            for j in range(2*5):
                f.write(val)

    # train labels
    with open("data/train-labels.idx1-ubyte", 'wb') as f:
        f.write(train_l)
        
        for i in range(40):
            val = (i).to_bytes(1,'big')
            for j in range(8*5):
                f.write(val)


def main():
    #process(mod_null)
    #process(mod_p3)
    #process(mod_pixel)
    save_images(3)
    #prepslideshow()
    #processslideshow()

if __name__ == "__main__":
    main()









"""
# Reads ORL images.
# Splits the images into training, testing, and validation sets.
# Writes the sets and their labels in the MNIST format to data/
def process_original():
    import random
    random.seed(1)

    # Images
    magic_1 = (2048 + 1).to_bytes(4,'big')
    magic_3 = (2048 + 3).to_bytes(4,'big')
    small_len = (2*40).to_bytes(4,'big')
    big_len = (6*40).to_bytes(4,'big')
    row_len = (92).to_bytes(4,'big')
    col_len = (112).to_bytes(4,'big')
    
    # init with mnist random numbers
    test = magic_3+small_len+row_len+col_len
    valid = magic_3+small_len+row_len+col_len
    train = magic_3+big_len+row_len+col_len
    testvalid_l = magic_1+small_len
    train_l = magic_1+big_len

    # Read and assign images from each folder
    for fol in range(1,41):
        print(fol)
        # Read images
        images = []
        for img in range(1,11):
            images.append(read_image("orl_faces/s{}/{}.pgm".format(fol,img), "<"))

        # Shuffle and assign
        random.shuffle(images)
        for i in range(2):
            test += b"".join(images.pop())
            testvalid_l += (fol-1).to_bytes(1,'big')
        for i in range(2):
            valid += b"".join(images.pop())
        for i in range(6):
            train += b"".join(images.pop())
            train_l += (fol-1).to_bytes(1,'big')
        
    # Write
    with open("data/test-images.idx3-ubyte", 'wb') as f:
        f.write(test)
    with open("data/valid-images.idx3-ubyte", 'wb') as f:
        f.write(valid)
    with open("data/train-images.idx3-ubyte", 'wb') as f:
        f.write(train)
    with open("data/test-labels.idx1-ubyte", 'wb') as f:
        f.write(testvalid_l)
    with open("data/valid-labels.idx1-ubyte", 'wb') as f:
        f.write(testvalid_l)
    with open("data/train-labels.idx1-ubyte", 'wb') as f:
        f.write(train_l)


# Preprocessing orl imgages with p3
# Converts them to p3 and then saves them in the MNIST format 
# Also splits them into the training, testing, and validation sets
def process_p3():
    import random
    random.seed(1)

    # Images
    magic_1 = (2048 + 1).to_bytes(4,'big')
    magic_3 = (2048 + 3).to_bytes(4,'big')
    small_len = (2*40).to_bytes(4,'big')
    big_len = (6*40).to_bytes(4,'big')
    row_len = (92).to_bytes(4,'big')
    col_len = (112).to_bytes(4,'big')
    
    # init with mnist random numbers
    test = magic_3+small_len+row_len+col_len
    valid = magic_3+small_len+row_len+col_len
    train = magic_3+big_len+row_len+col_len
    testvalid_l = magic_1+small_len
    train_l = magic_1+big_len

    # Read and assign images from each folder
    for fol in range(1,41):
        print(fol)
        # Read images
        images = []
        for img in range(1,11):
            img = p3(read_image("orl_faces/s{}/{}.pgm".format(fol,img), "<"))
            images.append(img)

        # Shuffle and assign
        random.shuffle(images)
        for i in range(2):
            test += images.pop()
            testvalid_l += (fol-1).to_bytes(1,'big')
        for i in range(2):
            valid += images.pop()
        for i in range(6):
            train += images.pop()
            train_l += (fol-1).to_bytes(1,'big')
        
    # Write
    with open("data/test-images.idx3-ubyte", 'wb') as f:
        f.write(test)
    with open("data/valid-images.idx3-ubyte", 'wb') as f:
        f.write(valid)
    with open("data/train-images.idx3-ubyte", 'wb') as f:
        f.write(train)
    with open("data/test-labels.idx1-ubyte", 'wb') as f:
        f.write(testvalid_l)
    with open("data/valid-labels.idx1-ubyte", 'wb') as f:
        f.write(testvalid_l)
    with open("data/train-labels.idx1-ubyte", 'wb') as f:
        f.write(train_l)
"""
