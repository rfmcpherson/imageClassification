with open("mnist-data/train-images.idx3-ubyte", "rb") as f:
    data = f.read()

train_b = b''
valid_b = b''

# magic
train_b += bytes([0, 0, 8, 3])
valid_b += bytes([0, 0, 8, 3])

# count
train_b += bytes([0, 0, 195, 80])
valid_b += bytes([0,0, 39, 16])

# sizes
train_b += bytes([0, 0, 0, 28])
train_b += bytes([0, 0, 0, 28])
valid_b += bytes([0, 0, 0, 28])
valid_b += bytes([0, 0, 0, 28])

# images
data = data[4*4:]
train_data = data[:28*28*50000]
valid_data = data[28*28*50000:]
train_b += train_data
valid_b += valid_data


with open("mnist-data/train-images-new.idx3-ubyte","wb") as f:
    f.write(train_b)

with open("mnist-data/valid-images-new.idx3-ubyte","wb") as f:
    f.write(valid_b)





with open("mnist-data/train-labels.idx1-ubyte", "rb") as f:
    data = f.read()

train_b = b''
valid_b = b''

# magic
train_b += bytes([0, 0, 8, 1])
valid_b += bytes([0, 0, 8, 1])

# count
train_b += bytes([0, 0, 195, 80])
valid_b += bytes([0, 0, 39, 16])

# images
data = data[2*4:]
train_data = data[:50000]
valid_data = data[50000:]
train_b += train_data
valid_b += valid_data


with open("mnist-data/train-labels-new.idx1-ubyte","wb") as f:
    f.write(train_b)

with open("mnist-data/valid-labels-new.idx1-ubyte","wb") as f:
    f.write(valid_b)


