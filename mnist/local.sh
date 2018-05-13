#!/bin/sh
th torch-scripts/train-on-mnist.lua \
--save out_local \
-b 1000 \
--top 5 \
--progress \
--train /u/richard/p3/class/mnist/mnist-data/norm/train \
--valid /u/richard/p3/class/mnist/mnist-data/norm/valid \
--test /u/richard/p3/class/mnist/mnist-data/norm/t10k
