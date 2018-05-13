#!/bin/sh

# Vairables
pwd=$(pwd)
rdir="temp/"

# Create top level directory
mkdir -p $rdir

th torch-scripts/train-on-mnist.lua \
--progress \
-b 10 \
-r 0.001 \
--rdir $rdir \
--model convnetbasic \
--train /u/richard/p3/class/orl/data/norm/train \
--test /u/richard/p3/class/orl/data/norm/test \
--valid /u/richard/p3/class/orl/data/norm/valid
