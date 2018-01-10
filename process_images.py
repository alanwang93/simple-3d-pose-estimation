#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Yifan WANG <yifanwang1993@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Process an image dataset to produce HDF5 annotations for stacked hourglass model

Original HDF5 file structure:
<HDF5 dataset "center": shape (11731, 2), type "<f8">
<HDF5 dataset "imgname": shape (11731,), type "|S13">
<HDF5 dataset "index": shape (11731,), type "<i8">
<HDF5 dataset "name": shape (), type "|O">
<HDF5 dataset "normalize": shape (11731,), type "<i8">
<HDF5 dataset "part": shape (11731, 16, 2), type "<f8">
<HDF5 dataset "person": shape (11731,), type "<i8">
<HDF5 dataset "scale": shape (11731,), type "<f8">
<HDF5 dataset "torsoangle": shape (11731,), type "<f8">
<HDF5 dataset "visible": shape (11731, 16), type "<f8">
"""
from os import listdir
from os.path import isfile, join
import h5py, argparse
import numpy as np
from PIL import Image

def main(args):
    path = args.path
    name = args.name
    # produce image list
    img_file = open('stacked_hourglass/annot/' + name + '_images.txt', 'w')

    filenames = [f for f in listdir(path) if isfile(join(path, f))]
    n = len(filenames)
    for f in filenames:
        img_file.write(f + '\n')
    # produce h5 file
    h5 = h5py.File('stacked_hourglass/annot/' + name + '.h5', 'w')
    # tags = ['part','center','scale','normalize','torsoangle','visible']
    part = h5.create_dataset("part", (n,16,2), dtype='i8')
    center = h5.create_dataset("center", (n,2), dtype='f8')
    scale = h5.create_dataset("scale", (n,), dtype='f8')
    imgname = h5.create_dataset("imgname", (n,), dtype='S13')
    index = h5.create_dataset("index", (n,), dtype='i8')
    normalize = h5.create_dataset("normalize", (n,), dtype='i8')
    torsoangle = h5.create_dataset("torsoangle", (n,), dtype='f8')
    visible = h5.create_dataset("visible", (n,16), dtype='f8')

    for i, f in enumerate(filenames):
        print(type(np.string_(f)))
        imgname[i] = np.string_(f)
        index[i] = i
        scale[i] = 1.
        normalize[i] = -1
        torsoangle[i] = 0.
        with Image.open(join(path, f)) as img:
            w, h = img.size
            center[i] = [w/2.,h/2.]

    h5.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default=None ,
                        help='Directory of images')
    parser.add_argument('--name', type=str, default="mytest")
    args = parser.parse_args()
    main(args)
