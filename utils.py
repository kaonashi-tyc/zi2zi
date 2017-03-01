# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import scipy.misc as misc
import numpy as np
from cStringIO import StringIO


def pad_seq(seq, batch_size):
    # pad the sequence to be the multiples of batch_size
    seq_len = len(seq)
    if seq_len % batch_size == 0:
        return seq
    padded = batch_size - (seq_len % batch_size)
    seq.extend(seq[:padded])
    return seq


def bytes_to_file(bytes_img):
    return StringIO(bytes_img)


def normalize_image(img):
    """
    Make image zero centered and in between (-1, 1)
    """
    normalized = (img / 127.5) - 1.
    return normalized


def read_split_image(img):
    mat = misc.imread(img).astype(np.float)
    side = int(mat.shape[1] / 2)
    assert side * 2 == mat.shape[1]
    img_A = mat[:, :side]  # target
    img_B = mat[:, side:]  # source

    return img_A, img_B


def augment_image(img, multiplier=1.05):
    # augment the image by:
    # 1) enlarge the image
    # 2) random crop the image back to its original size
    w, h, _ = img.shape
    nw = int(multiplier * w)
    nh = int(multiplier * h)
    enlarged = misc.imresize(img, [nw, nh])
    x = int(np.ceil(np.random.uniform(0.01, nw - w)))
    y = int(np.ceil(np.random.uniform(0.01, nh - h)))
    return enlarged[x:x + w, y:y + h]


def scale_back(images):
    return (images + 1.) / 2.


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = image

    return img
