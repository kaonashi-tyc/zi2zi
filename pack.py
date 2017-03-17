# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import glob
import os
import cPickle as pickle
import random


def pickle_examples(idx_paths, train_path, val_path, train_val_split=0.066666):
    """
    Compile a list of examples into pickled format, so during
    the training, all io will happen in memory
    """
    with open(train_path, 'wb') as ft:
        with open(val_path, 'wb') as fv:
            for idx, path in idx_paths:
                imgs = glob.glob(os.path.join(path, "*.jpg"))
                for img in imgs:
                    with open(img, 'rb') as f:
                        print("img %s" % img, idx)
                        img_bytes = f.read()
                        r = random.random()
                        example = (idx, img_bytes)
                        if r < train_val_split:
                            pickle.dump(example, fv)
                        else:
                            pickle.dump(example, ft)


if __name__ == "__main__":
    """
    idx_paths = list()
    idx_paths.append((1, "/Users/ytian/Documents/fonts_imgs/cn/01-41/01"))
    idx_paths.append((2, "/Users/ytian/Documents/fonts_imgs/cn/01-41/03"))
    idx_paths.append((3, "/Users/ytian/Documents/fonts_imgs/cn/01-41/04"))
    idx_paths.append((4, "/Users/ytian/Documents/fonts_imgs/cn/01-41/05"))
    idx_paths.append((5, "/Users/ytian/Documents/fonts_imgs/cn/01-41/07"))
    idx_paths.append((6, "/Users/ytian/Documents/fonts_imgs/cn/01-41/10"))
    idx_paths.append((7, "/Users/ytian/Documents/fonts_imgs/cn/01-41/24"))
    idx_paths.append((8, "/Users/ytian/Documents/fonts_imgs/cn/01-41/25"))
    idx_paths.append((9, "/Users/ytian/Documents/fonts_imgs/cn/01-41/26"))
    idx_paths.append((10, "/Users/ytian/Documents/fonts_imgs/cn/01-41/23"))
    idx_paths.append((11, "/Users/ytian/Documents/fonts_imgs/cn/01-41/27"))
    idx_paths.append((12, "/Users/ytian/Documents/fonts_imgs/cn/01-41/28"))
    idx_paths.append((13, "/Users/ytian/Documents/fonts_imgs/cn/01-41/32"))
    idx_paths.append((14, "/Users/ytian/Documents/fonts_imgs/cn/01-41/34"))
    idx_paths.append((15, "/Users/ytian/Documents/fonts_imgs/jp/50-85/50"))
    idx_paths.append((16, "/Users/ytian/Documents/fonts_imgs/jp/50-85/51"))
    idx_paths.append((17, "/Users/ytian/Documents/fonts_imgs/jp/50-85/86"))
    idx_paths.append((18, "/Users/ytian/Documents/fonts_imgs/jp/50-85/87"))
    idx_paths.append((19, "/Users/ytian/Documents/fonts_imgs/jp/50-85/88"))
    idx_paths.append((20, "/Users/ytian/Documents/fonts_imgs/jp/50-85/89"))
    idx_paths.append((21, "/Users/ytian/Documents/fonts_imgs/jp/50-85/58"))
    idx_paths.append((22, "/Users/ytian/Documents/fonts_imgs/jp/50-85/52"))
    idx_paths.append((23, "/Users/ytian/Documents/fonts_imgs/jp/50-85/67"))
    idx_paths.append((24, "/Users/ytian/Documents/fonts_imgs/jp/50-85/91"))
    idx_paths.append((25, "/Users/ytian/Documents/fonts_imgs/jp/50-85/66"))
    idx_paths.append((26, "/Users/ytian/Documents/fonts_imgs/jp/50-85/77"))
    idx_paths.append((27, "/Users/ytian/Documents/fonts_imgs/jp/50-85/90"))
    pickle_examples(idx_paths, "datasets/train.obj", "datasets/val.obj", train_val_split=0.5)
    """
    idx_paths = list()
    idx_paths.append((1, "/Users/ytian/Documents/fonts_imgs/kr_val"))
    #idx_paths.append((3, "/Users/ytian/Documents/fonts_imgs/cn/01-41/04"))
    #idx_paths.append((4, "/Users/ytian/Documents/fonts_imgs/cn/01-41/05"))
    #idx_paths.append((8, "/Users/ytian/Documents/fonts_imgs/cn/01-41/25"))
    #idx_paths.append((13, "/Users/ytian/Documents/fonts_imgs/cn/01-41/32"))
    pickle_examples(idx_paths, "datasets/kr.obj", "datasets/val.obj", train_val_split=0.08)
    # """