# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import argparse
import sys
import numpy as np
import os
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import json
import collections

reload(sys)
sys.setdefaultencoding("utf-8")

CN_CHARSET = None
CN_T_CHARSET = None
JP_CHARSET = None
KR_CHARSET = None

DEFAULT_CHARSET = "./charset/cjk.json"


def load_global_charset():
    global CN_CHARSET, JP_CHARSET, KR_CHARSET, CN_T_CHARSET
    cjk = json.load(open(DEFAULT_CHARSET))
    CN_CHARSET = cjk["gbk"]
    JP_CHARSET = cjk["jp"]
    KR_CHARSET = cjk["kr"]
    CN_T_CHARSET = cjk["gb2312_t"]


def draw_single_char(ch, font, canvas_size, x_offset, y_offset):
    img = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((x_offset, y_offset), ch, (0, 0, 0), font=font)
    return img


def draw_example(ch, src_font, dst_font, canvas_size, x_offset, y_offset, filter_hashes):
    dst_img = draw_single_char(ch, dst_font, canvas_size, x_offset, y_offset)
    # check the filter example in the hashes or not
    dst_hash = hash(dst_img.tobytes())
    if dst_hash in filter_hashes:
        return None
    src_img = draw_single_char(ch, src_font, canvas_size, x_offset, y_offset)
    example_img = Image.new("RGB", (canvas_size * 2, canvas_size), (255, 255, 255))
    example_img.paste(dst_img, (0, 0))
    example_img.paste(src_img, (canvas_size, 0))
    return example_img


def filter_recurring_hash(charset, font, canvas_size, x_offset, y_offset):
    """ Some characters are missing in a given font, filter them
    by checking the recurring hashes
    """
    _charset = charset[:]
    np.random.shuffle(_charset)
    sample = _charset[:2000]
    hash_count = collections.defaultdict(int)
    for c in sample:
        img = draw_single_char(c, font, canvas_size, x_offset, y_offset)
        hash_count[hash(img.tobytes())] += 1
    recurring_hashes = filter(lambda d: d[1] > 2, hash_count.items())
    return [rh[0] for rh in recurring_hashes]


def font2img(src, dst, charset, char_size, canvas_size,
             x_offset, y_offset, sample_count, sample_dir, label=0, filter_by_hash=True):
    src_font = ImageFont.truetype(src, size=char_size)
    dst_font = ImageFont.truetype(dst, size=char_size)

    filter_hashes = set()
    if filter_by_hash:
        filter_hashes = set(filter_recurring_hash(charset, dst_font, canvas_size, x_offset, y_offset))
        print("filter hashes -> %s" % (",".join([str(h) for h in filter_hashes])))

    count = 0

    for c in charset:
        if count == sample_count:
            break
        e = draw_example(c, src_font, dst_font, canvas_size, x_offset, y_offset, filter_hashes)
        if e:
            e.save(os.path.join(sample_dir, "%d_%04d.jpg" % (label, count)))
            count += 1
            if count % 100 == 0:
                print("processed %d chars" % count)


load_global_charset()
parser = argparse.ArgumentParser(description='Convert font to images')
parser.add_argument('--src_font', dest='src_font', required=True, help='path of the source font')
parser.add_argument('--dst_font', dest='dst_font', required=True, help='path of the target font')
parser.add_argument('--filter', dest='filter', type=int, default=0, help='filter recurring characters')
parser.add_argument('--charset', dest='charset', type=str, default='CN',
                    help='charset, can be either: CN, JP, KR or a one line file')
parser.add_argument('--shuffle', dest='shuffle', type=int, default=0, help='shuffle a charset before processings')
parser.add_argument('--char_size', dest='char_size', type=int, default=150, help='character size')
parser.add_argument('--canvas_size', dest='canvas_size', type=int, default=256, help='canvas size')
parser.add_argument('--x_offset', dest='x_offset', type=int, default=20, help='x offset')
parser.add_argument('--y_offset', dest='y_offset', type=int, default=20, help='y_offset')
parser.add_argument('--sample_count', dest='sample_count', type=int, default=1000, help='number of characters to draw')
parser.add_argument('--sample_dir', dest='sample_dir', help='directory to save examples')
parser.add_argument('--label', dest='label', type=int, default=0, help='label as the prefix of examples')

args = parser.parse_args()

if __name__ == "__main__":
    if args.charset in ['CN', 'JP', 'KR', 'CN_T']:
        charset = locals().get("%s_CHARSET" % args.charset)
    else:
        charset = [c for c in open(args.charset).readline()[:-1].decode("utf-8")]
    if args.shuffle:
        np.random.shuffle(charset)
    font2img(args.src_font, args.dst_font, charset, args.char_size,
             args.canvas_size, args.x_offset, args.y_offset,
             args.sample_count, args.sample_dir, args.label, args.filter)
