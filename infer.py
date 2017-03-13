# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import os
import argparse
import imageio
import glob
import scipy.misc as misc
from model.unet import UNet


def compile_frames_to_gif(frame_dir, gif_file):
    frames = sorted(glob.glob(os.path.join(frame_dir, "*.png")))
    print(frames)
    images = [misc.imresize(imageio.imread(f), interp='nearest', size=0.25) for f in frames]
    imageio.mimsave(gif_file, images, duration=0.1)
    return gif_file


parser = argparse.ArgumentParser(description='')
parser.add_argument('--experiment_dir', dest='experiment_dir', required=True,
                    help='experiment directory, data, samples,checkpoints,etc')
parser.add_argument('--experiment_id', dest='experiment_id', type=int, default=0,
                    help='sequence id for the experiments you prepare to run')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='number of examples in batch')
parser.add_argument('--source_obj', dest='source_obj', type=str, required=True, help='the source images for inference')
parser.add_argument('--embedding_ids', default='embedding_ids', type=str, help='embeddings involved')
parser.add_argument('--save_dir', default='save_dir', type=str, help='path to save inferred images')
parser.add_argument('--inst_norm', dest='inst_norm', type=bool, default=False,
                    help='use conditional instance normalization in your model')
parser.add_argument('--interpolate', dest='interpolate', type=bool, default=False,
                    help='interpolate between different embedding vectors')
parser.add_argument('--steps', dest='steps', type=int, default=10, help='interpolation steps in between vectors')
parser.add_argument('--output_gif', dest='output_gif', type=str, default=None, help='output name transition gif')
args = parser.parse_args()


def main(_):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        model = UNet(args.experiment_dir, batch_size=args.batch_size, experiment_id=args.experiment_id)
        model.register_session(sess)
        model.build_model(is_training=False, inst_norm=args.inst_norm)
        embedding_ids = [int(i) for i in args.embedding_ids.split(",")]
        if not args.interpolate:
            if len(embedding_ids) == 1:
                embedding_ids = embedding_ids[0]
            model.infer(source_obj=args.source_obj, embedding_ids=embedding_ids, save_dir=args.save_dir)
        else:
            if len(embedding_ids) != 2:
                raise Exception("for interpolation, len(embedding_ids) has to equal 2")
            model.interpolate(source_obj=args.source_obj, between=embedding_ids, save_dir=args.save_dir,
                              steps=args.steps)
            if args.output_gif:
                gif_path = os.path.join(args.save_dir, args.output_gif)
                compile_frames_to_gif(args.save_dir, gif_path)
                print("gif saved at %s" % gif_path)


if __name__ == '__main__':
    tf.app.run()
