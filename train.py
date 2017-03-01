# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import argparse

from model import UNet

parser = argparse.ArgumentParser(description='')
parser.add_argument('--experiment_dir', dest='experiment_dir', required=True,
                    help='experiment directory, data, samples,checkpoints,etc')
parser.add_argument('--experiment_id', dest='experiment_id', type=int, default=0,
                    help='sequence id for the experiments you prepare to run')
parser.add_argument('--image_size', dest='image_size', type=int, default=256,
                    help="size of your input and output image")
parser.add_argument('--embedding_num', dest='embedding_num', type=int, default=40,
                    help="number for distinct embeddings")
parser.add_argument('--embedding_dim', dest='embedding_dim', type=int, default=128, help="dimension for embedding")
parser.add_argument('--epoch', dest='epoch', type=int, default=100, help='number of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='number of examples in batch')
parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='initial learning rate for adam')
parser.add_argument('--schedule', dest='schedule', type=int, default=10, help='number of epochs to half learning rate')
parser.add_argument('--resume', dest='resume', type=bool, default=True, help='resume from previous training')
parser.add_argument('--freeze_encoder', dest='freeze_encoder', type=bool, default=False,
                    help="freeze encoder weights during training")
parser.add_argument('--fine_tune', dest='fine_tune', type=int, default=0,
                    help='specific labels id to be fine tuned')

args = parser.parse_args()


def main(_):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        model = UNet(args.experiment_dir, batch_size=args.batch_size, experiment_id=args.experiment_id,
                     input_width=args.image_size, output_width=args.image_size, embedding_num=args.embedding_num,
                     embedding_dim=args.embedding_dim)
        model.register_session(sess)
        model.build_model()
        model.train(lr=args.lr, epoch=args.epoch, resume=args.resume,
                    schedule=args.schedule, freeze_encoder=args.freeze_encoder, fine_tune=args.fine_tune)


if __name__ == '__main__':
    tf.app.run()
