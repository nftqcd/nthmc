import argparse

parser = argparse.ArgumentParser(description="Read a batch of lattice configs and check plaquette and topology values.")
parser.add_argument("file", help="file name")
args = parser.parse_args()

import nthmc, ftr
import tensorflow as tf
import numpy

x = tf.constant(numpy.load(args.file, allow_pickle=False, fix_imports=False), dtype=tf.float64)
tf.print('shape:',x.shape,summarize=-1)

act = nthmc.U1d2(ftr.Ident(), x.shape[0], rng=None)
tf.print('plaq:',act.plaquette(x),summarize=-1)
tf.print('topo:',act.topoCharge(x),summarize=-1)
