import tensorflow as tf
import tensorflow.keras as tk
import numpy
import nthmc, ftr, evolve, forcetrain

trajLength = 4.0
nstep = 18

conf = nthmc.Conf(nbatch=1024, nepoch=1, nstepEpoch=16*8192, nstepMixing=128, initDt=trajLength/nstep, stepPerTraj=nstep, trainDt=False, nthr=10, nthrIop=1, seed=7*11*13*23*131)
nthmc.setup(conf)
rng = tf.random.Generator.from_seed(conf.seed)
actionFun = lambda: nthmc.U1d2(beta=5.0, beta0=0.0, size=(64,64), transform=ftr.Ident(), nbatch=conf.nbatch, rng=rng.split()[0])
mcmcFun = lambda action: nthmc.Metropolis(conf, evolve.Omelyan2MN(conf, action))

i = 65536
x0 = tf.concat([
    tf.constant(numpy.load(f'configs/s_hmc_l64_b4/conf.b4.0.n128.l64_64.{i:05d}.npy'), dtype=tf.float64),
    tf.constant(numpy.load(f'configs/s_hmc_l64_b4/conf.b4.0.n128.l64_64.{2*i:05d}.npy'), dtype=tf.float64),
    tf.constant(numpy.load(f'configs/s_hmc_l64_b4/conf.b4.0.n128.l64_64.{3*i:05d}.npy'), dtype=tf.float64),
    tf.constant(numpy.load(f'configs/s_hmc_l64_b4/conf.b4.0.n128.l64_64.{4*i:05d}.npy'), dtype=tf.float64),
    tf.constant(numpy.load(f'configs/s_hmc_l64_b4/conf.b4.0.n128.l64_64.{5*i:05d}.npy'), dtype=tf.float64),
    tf.constant(numpy.load(f'configs/s_hmc_l64_b4/conf.b4.0.n128.l64_64.{6*i:05d}.npy'), dtype=tf.float64),
    tf.constant(numpy.load(f'configs/s_hmc_l64_b4/conf.b4.0.n128.l64_64.{7*i:05d}.npy'), dtype=tf.float64),
    tf.constant(numpy.load(f'configs/s_hmc_l64_b4/conf.b4.0.n128.l64_64.{8*i:05d}.npy'), dtype=tf.float64),
], axis=0)
forcetrain.runInfer(conf, actionFun, mcmcFun, x0=x0, saveFile='configs/s_hmc_l64_b5_a/conf', saveFreq=128*16)
