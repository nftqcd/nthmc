import tensorflow as tf
import tensorflow.keras as tk
import numpy
import nthmc, ftr, evolve, forcetrain

trajLength = 4.0
nstep = 18

conf = nthmc.Conf(nbatch=1024, nepoch=1, nstepEpoch=16*8192, nstepMixing=128, initDt=trajLength/nstep, stepPerTraj=nstep, trainDt=False, nthr=10, nthrIop=1, seed=7*11*13*23*131*9)
nthmc.setup(conf)
rng = tf.random.Generator.from_seed(conf.seed)
actionFun = lambda: nthmc.U1d2(beta=6.0, beta0=0.0, size=(64,64), transform=ftr.Ident(), nbatch=conf.nbatch, rng=rng.split()[0])
mcmcFun = lambda action: nthmc.Metropolis(conf, evolve.Omelyan2MN(conf, action))

x0 = tf.constant(numpy.load(f'configs/s_hmc_l64_b5_a/conf.b5.0.n1024.l64_64.{1*32768}.npy'), dtype=tf.float64)
forcetrain.runInfer(conf, actionFun, mcmcFun, x0=x0, saveFile='configs/s_hmc_l64_b6_a/conf', saveFreq=512*16)
