import tensorflow as tf
import tensorflow.keras as tk
import math, numpy
import nthmc, ftr, evolve, forcetrain

trajLength = 4.0
nstep = 25

conf = nthmc.Conf(nbatch=32, nepoch=3, nstepEpoch=1024, nstepMixing=128, initDt=trajLength/nstep, stepPerTraj=nstep, trainDt=False, nthr=10, nthrIop=1, seed=7*11*13*7*15*31*1111)
nthmc.setup(conf)

op0 = (((1,2,-1,-2), (1,-2,-1,2)),)
op1 = (((2,-1,-2,1), (2,1,-2,-1)),)
conv = lambda: ftr.Scalar(output=1, init=math.tan(math.pi*0.3))     # coeff = atan(x)/pi  c.f. GenericStoutSmear.beta
transform = lambda: ftr.TransformChain([
    ftr.GenericStoutSmear(((0,0),(2,2)), op0, [], conv()),
    ftr.GenericStoutSmear(((0,1),(2,2)), op0, [], conv()),
    ftr.GenericStoutSmear(((1,0),(2,2)), op0, [], conv()),
    ftr.GenericStoutSmear(((1,1),(2,2)), op0, [], conv()),
    ftr.GenericStoutSmear(((0,0),(2,2)), op1, [], conv()),
    ftr.GenericStoutSmear(((1,0),(2,2)), op1, [], conv()),
    ftr.GenericStoutSmear(((0,1),(2,2)), op1, [], conv()),
    ftr.GenericStoutSmear(((1,1),(2,2)), op1, [], conv()),
    ftr.GenericStoutSmear(((0,0),(2,2)), op0, [], conv()),
    ftr.GenericStoutSmear(((0,1),(2,2)), op0, [], conv()),
    ftr.GenericStoutSmear(((1,0),(2,2)), op0, [], conv()),
    ftr.GenericStoutSmear(((1,1),(2,2)), op0, [], conv()),
    ftr.GenericStoutSmear(((0,0),(2,2)), op1, [], conv()),
    ftr.GenericStoutSmear(((1,0),(2,2)), op1, [], conv()),
    ftr.GenericStoutSmear(((0,1),(2,2)), op1, [], conv()),
    ftr.GenericStoutSmear(((1,1),(2,2)), op1, [], conv()),
])
rng = tf.random.Generator.from_seed(conf.seed)
actionFun = lambda: nthmc.U1d2(beta=7.0, beta0=4.0, size=(64,64), transform=transform(), nbatch=conf.nbatch, rng=rng.split()[0])
mcmcFun = lambda action: nthmc.Metropolis(conf, evolve.Omelyan2MN(conf, action))

x0 = tf.constant(numpy.load('configs/s_hmc_l64_b5_a/conf.b5.0.n1024.l64_64.08192.npy')[0:32], dtype=tf.float64)
forcetrain.runInfer(conf, actionFun, mcmcFun, x0=x0, saveFile='configs/s_fthmc_l64_b6_c3/conf')
