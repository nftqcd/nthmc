import tensorflow as tf
import tensorflow.keras as tk
import nthmc, ftr, evolve, forcetrain

trajLength = 4.0
nstep = 10
fac = (70./64.)**2

conf = nthmc.Conf(nbatch=128, nepoch=13, nstepEpoch=8192, nstepMixing=128, initDt=trajLength/nstep, stepPerTraj=nstep, trainDt=False, nthr=10, nthrIop=1)
nthmc.setup(conf)
rng = tf.random.Generator.from_seed(conf.seed)
actionFun = lambda: nthmc.U1d2(beta=13.*fac, beta0=0.0, size=(70,70), transform=ftr.Ident(), nbatch=conf.nbatch, rng=rng.split()[0])
mcmcFun = lambda action: nthmc.Metropolis(conf, evolve.Omelyan2MN(conf, action))

forcetrain.runInfer(conf, actionFun, mcmcFun, saveFile='conf.s_hmc_l70_params')
